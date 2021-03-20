import sys
from collections import OrderedDict
from typing import List

from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

import torch.nn as nn
import torch

from asdl.asdl_ast import AbstractSyntaxTree, AbstractSyntaxNode, SyntaxToken, chain
from edit_model.embedder import EmbeddingTable
from edit_model.gnn import AdjacencyList, GatedGraphNeuralNetwork
from edit_model import nn_utils


class GraphChangeEncoder(nn.Module):
    change_tags = ('ADD', 'DEL', 'SAME', 'REPLACE')
    change_tag2id = {tag: i for i, tag in enumerate(change_tags)}

    def __init__(self, change_vector_dim, layer_time_steps, dropout, syntax_tree_embedder, tag_embed_size=32,
                 gnn_use_bias_for_message_linear=True, master_node_option=None, connections=None):
        super(GraphChangeEncoder, self).__init__()

        self.tag_embed_size = tag_embed_size
        self.syntax_tree_embedder = syntax_tree_embedder

        assert master_node_option in (None, 'single_node', 'double_node')
        self.master_node_option = master_node_option

        if connections is None:
            connections = ["top_down", "bottom_up"]
        assert connections in (["top_down", "bottom_up"], ["top_down", "bottom_up", "next_sibling", "prev_sibling"])
        self.top_down_connection = 'top_down' in connections
        self.bottom_up_connection = 'bottom_up' in connections
        self.next_sibling_connection = 'next_sibling' in connections
        self.prev_sibling_connection = 'prev_sibling' in connections

        num_edge_types = 8 if master_node_option == 'None' is None else 12
        if self.next_sibling_connection and self.prev_sibling_connection:
            num_edge_types += 2

        self.change_tag_embedding = nn.Embedding(len(self.change_tags), tag_embed_size)
        self.gnn = GatedGraphNeuralNetwork(hidden_size=change_vector_dim // 2,
                                           num_edge_types=num_edge_types,
                                           layer_timesteps=layer_time_steps,
                                           residual_connections={1: [0]},
                                           state_to_message_dropout=dropout,
                                           rnn_dropout=dropout,
                                           use_bias_for_message_linear=gnn_use_bias_for_message_linear)

        self.layer1_readout_weight = nn.Linear(self.gnn.hidden_size, 1)
        self.layer2_readout_weight = nn.Linear(self.gnn.hidden_size, 1)

    @property
    def device(self):
        return self.syntax_tree_embedder.device

    def get_batch_adjacency_lists(self, examples):
        example_node2batch_graph_node = OrderedDict()
        ast_adj_list = []
        reversed_ast_adj_list = []

        if self.next_sibling_connection and self.prev_sibling_connection:
            next_sibling_adj_list = []
            prev_sibling_adj_list = []

        same_edges_src2tgt_list = []
        same_edges_tgt2src_list = []
        replace_edges_src2tgt_list = []
        replace_edges_tgt2src_list = []

        terminal_tokens_adj_list = []
        reversed_terminal_tokens_adj_list = []

        def _build_batch_graph(_example_id, _prev_code_ast, _updated_code_ast):
            for node_source, syntax_tree in [(0, _prev_code_ast), (1, _updated_code_ast)]:
                for node_s_id, node_t_id in syntax_tree.adjacency_list:
                    # source -> target
                    node_s_batch_id = example_node2batch_graph_node.setdefault((_example_id, node_source, node_s_id),
                                                                         len(example_node2batch_graph_node))
                    node_t_batch_id = example_node2batch_graph_node.setdefault((_example_id, node_source, node_t_id),
                                                                         len(example_node2batch_graph_node))

                    ast_adj_list.append((node_s_batch_id, node_t_batch_id))
                    reversed_ast_adj_list.append((node_t_batch_id, node_s_batch_id))

                for i in range(len(syntax_tree.syntax_tokens_and_ids) - 1):
                    cur_token_id, cur_token = syntax_tree.syntax_tokens_and_ids[i]
                    next_token_id, next_token = syntax_tree.syntax_tokens_and_ids[i + 1]

                    cur_token_batch_id = example_node2batch_graph_node[(_example_id, node_source, cur_token_id)]
                    next_token_batch_id = example_node2batch_graph_node[(_example_id, node_source, next_token_id)]

                    terminal_tokens_adj_list.append((cur_token_batch_id, next_token_batch_id))
                    reversed_terminal_tokens_adj_list.append((next_token_batch_id, cur_token_batch_id))

                if self.prev_sibling_connection or self.next_sibling_connection:
                    for left_node_id, right_node_id in syntax_tree.next_siblings_adjacency_list:
                        left_node_batch_id = example_node2batch_graph_node[(_example_id, node_source, left_node_id)]
                        right_node_batch_id = example_node2batch_graph_node[(_example_id, node_source, right_node_id)]
                        if self.next_sibling_connection:
                            next_sibling_adj_list.append((left_node_batch_id, right_node_batch_id))
                        if self.prev_sibling_connection:
                            prev_sibling_adj_list.append((right_node_batch_id, left_node_batch_id))

        def _register_change_edges(_example_id, _edges, _forward_list, _reverse_list):
            for _src_node_id, _tgt_node_id in _edges:
                node_s_batch_id = example_node2batch_graph_node[_example_id, 0, _src_node_id]
                node_t_batch_id = example_node2batch_graph_node[_example_id, 1, _tgt_node_id]

                _forward_list.append((node_s_batch_id, node_t_batch_id))
                _reverse_list.append((node_t_batch_id, node_s_batch_id))

        for example_id, example in enumerate(examples):
            _build_batch_graph(example_id, example.prev_code_ast, example.updated_code_ast)

            # add change edges
            same_edges, replace_edges = example.change_edges
            _register_change_edges(example_id, same_edges, same_edges_src2tgt_list, same_edges_tgt2src_list)
            _register_change_edges(example_id, replace_edges, replace_edges_src2tgt_list, replace_edges_tgt2src_list)

        example2batch_nodes_map = dict()
        for (_e_id, _node_src, _node_id), _node_batch_id in example_node2batch_graph_node.items():
            example2batch_nodes_map.setdefault(_e_id, []).append(_node_batch_id)

        tree_nodes_num = len(example_node2batch_graph_node)

        if self.next_sibling_connection and self.prev_sibling_connection:
            adj_lists = [ast_adj_list, reversed_ast_adj_list,
                         next_sibling_adj_list, prev_sibling_adj_list,
                         terminal_tokens_adj_list, reversed_terminal_tokens_adj_list,
                         same_edges_src2tgt_list, same_edges_tgt2src_list,
                         replace_edges_src2tgt_list, replace_edges_tgt2src_list]
        else:
            adj_lists = [ast_adj_list, reversed_ast_adj_list,
                         terminal_tokens_adj_list, reversed_terminal_tokens_adj_list,
                         same_edges_src2tgt_list, same_edges_tgt2src_list,
                         replace_edges_src2tgt_list, replace_edges_tgt2src_list]

        if self.master_node_option:
            if self.master_node_option == 'single_node':
                master_node_id_start = tree_nodes_num
                src2master_adj_list = []
                master2src_adj_list = []
                tgt2master_adj_list = []
                master2tgt_adj_list = []

                for e_id, example in enumerate(examples):
                    master_node_id = master_node_id_start + e_id
                    for src_node_id in example.prev_code_ast.id2node:
                        src_node_batch_id = example_node2batch_graph_node[(e_id, 0, src_node_id)]
                        src2master_adj_list.append((src_node_batch_id, master_node_id))
                        master2src_adj_list.append((master_node_id, src_node_batch_id))

                    for tgt_node_id in example.updated_code_ast.id2node:
                        tgt_node_batch_id = example_node2batch_graph_node[(e_id, 1, tgt_node_id)]
                        tgt2master_adj_list.append((tgt_node_batch_id, master_node_id))
                        master2tgt_adj_list.append((master_node_id, tgt_node_batch_id))

                adj_lists.extend([src2master_adj_list, master2src_adj_list, tgt2master_adj_list, master2tgt_adj_list])
                all_nodes_num = tree_nodes_num + len(examples)
            else:
                master_node_id_start = tree_nodes_num
                prev_ast_node2master_adj_list = []
                prev_ast_master2node_adj_list = []
                updated_ast_node2master_adj_list = []
                updated_ast_master2node_adj_list = []

                for e_id, example in enumerate(examples):
                    prev_ast_master_node_id = master_node_id_start
                    updated_ast_master_node_id = prev_ast_master_node_id + 1

                    for src_node_id in example.prev_code_ast.id2node:
                        src_node_batch_id = example_node2batch_graph_node[(e_id, 0, src_node_id)]
                        prev_ast_node2master_adj_list.append((src_node_batch_id, prev_ast_master_node_id))
                        prev_ast_master2node_adj_list.append((prev_ast_master_node_id, src_node_batch_id))

                    for tgt_node_id in example.updated_code_ast.id2node:
                        tgt_node_batch_id = example_node2batch_graph_node[(e_id, 1, tgt_node_id)]
                        updated_ast_node2master_adj_list.append((tgt_node_batch_id, updated_ast_master_node_id))
                        updated_ast_master2node_adj_list.append((updated_ast_master_node_id, tgt_node_batch_id))

                    master_node_id_start = updated_ast_master_node_id + 1

                adj_lists.extend([prev_ast_node2master_adj_list, prev_ast_master2node_adj_list,
                                  updated_ast_node2master_adj_list, updated_ast_master2node_adj_list])
                all_nodes_num = tree_nodes_num + len(examples) * 2
        else:
            all_nodes_num = tree_nodes_num

        adj_list_objs = [AdjacencyList(node_num=all_nodes_num, adj_list=adj_list, device=self.device) for adj_list in adj_lists]

        return adj_list_objs, example_node2batch_graph_node, example2batch_nodes_map

    def forward(self, examples, prev_code_token_encoding, updated_code_token_encoding):
        batch_adj_lists, example_node2batch_node_map, example2batch_nodes_map = self.get_batch_adjacency_lists(examples)

        # get initial embeddings for every nodes
        # (V, D)
        init_node_embeddings = self.embed_change_graph_nodes(examples,
                                                             example_node2batch_node_map,
                                                             prev_code_token_encoding.encoding,
                                                             updated_code_token_encoding.encoding)

        # append master nodes
        if self.master_node_option == 'single_node':
            init_master_node_embeddings = torch.zeros(len(examples), init_node_embeddings.size(1), device=self.device)
        elif self.master_node_option == 'double_node':
            init_master_node_embeddings = torch.zeros(len(examples) * 2, init_node_embeddings.size(1), device=self.device)

        if self.master_node_option:
            init_node_embeddings = torch.cat([init_node_embeddings, init_master_node_embeddings], dim=0)

        # perform encoding
        # (V, D)
        flattened_node_encodings = self.gnn(init_node_embeddings, batch_adj_lists, return_all_states=True)

        change_vectors = self.get_change_vector(flattened_node_encodings, examples, example2batch_nodes_map)

        return change_vectors

    def get_change_vector(self, flattened_node_encodings, examples, example2batch_nodes_map):
        # perform average pooling over source and target nodes

        assert len(flattened_node_encodings) == 2

        change_vectors = []
        for example_id, example in enumerate(examples):
            batch_node_ids_for_this_example = example2batch_nodes_map[example_id]

            # (num_nodes_in_this_example, hidden_size)
            node_representations_layer1 = flattened_node_encodings[0][batch_node_ids_for_this_example]
            node_representations_layer1 = torch.sigmoid(self.layer1_readout_weight(node_representations_layer1)) * node_representations_layer1
            node_representations_layer1 = node_representations_layer1.sum(dim=0)

            node_representations_layer2 = flattened_node_encodings[1][batch_node_ids_for_this_example]
            node_representations_layer2 = torch.sigmoid(self.layer2_readout_weight(node_representations_layer2)) * node_representations_layer2
            node_representations_layer2 = node_representations_layer2.sum(dim=0)

            change_vectors.append(torch.cat([node_representations_layer1, node_representations_layer2]))

        return torch.stack(change_vectors)

    def embed_change_graph_nodes(self, examples, example_node2batch_node_map, prev_code_token_encoding, updated_code_token_encoding):
        # pre-process change sequence
        change_tag_map = dict()
        for example_id, example in enumerate(examples):
            prev_token_ptr = updated_token_ptr = 0
            for entry in example.change_seq:
                tag, token = entry
                if tag == 'SAME' or tag == 'REPLACE':
                    change_tag_map[(example_id, 0, prev_token_ptr)] = tag
                    change_tag_map[(example_id, 1, updated_token_ptr)] = tag

                    prev_token_ptr += 1
                    updated_token_ptr += 1
                elif tag == 'ADD':
                    change_tag_map[(example_id, 1, updated_token_ptr)] = tag

                    updated_token_ptr += 1
                elif tag == 'DEL':
                    change_tag_map[(example_id, 0, prev_token_ptr)] = tag

                    prev_token_ptr += 1

        embed_indices = []
        node_num = len(example_node2batch_node_map)
        change_tag_indices = [0] * node_num
        change_tag_mask = [0.] * node_num

        example_ids_for_prev_code_tokens = []
        prev_token_pos = []
        prev_token_batch_node_ids = []
        example_ids_for_updated_code_tokens = []
        updated_token_pos = []
        updated_token_batch_node_ids = []

        for (example_id, node_source, node_id), batch_node_id in example_node2batch_node_map.items():
            example = examples[example_id]
            syntax_tree = example.prev_code_ast if node_source == 0 else example.updated_code_ast
            node = syntax_tree.id2node[node_id]

            if isinstance(node, AbstractSyntaxNode):
                embed_idx = self.syntax_tree_embedder.grammar.type2id[node.production.type] + len(self.syntax_tree_embedder.vocab)
            elif isinstance(node, SyntaxToken):
                if node.position >= 0:
                    embed_idx = 0
                    if node_source == 0:
                        example_ids_for_prev_code_tokens.append(example_id)
                        prev_token_pos.append(node.position)
                        prev_token_batch_node_ids.append(batch_node_id)
                    else:
                        example_ids_for_updated_code_tokens.append(example_id)
                        updated_token_pos.append(node.position)
                        updated_token_batch_node_ids.append(batch_node_id)

                    change_tag = change_tag_map[(example_id, node_source, node.position)]
                    change_tag_indices[batch_node_id] = GraphChangeEncoder.change_tag2id[change_tag]
                    change_tag_mask[batch_node_id] = 1.
                else:
                    embed_idx = self.syntax_tree_embedder.vocab[node.value]

            embed_indices.append(embed_idx)

        for example_id, example in enumerate(examples):
            for prev_node_id, updated_node_id in example.change_edges[0]:
                prev_node_batch_id = example_node2batch_node_map[(example_id, 0, prev_node_id)]
                change_tag_indices[prev_node_batch_id] = GraphChangeEncoder.change_tag2id['SAME']
                change_tag_mask[prev_node_batch_id] = 1.

                updated_node_batch_id = example_node2batch_node_map[(example_id, 1, updated_node_id)]
                change_tag_indices[updated_node_batch_id] = GraphChangeEncoder.change_tag2id['SAME']
                change_tag_mask[updated_node_batch_id] = 1.

        # (all_node_num, embedding_dim)
        node_embedding = self.syntax_tree_embedder.forward(torch.tensor(embed_indices, dtype=torch.long, device=self.device))

        prev_syntax_token_embedding = prev_code_token_encoding[example_ids_for_prev_code_tokens, prev_token_pos]
        node_embedding[prev_token_batch_node_ids] = prev_syntax_token_embedding

        updated_syntax_token_embedding = updated_code_token_encoding[example_ids_for_updated_code_tokens, updated_token_pos]
        node_embedding[updated_token_batch_node_ids] = updated_syntax_token_embedding

        tree_source_indicator = torch.zeros(node_num, 2, device=self.device)
        tree_src_ids = [[x[1]] for x in example_node2batch_node_map.keys()]
        tree_source_indicator.scatter_(1, torch.tensor(tree_src_ids, device=self.device), 1)

        change_tag_embedding = self.change_tag_embedding(
            torch.tensor(change_tag_indices, dtype=torch.long, device=self.device)) * torch.tensor(change_tag_mask, device=self.device).unsqueeze(-1)

        node_embedding = torch.cat([tree_source_indicator, change_tag_embedding, node_embedding], dim=-1)

        return node_embedding

    def encode_code_changes(self, examples, code_encoder, batch_size=32):
        change_vecs = []

        for batch_examples in tqdm(nn_utils.batch_iter(examples, batch_size), file=sys.stdout, total=len(examples) // batch_size):
            previous_code_chunk_list = [e.previous_code_chunk for e in batch_examples]
            updated_code_chunk_list = [e.updated_code_chunk for e in batch_examples]
            context_list = [e.context for e in batch_examples]

            embedding_cache = EmbeddingTable(
                chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
            code_encoder.code_token_embedder.populate_embedding_table(embedding_cache)

            batched_prev_code = code_encoder.encode(previous_code_chunk_list, embedding_cache=embedding_cache)
            batched_updated_code = code_encoder.encode(updated_code_chunk_list, embedding_cache=embedding_cache)

            batch_change_vecs = self.forward(batch_examples, batched_prev_code, batched_updated_code).data.cpu().numpy()
            change_vecs.append(batch_change_vecs)

        change_vecs = np.concatenate(change_vecs, axis=0)

        return change_vecs

    @staticmethod
    def get_syntax_token_change_edges(example):
        token_same_edges = []
        token_replace_edges = []

        prev_token_ptr = updated_token_ptr = 0
        for entry in example.change_seq:
            tag, token = entry
            if tag == 'SAME' or tag == 'REPLACE':
                if prev_token_ptr in example.prev_code_ast.syntax_token_position2id and \
                        updated_token_ptr in example.updated_code_ast.syntax_token_position2id:

                    prev_node_id = example.prev_code_ast.syntax_token_position2id[prev_token_ptr]
                    tgt_node_id = example.updated_code_ast.syntax_token_position2id[updated_token_ptr]

                    if tag == 'SAME':
                        token_same_edges.append((prev_node_id, tgt_node_id))
                    else:
                        token_replace_edges.append((prev_node_id, tgt_node_id))

                prev_token_ptr += 1
                updated_token_ptr += 1
            elif tag == 'ADD':
                updated_token_ptr += 1
            elif tag == 'DEL':
                prev_token_ptr += 1

        return token_same_edges, token_replace_edges

    @staticmethod
    def compute_change_edges(example) -> List:
        same_edges = set()
        prev_code_ast = example.prev_code_ast
        updated_code_ast = example.updated_code_ast

        def _link_subtrees(src_node, tgt_node):
            _same_edges = set()
            _same_edges.add((src_node.id, tgt_node.id))
            if isinstance(src_node, AbstractSyntaxNode):
                for src_field, tgt_field in zip(src_node.fields, tgt_node.fields):
                    for src_child, tgt_child in zip(src_field.as_value_list, tgt_field.as_value_list):
                        _same_edges.add((src_child.id, tgt_child.id))
                        _same_edges.update(_link_subtrees(src_child, tgt_child))

            return _same_edges

        def _travel(node):
            if isinstance(node, AbstractSyntaxNode):
                search_results = prev_code_ast.find_node(node)
                if search_results:
                    for src_node_id, src_node in search_results:
                        sub_tree_edges = _link_subtrees(src_node, node)
                        # if src_node_id != node.id and src_node.size > 3:
                        #     print(sub_tree_edges)
                        same_edges.update(sub_tree_edges)
                elif isinstance(node, AbstractSyntaxNode):
                    for field in node.fields:
                        for child_node in field.as_value_list:
                            _travel(child_node)

        _travel(updated_code_ast.root_node)

        token_same_edges, token_replace_edges = GraphChangeEncoder.get_syntax_token_change_edges(example)
        same_edges.update(token_same_edges)

        same_edges = sorted(same_edges, key=lambda x: x[0])
        # print(same_edges)

        return same_edges, token_replace_edges
