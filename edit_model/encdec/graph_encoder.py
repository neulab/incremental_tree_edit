from collections import OrderedDict, namedtuple
from typing import List
import sys

import torch
import torch.nn as nn

from asdl.asdl_ast import AbstractSyntaxTree
from edit_model.gnn import AdjacencyList, GatedGraphNeuralNetwork
from edit_model.utils import get_method_args_dict

TreeEncodingResult = namedtuple('TreeEncodingResult', ['data', 'encoding', 'mask', 'syntax_token_mask'])


class SyntaxTreeEncoder(nn.Module):
    def __init__(self, hidden_size, syntax_tree_embedder, connections, layer_timesteps, residual_connections, dropout,
                 vocab, grammar, **kwargs):
        super(SyntaxTreeEncoder, self).__init__()

        self.grammar = grammar
        self.vocab = vocab

        self.connections = connections
        self.token_bidirectional_connection = 'bi_token' in connections
        self.top_down_connection = 'top_down' in connections
        self.bottom_up_connection = 'bottom_up' in connections
        self.next_sibling_connection = 'next_sibling' in connections
        self.prev_sibling_connection = 'prev_sibling' in connections
        self.gnn_use_bias_for_message_linear = kwargs.pop('gnn_use_bias_for_message_linear', True)

        self.num_edge_types = 0
        if self.token_bidirectional_connection:
            self.num_edge_types += 2
        if self.top_down_connection:
            self.num_edge_types += 1
        if self.bottom_up_connection:
            self.num_edge_types += 1
        if self.next_sibling_connection:
            self.num_edge_types += 1
        if self.prev_sibling_connection:
            self.num_edge_types += 1

        assert self.num_edge_types > 0

        self.syntax_tree_embedder = syntax_tree_embedder
        self.gnn = GatedGraphNeuralNetwork(hidden_size=hidden_size,
                                           num_edge_types=self.num_edge_types,
                                           layer_timesteps=layer_timesteps,
                                           residual_connections=residual_connections,
                                           state_to_message_dropout=dropout,
                                           rnn_dropout=dropout,
                                           use_bias_for_message_linear=self.gnn_use_bias_for_message_linear)

    @property
    def device(self):
        return self.syntax_tree_embedder.device

    def forward(self, batch_syntax_trees, prev_code_token_encoding=None):
        # combine ASTs into a huge graph
        batch_adj_lists, example_node2batch_node_map, batch_node2_example_node_map = self.get_batch_adjacency_lists(batch_syntax_trees)

        # get initial embeddings for every nodes
        # (V, D)
        if prev_code_token_encoding is not None:
            init_node_embeddings = self.syntax_tree_embedder.embed_syntax_tree(
                batch_syntax_trees, batch_node2_example_node_map, prev_code_token_encoding)
        else:
            init_node_embeddings = torch.zeros(len(batch_node2_example_node_map), self.syntax_tree_embedder.embedding_dim,
                                               dtype=torch.float).to(self.device)

        # perform encoding
        # (V, D)
        flattened_node_encodings = self.gnn(init_node_embeddings, batch_adj_lists)

        # split node encodings from the huge graph, List[Variable[batch_node_num]]
        encoding_result = self.to_batch_encoding(flattened_node_encodings, batch_syntax_trees, example_node2batch_node_map)

        return encoding_result

    def get_batch_adjacency_lists(self, batch_syntax_trees: List[AbstractSyntaxTree]):
        example_node2batch_graph_node = OrderedDict()

        # add parent -> child node on ASTs
        ast_adj_list = []
        reversed_ast_adj_list = []
        terminal_tokens_adj_list = []
        reversed_terminal_tokens_adj_list = []
        next_sibling_adj_list = []
        prev_sibling_adj_list = []
        for e_id, syntax_tree in enumerate(batch_syntax_trees):
            for node_s_id, node_t_id in syntax_tree.adjacency_list:
                # source -> target
                node_s_batch_id = example_node2batch_graph_node.setdefault((e_id, node_s_id),
                                                                     len(example_node2batch_graph_node))
                node_t_batch_id = example_node2batch_graph_node.setdefault((e_id, node_t_id),
                                                                     len(example_node2batch_graph_node))

                if self.top_down_connection:
                    ast_adj_list.append((node_s_batch_id, node_t_batch_id))
                if self.bottom_up_connection:
                    reversed_ast_adj_list.append((node_t_batch_id, node_s_batch_id))

            # add bi-directional connection between adjacent terminal nodes
            if self.token_bidirectional_connection:
                for i in range(len(syntax_tree.syntax_tokens_and_ids) - 1):
                    cur_token_id, cur_token = syntax_tree.syntax_tokens_and_ids[i]
                    next_token_id, next_token = syntax_tree.syntax_tokens_and_ids[i + 1]

                    cur_token_batch_id = example_node2batch_graph_node[(e_id, cur_token_id)]
                    next_token_batch_id = example_node2batch_graph_node[(e_id, next_token_id)]

                    terminal_tokens_adj_list.append((cur_token_batch_id, next_token_batch_id))
                    reversed_terminal_tokens_adj_list.append((next_token_batch_id, cur_token_batch_id))

            if self.prev_sibling_connection or self.next_sibling_connection:
                for left_node_id, right_node_id in syntax_tree.next_siblings_adjacency_list:
                    left_node_batch_id = example_node2batch_graph_node[(e_id, left_node_id)]
                    right_node_batch_id = example_node2batch_graph_node[(e_id, right_node_id)]
                    if self.next_sibling_connection:
                        next_sibling_adj_list.append((left_node_batch_id, right_node_batch_id))
                    if self.prev_sibling_connection:
                        prev_sibling_adj_list.append((right_node_batch_id, left_node_batch_id))

        batch_graph_node2example_node = OrderedDict([(v, k) for k, v in example_node2batch_graph_node.items()])

        all_nodes_num = len(example_node2batch_graph_node)
        adj_lists = []
        if self.top_down_connection:
            ast_adj_list = AdjacencyList(node_num=all_nodes_num, adj_list=ast_adj_list, device=self.device)
            adj_lists.append(ast_adj_list)
        if self.bottom_up_connection:
            reversed_ast_adj_list = AdjacencyList(node_num=all_nodes_num, adj_list=reversed_ast_adj_list, device=self.device)
            adj_lists.append(reversed_ast_adj_list)

        if self.token_bidirectional_connection and terminal_tokens_adj_list:
            terminal_tokens_adj_list = AdjacencyList(node_num=all_nodes_num, adj_list=terminal_tokens_adj_list, device=self.device)
            reversed_terminal_tokens_adj_list = AdjacencyList(node_num=all_nodes_num, adj_list=reversed_terminal_tokens_adj_list, device=self.device)

            adj_lists.append(terminal_tokens_adj_list)
            adj_lists.append(reversed_terminal_tokens_adj_list)

        if self.prev_sibling_connection and prev_sibling_adj_list:
            prev_sibling_adj_list = AdjacencyList(node_num=all_nodes_num, adj_list=prev_sibling_adj_list, device=self.device)
            adj_lists.append(prev_sibling_adj_list)

        if self.next_sibling_connection and next_sibling_adj_list:
            next_sibling_adj_list = AdjacencyList(node_num=all_nodes_num, adj_list=next_sibling_adj_list,
                                                  device=self.device)
            adj_lists.append(next_sibling_adj_list)

        # print(f'batch size: {len(batch_syntax_trees)}, '
        #       f'total_edges: {sum(adj_list.edge_num for adj_list in adj_lists)}, '
        #       f'total nodes: {len(example_node2batch_graph_node)}, '
        #       f'max edges: {max(len(tree.adjacency_list) for tree in batch_syntax_trees)}, '
        #       f'max nodes: {max(tree.node_num for tree in batch_syntax_trees)}', file=sys.stderr)

        return adj_lists, \
               example_node2batch_graph_node, \
               batch_graph_node2example_node

    def to_batch_encoding(self, flattened_node_encodings, batch_syntax_trees, example_node2batch_node_map):
        max_node_num = max(tree.node_num for tree in batch_syntax_trees)
        index_list = []
        mask_list = []
        syntax_token_mask_list = []

        for e_id, syntax_tree in enumerate(batch_syntax_trees):
            example_nodes_with_batch_id = [(example_node_id, batch_node_id)
                                           for (_e_id, example_node_id), batch_node_id
                                           in example_node2batch_node_map.items()
                                           if _e_id == e_id]
            # example_nodes_batch_id = list(map(lambda x: x[1], sorted(example_nodes_with_batch_id, key=lambda t: t[0])))
            sorted_example_nodes_with_batch_id = sorted(example_nodes_with_batch_id, key=lambda t: t[0])
            example_nodes_batch_id = [t[1] for t in sorted_example_nodes_with_batch_id]

            example_idx_list = example_nodes_batch_id + [0] * (max_node_num - syntax_tree.node_num)
            example_node_masks = [0] * len(example_nodes_batch_id) + [1] * (max_node_num - syntax_tree.node_num)
            syntax_token_masks = [0 if syntax_tree.is_syntax_token(node_id) else 1 for node_id, batch_node_id in sorted_example_nodes_with_batch_id] + [1] * (max_node_num - syntax_tree.node_num)

            index_list.append(example_idx_list)
            mask_list.append(example_node_masks)
            syntax_token_mask_list.append(syntax_token_masks)

        # (batch_size, max_node_num, node_encoding_size)
        batch_node_encoding = flattened_node_encodings[index_list, :]
        batch_node_encoding_mask = torch.tensor(mask_list, dtype=torch.bool, device=self.device) # uint8 -> bool, pytorch version upgrade
        batch_node_syntax_token_mask = torch.tensor(syntax_token_mask_list, dtype=torch.bool, device=self.device) # FIXME: [Ziyu] syntax_token_mask_list?

        batch_node_encoding.data.masked_fill_(batch_node_encoding_mask.unsqueeze(-1), 0.)

        return TreeEncodingResult(batch_syntax_trees, batch_node_encoding, batch_node_encoding_mask, batch_node_syntax_token_mask)
