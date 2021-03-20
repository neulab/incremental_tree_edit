import torch
import torch.nn as nn
import tqdm
import sys
from itertools import chain
import numpy as np

from edit_model.edit_encoder.sequential_change_encoder import SequentialChangeEncoder
from edit_model.edit_encoder.graph_change_encoder import GraphChangeEncoder
from edit_model import nn_utils
from edit_model.embedder import EmbeddingTable


class HybridChangeEncoder(nn.Module):
    def __init__(self, change_vector_dim, token_encoding_size, syntax_tree_embedder,
                 layer_timesteps, dropout, vocab,
                 tag_embed_size=32, gnn_use_bias_for_message_linear=True):
        super(HybridChangeEncoder, self).__init__()
        
        self.seq_change_encoder = SequentialChangeEncoder(token_encoding_size, change_vector_dim, tag_embed_size, vocab=vocab)
        self.graph_change_encoder = GraphChangeEncoder(change_vector_dim, layer_timesteps, dropout, syntax_tree_embedder,
                                                       tag_embed_size=tag_embed_size,
                                                       gnn_use_bias_for_message_linear=gnn_use_bias_for_message_linear)

        self.change_vector_dim = change_vector_dim
        self.combo_linear = nn.Linear(change_vector_dim * 2, change_vector_dim)

    @property
    def device(self):
        return self.seq_change_encoder.device

    def forward(self, examples, prev_code_token_encoding, updated_code_token_encoding):
        seq_change_vec = self.seq_change_encoder(examples, prev_code_token_encoding, updated_code_token_encoding)
        graph_change_vec = self.graph_change_encoder(examples, prev_code_token_encoding, updated_code_token_encoding)
        change_vec = self.combo_linear(torch.cat([seq_change_vec, graph_change_vec], dim=-1))

        return change_vec

    def encode_code_changes(self, examples, code_encoder, batch_size=32):
        change_vecs = []

        for batch_examples, sorted_example_ids, example_old2new_pos in tqdm(nn_utils.batch_iter(examples, batch_size, sort_func=lambda e: -len(e.change_seq), return_sort_map=True),
                                                                            total=len(examples) // batch_size, file=sys.stdout):
            previous_code_chunk_list = [e.previous_code_chunk for e in batch_examples]
            updated_code_chunk_list = [e.updated_code_chunk for e in batch_examples]
            context_list = [e.context for e in batch_examples]

            embedding_cache = EmbeddingTable(
                chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
            code_encoder.code_token_embedder.populate_embedding_table(embedding_cache)

            batched_prev_code = code_encoder.encode(previous_code_chunk_list, embedding_cache=embedding_cache)
            batched_updated_code = code_encoder.encode(updated_code_chunk_list, embedding_cache=embedding_cache)

            batch_change_vecs = self.forward(batch_examples, batched_prev_code, batched_updated_code).data.cpu().numpy()
            batch_change_vecs = batch_change_vecs[example_old2new_pos]
            change_vecs.append(batch_change_vecs)

        change_vecs = np.concatenate(change_vecs, axis=0)

        return change_vecs
