# coding=utf-8
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from edit_model.data_model import BatchedCodeChunk
from edit_model import nn_utils


from .encoder import EncodingResult


class SequentialEncoder(nn.Module):
    """encode the input data"""

    def __init__(self, token_embed_size, token_encoding_size, token_embedder, vocab):
        super(SequentialEncoder, self).__init__()

        self.vocab = vocab

        self.token_embedder = token_embedder
        self.encoder_lstm = nn.LSTM(token_embed_size, token_encoding_size // 2, bidirectional=True)

    @property
    def device(self):
        return self.token_embedder.device

    def forward(self, prev_data, is_sorted=False, embedding_cache=None):
        batched_code_lens = [len(code) for code in prev_data]

        if is_sorted is False:
            original_prev_data = prev_data
            sorted_example_ids, example_old2new_pos = nn_utils.get_sort_map(batched_code_lens)
            prev_data = [prev_data[i] for i in sorted_example_ids]

        if embedding_cache:
            # (code_seq_len, batch_size, token_embed_size)
            token_embed = embedding_cache.get_embed_for_token_sequences(prev_data)
        else:
            # (code_seq_len, batch_size, token_embed_size)
            token_embed = self.token_embedder.get_embed_for_token_sequences(prev_data)

        packed_token_embed = pack_padded_sequence(token_embed, [len(code) for code in prev_data])

        # token_encodings: (tgt_query_len, batch_size, hidden_size)
        token_encodings, (last_state, last_cell) = self.encoder_lstm(packed_token_embed)
        token_encodings, _ = pad_packed_sequence(token_encodings)

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        # (batch_size, tgt_query_len, hidden_size)
        token_encodings = token_encodings.permute(1, 0, 2)
        if is_sorted is False:
            token_encodings = token_encodings[example_old2new_pos]
            last_state = last_state[example_old2new_pos]
            last_cell = last_cell[example_old2new_pos]
            prev_data = original_prev_data

        return EncodingResult(prev_data, token_encodings, last_state, last_cell,
                              nn_utils.length_array_to_mask_tensor(batched_code_lens, device=self.device))


class ContextEncoder(SequentialEncoder):
    def __init__(self, **kwargs):
        super(ContextEncoder, self).__init__(**kwargs)

    def forward(self, context_list, is_sorted=False):
        return super(ContextEncoder, self).forward(context_list, is_sorted=is_sorted)
