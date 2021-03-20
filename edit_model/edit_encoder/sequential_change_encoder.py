from itertools import chain

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import sys

from edit_components.change_entry import ChangeExample
from edit_model import nn_utils


class SequentialChangeEncoder(nn.Module):
    """project a CodeChange instance into distributed vectors"""
    change_tags = ('ADD', 'DEL', 'SAME', 'REPLACE')
    change_tag2id = {tag: i for i, tag in enumerate(change_tags)}

    def __init__(self, token_encoding_size, change_vector_size, tag_embed_size, vocab,
                 no_unchanged_token_encoding_in_diff_seq=False):
        super(SequentialChangeEncoder, self).__init__()

        self.token_encoding_size = token_encoding_size
        self.change_vector_size = change_vector_size
        self.tag_embed_size = tag_embed_size

        self.tag_embedding = nn.Embedding(len(self.change_tags), self.tag_embed_size)
        self.change_seq_encoder_lstm = nn.LSTM(self.tag_embed_size + self.token_encoding_size * 2, self.change_vector_size // 2,
                                               bidirectional=True)

        self.vocab = vocab
        self.no_unchanged_token_encoding_in_diff_seq = no_unchanged_token_encoding_in_diff_seq

    @property
    def device(self):
        return self.tag_embedding.weight.device

    def forward(self, code_changes, batched_prev_code, batched_tgt_code):
        """given the token encodings of the previous and updated code,
           and the diff information (alignment between the tokens between the
           previous and updated code), generate the diff representation"""

        # (batch_size, diff_seq_len, token_encoding_size + tag_embedding_size)
        aligned_encoding = self.get_aligned_token_encoding(code_changes, batched_prev_code, batched_tgt_code)

        # (batch_size, change_vec_size)
        change_vectors = self.get_change_vector_from_aligned_encoding(code_changes, aligned_encoding)

        return change_vectors

    @staticmethod
    def populate_aligned_token_index_and_mask(example):
        change_seq_len = len(example.change_seq)

        prev_token_index = torch.zeros(change_seq_len, dtype=torch.long)
        prev_token_mask = torch.zeros(change_seq_len, dtype=torch.float)

        updated_token_index = torch.zeros(change_seq_len, dtype=torch.long)
        updated_token_mask = torch.zeros(change_seq_len, dtype=torch.float)

        tag_index = torch.zeros(change_seq_len, dtype=torch.float)

        prev_token_ptr = updated_token_ptr = 0
        for i, entry in enumerate(example.change_seq):
            # entry: (TAG, TOKEN)
            tag, token = entry
            if tag == 'SAME' or tag == 'REPLACE':
                prev_token_index[i] = prev_token_ptr
                updated_token_index[i] = updated_token_ptr

                prev_token_ptr += 1
                updated_token_ptr += 1

                prev_token_mask[i] = 1
                updated_token_mask[i] = 1
            elif tag == 'ADD':
                updated_token_index[i] = updated_token_ptr

                updated_token_ptr += 1
                updated_token_mask[i] = 1
            elif tag == 'DEL':
                prev_token_index[i] = prev_token_ptr

                prev_token_ptr += 1
                prev_token_mask[i] = 1
            else:
                raise ValueError('Unknown diff tag [%s]' % tag)

            tag_index[i] = SequentialChangeEncoder.change_tag2id[tag]

        example.prev_token_index = prev_token_index
        example.prev_token_mask = prev_token_mask
        example.updated_token_index = updated_token_index
        example.updated_token_mask = updated_token_mask
        example.tag_index = tag_index

    def get_aligned_token_encoding(self, examples, batched_prev_code, batched_tgt_code):
        max_change_seq_len = max(len(change.change_seq) for change in examples)
        batch_size = len(examples)

        prev_token_index = torch.zeros(batch_size, max_change_seq_len, dtype=torch.long)
        prev_token_mask = torch.zeros(batch_size, max_change_seq_len, dtype=torch.float)

        updated_token_index = torch.zeros(batch_size, max_change_seq_len, dtype=torch.long)
        updated_token_mask = torch.zeros(batch_size, max_change_seq_len, dtype=torch.float)

        tag_index = torch.zeros(batch_size, max_change_seq_len, dtype=torch.long)

        if hasattr(examples[0], 'prev_token_index'):
            for batch_id, example in enumerate(examples):
                change_seq = example.change_seq

                prev_token_index[batch_id, :len(change_seq)] = example.prev_token_index
                prev_token_mask[batch_id, :len(change_seq)] = example.prev_token_mask

                updated_token_index[batch_id, :len(change_seq)] = example.updated_token_index
                updated_token_mask[batch_id, :len(change_seq)] = example.updated_token_mask

                tag_index[batch_id, :len(change_seq)] = example.tag_index
        else:
            for batch_id in range(len(examples)):
                change = examples[batch_id]
                prev_token_ptr = updated_token_ptr = 0
                for i, entry in enumerate(change.change_seq):
                    # entry: (TAG, TOKEN)
                    tag, token = entry
                    if tag == 'SAME' or tag == 'REPLACE':
                        prev_token_index[batch_id, i] = prev_token_ptr
                        updated_token_index[batch_id, i] = updated_token_ptr

                        prev_token_ptr += 1
                        updated_token_ptr += 1

                        if self.no_unchanged_token_encoding_in_diff_seq:
                            if tag == 'SAME':
                                prev_token_mask[batch_id, i] = 0
                                updated_token_mask[batch_id, i] = 0
                            else:
                                prev_token_mask[batch_id, i] = 0
                                updated_token_mask[batch_id, i] = 1
                        else:
                            prev_token_mask[batch_id, i] = 1
                            updated_token_mask[batch_id, i] = 1
                    elif tag == 'ADD':
                        updated_token_index[batch_id, i] = updated_token_ptr

                        updated_token_ptr += 1
                        updated_token_mask[batch_id, i] = 1
                    elif tag == 'DEL':
                        prev_token_index[batch_id, i] = prev_token_ptr

                        prev_token_ptr += 1
                        prev_token_mask[batch_id, i] = 1
                    else:
                        raise ValueError('Unknown diff tag [%s]' % tag)

                    tag_index[batch_id, i] = self.change_tag2id[tag]

        prev_token_index = prev_token_index.to(self.device)
        prev_token_mask = prev_token_mask.to(self.device)
        updated_token_index = updated_token_index.to(self.device)
        updated_token_mask = updated_token_mask.to(self.device)
        tag_index = tag_index.to(self.device)

        # prev_token_index_old = torch.zeros(batch_size, max_change_seq_len, dtype=torch.long, device=self.device)
        # prev_token_mask_old = torch.zeros(batch_size, max_change_seq_len, dtype=torch.float, device=self.device)
        #
        # updated_token_index_old = torch.zeros(batch_size, max_change_seq_len, dtype=torch.long, device=self.device)
        # updated_token_mask_old = torch.zeros(batch_size, max_change_seq_len, dtype=torch.float, device=self.device)
        #
        # tag_index_old = torch.zeros(batch_size, max_change_seq_len, dtype=torch.long, device=self.device)
        #
        # for batch_id in range(len(examples)):
        #     change = examples[batch_id]
        #     prev_token_ptr = updated_token_ptr = 0
        #     for i, entry in enumerate(change.change_seq):
        #         # entry: (TAG, TOKEN)
        #         tag, token = entry
        #         if tag == 'SAME' or tag == 'REPLACE':
        #             prev_token_index_old[batch_id, i] = prev_token_ptr
        #             updated_token_index_old[batch_id, i] = updated_token_ptr
        #
        #             prev_token_ptr += 1
        #             updated_token_ptr += 1
        #
        #             if self.no_unchanged_token_encoding_in_diff_seq:
        #                 if tag == 'SAME':
        #                     prev_token_mask_old[batch_id, i] = 0
        #                     updated_token_mask_old[batch_id, i] = 0
        #                 else:
        #                     prev_token_mask_old[batch_id, i] = 0
        #                     updated_token_mask_old[batch_id, i] = 1
        #             else:
        #                 prev_token_mask_old[batch_id, i] = 1
        #                 updated_token_mask_old[batch_id, i] = 1
        #         elif tag == 'ADD':
        #             updated_token_index_old[batch_id, i] = updated_token_ptr
        #
        #             updated_token_ptr += 1
        #             updated_token_mask_old[batch_id, i] = 1
        #         elif tag == 'DEL':
        #             prev_token_index_old[batch_id, i] = prev_token_ptr
        #
        #             prev_token_ptr += 1
        #             prev_token_mask_old[batch_id, i] = 1
        #         else:
        #             raise ValueError('Unknown diff tag [%s]' % tag)
        #
        #         tag_index_old[batch_id, i] = self.change_tag2id[tag]
        #
        # assert torch.all(torch.eq(prev_token_index, prev_token_index_old))
        # assert torch.all(torch.eq(prev_token_mask, prev_token_mask_old))
        # assert torch.all(torch.eq(updated_token_index, updated_token_index_old))
        # assert torch.all(torch.eq(updated_token_mask, updated_token_mask_old))
        # assert torch.all(torch.eq(tag_index, tag_index_old))

        prev_token_index = prev_token_index.unsqueeze(2).expand(batch_size, max_change_seq_len, batched_prev_code.encoding.size(-1))
        updated_token_index = updated_token_index.unsqueeze(2).expand(batch_size, max_change_seq_len, batched_prev_code.encoding.size(-1))

        # (batch_size, max_change_seq_len, encoding_size)
        aligned_prev_encoding = batched_prev_code.encoding.gather(dim=1, index=prev_token_index) * prev_token_mask.unsqueeze(-1)

        # (batch_size, max_change_seq_len, encoding_size)
        aligned_updated_encoding = batched_tgt_code.encoding.gather(dim=1, index=updated_token_index) * updated_token_mask.unsqueeze(-1)

        # get the tag encoding
        # (batch_size, max_change_seq_len, tag_embed_size)
        tag_embed = self.tag_embedding(Variable(tag_index))

        aligned_change_seq_encoding_with_tag = torch.cat([tag_embed, aligned_prev_encoding, aligned_updated_encoding], dim=-1)

        return aligned_change_seq_encoding_with_tag

    def get_change_vector_from_aligned_encoding(self, code_changes, aligned_change_seq_encoding):
        packed_encoding = pack_padded_sequence(aligned_change_seq_encoding.permute(1, 0, 2),
                                               [len(change.change_seq) for change in code_changes])

        # token_encodings: (seq_len, batch_size, hidden_size)
        change_seq_encodings, (last_state, last_cell) = self.change_seq_encoder_lstm(packed_encoding)
        change_seq_encodings, _ = pad_packed_sequence(change_seq_encodings)

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)

        return last_state

    def encode_code_change(self, prev_code_tokens, updated_code_tokens, code_encoder):
        from edit_model.embedder import EmbeddingTable

        embedding_cache = EmbeddingTable(prev_code_tokens + updated_code_tokens)
        code_encoder.code_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = code_encoder.encode([prev_code_tokens], embedding_cache=embedding_cache)
        batched_updated_code = code_encoder.encode([updated_code_tokens], embedding_cache=embedding_cache)

        example = ChangeExample(prev_code_tokens, updated_code_tokens, context=None)

        change_vec = self.forward([example], batched_prev_code, batched_updated_code).data.cpu().numpy()[0]

        return change_vec

    def encode_code_changes(self, examples, code_encoder, batch_size=32):
        """encode each change in the list `code_changes`,
        return a 2D numpy array of shape (len(code_changes), code_change_embed_dim)"""
        from edit_model.embedder import EmbeddingTable

        sorted_example_ids, example_old2new_pos = nn_utils.get_sort_map([len(c.change_seq) for c in examples])
        sorted_examples = [examples[i] for i in sorted_example_ids]

        change_vecs = []

        for batch_examples in tqdm(nn_utils.batch_iter(sorted_examples, batch_size), file=sys.stdout, total=len(examples) // batch_size):
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
        # unsort all batch_examples
        change_vecs = change_vecs[example_old2new_pos]

        return change_vecs
