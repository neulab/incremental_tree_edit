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
from edit_model.embedder import EmbeddingTable


class BagOfEditsChangeEncoder(nn.Module):
    """project a CodeChange instance into distributed vectors"""

    def __init__(self, token_embedder, vocab, **kwargs):
        super(BagOfEditsChangeEncoder, self).__init__()

        self.token_embedder = token_embedder
        self.token_embedding_size = self.token_embedder.weight.size(1)
        self.vocab = vocab
        self.change_vector_size = self.token_embedding_size * 2

    @property
    def device(self):
        return self.token_embedder.device

    def forward(self, code_changes, *args, **kwargs):
        """
        given the token encodings of the previous and updated code,
        and the diff information (alignment between the tokens between the
        previous and updated code), generate the diff representation
        """

        added_tokens = []
        added_token_batch_ids = []
        deled_tokens = []
        deled_token_batch_ids = []
        for e_id, example in enumerate(code_changes):
            for entry in example.change_seq:
                tag, token = entry
                if tag == 'ADD':
                    token_id = self.vocab[token]
                    added_tokens.append(token_id)
                    added_token_batch_ids.append(e_id)
                elif tag == 'DEL':
                    token_id = self.vocab[token]
                    deled_tokens.append(token_id)
                    deled_token_batch_ids.append(e_id)
                elif tag == 'REPLACE':
                    added_token_id = self.vocab[token[1]]
                    deled_token_id = self.vocab[token[0]]

                    added_tokens.append(added_token_id)
                    deled_tokens.append(deled_token_id)

                    added_token_batch_ids.append(e_id)
                    deled_token_batch_ids.append(e_id)

        changed_token_ids = added_tokens + deled_tokens
        changed_token_ids = torch.tensor(changed_token_ids, dtype=torch.long, device=self.device)
        # (token_num, embed_size)
        changed_token_embeds = self.token_embedder.weight[changed_token_ids]

        added_token_embeds = changed_token_embeds[:len(added_tokens)]
        deled_token_embeds = changed_token_embeds[len(added_tokens):]

        added_change_embeds = torch.zeros(len(code_changes), self.token_embedding_size, dtype=torch.float,
                                          device=self.device)
        if added_token_batch_ids:
            added_change_embeds = added_change_embeds.scatter_add_(0,
                                                                   torch.tensor(added_token_batch_ids, device=self.device).unsqueeze(-1).expand_as(added_token_embeds),
                                                                   added_token_embeds)

        deled_change_embeds = torch.zeros(len(code_changes), self.token_embedding_size, dtype=torch.float,
                                          device=self.device)
        if deled_token_batch_ids:
            deled_change_embeds = deled_change_embeds.scatter_add_(0,
                                                                   torch.tensor(deled_token_batch_ids, device=self.device).unsqueeze(-1).expand_as(deled_token_embeds),
                                                                   deled_token_embeds)

        change_vectors = torch.cat([added_change_embeds, deled_change_embeds], dim=-1)

        return change_vectors

    def encode_code_change(self, prev_code_tokens, updated_code_tokens, code_encoder):
        example = ChangeExample(prev_code_tokens, updated_code_tokens, context=None)

        change_vec = self.forward([example]).data.cpu().numpy()[0]

        return change_vec

    def encode_code_changes(self, examples, code_encoder, batch_size=32):
        """encode each change in the list `code_changes`,
        return a 2D numpy array of shape (len(code_changes), code_change_embed_dim)"""

        change_vecs = []

        for batch_examples in tqdm(nn_utils.batch_iter(examples, batch_size), file=sys.stdout, total=len(examples)):
            batch_change_vecs = self.forward(batch_examples).data.cpu().numpy()
            change_vecs.append(batch_change_vecs)

        change_vecs = np.concatenate(change_vecs, axis=0)

        return change_vecs
