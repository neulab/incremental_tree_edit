from collections import OrderedDict
from itertools import chain
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from asdl.asdl_ast import AbstractSyntaxNode, SyntaxToken, AbstractSyntaxTree
from asdl.asdl import ASDLGrammar
from edit_model import nn_utils
from edit_model.utils import cached_property
from edit_components.vocab import VocabEntry
from edit_model.pointer_net import PointerNet

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 1 for (idx, char) in enumerate(ALPHABET)}  # "0" is PAD
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)} # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1


class Embedder:
    def __init__(self, vocab=None):
        self.vocab = vocab

    @property
    def device(self):
        return self.weight.device

    def populate_embedding_table(self, embedding_table):
        raise NotImplementedError()

    def to_input_variable(self, code_list, return_mask=False):
        word_ids = nn_utils.word2id(code_list, self.vocab)
        sents_t, masks = nn_utils.input_transpose(word_ids, pad_token=0)
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=self.device)

        if return_mask:
            mask_var = torch.tensor(masks, dtype=torch.long, device=self.device)
            return sents_var, mask_var

        return sents_var


class CodeTokenEmbedder(Embedder, nn.Embedding):
    def __init__(self, embedding_size, vocab: VocabEntry):
        nn.Embedding.__init__(self, len(vocab), embedding_size)
        Embedder.__init__(self, vocab)

        nn.init.xavier_normal_(self.weight.data)

    def forward(self, code_list):
        if isinstance(code_list, list):
            index = self.to_input_variable(code_list)
        else:
            index = code_list
        embedding = super(CodeTokenEmbedder, self).forward(index)

        return embedding

    def populate_embedding_table(self, embedding_table):
        tokens = list(embedding_table.tokens.keys())
        indices = [self.vocab[token] for token in tokens]
        token_embedding = super(CodeTokenEmbedder, self).forward(torch.tensor(indices, dtype=torch.long, device=self.device))

        embedding_table.init_with_embeddings(token_embedding)

    def get_embed_for_token_sequences(self, sequences):
        return self.forward(sequences)


class ConvolutionalCharacterEmbedder(nn.Module, Embedder):
    def __init__(self, embed_size: int, max_character_size):
        super(ConvolutionalCharacterEmbedder, self).__init__()

        self.max_character_size = max_character_size
        self.embed_size = embed_size

        self.conv11_layer = nn.Conv1d(in_channels=len(ALPHABET_DICT), out_channels=20, kernel_size=5)
        self.maxpool_layer = nn.MaxPool1d(kernel_size=5, stride=1)
        self.conv12_layer = nn.Conv1d(in_channels=20, out_channels=embed_size, kernel_size=12)

    @property
    def device(self):
        return self.conv11_layer.weight.device

    def populate_embedding_table(self, embedding_table):
        # tensorization
        # (word_num, max_character_size, encode_char_num)
        x = embedding_table.character_input_tensor(max_character_num=self.max_character_size).to(self.device)

        conv1 = F.leaky_relu(self.conv11_layer(x.permute(0, 2, 1)))
        maxpool = self.maxpool_layer(conv1)
        conv2 = self.conv12_layer(maxpool).squeeze(-1)

        embedding_table.init_with_embeddings(conv2)


class EmbeddingTable:
    def __init__(self, tokens):
        self.tokens = OrderedDict()
        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.tokens:
            self.tokens[token] = len(self.tokens)

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)

    def character_input_tensor(self, max_character_num=20, lowercase=True):
        # return: (word_num, max_character_num, char_num)
        idx_tensor = torch.zeros(len(self.tokens), max_character_num, len(ALPHABET_DICT), dtype=torch.float)
        for token, token_id in self.tokens.items():
            if lowercase:
                token = token.lower()

            token_trimmed = token[:max_character_num]
            token_char_seq = [t for t in token_trimmed] + ['PAD'] * (max_character_num - len(token_trimmed))
            token_char_seq_ids = [ALPHABET_DICT[char] if char in ALPHABET_DICT else ALPHABET_DICT['UNK'] for char in token_char_seq]
            idx_tensor[token_id, list(range(max_character_num)), token_char_seq_ids] = 1.0

        return idx_tensor

    def __getitem__(self, token):
        token_id = self.tokens[token]
        embed = self.embedding[token_id]

        return embed

    def init_with_embeddings(self, embedding_tensor):
        # input: (word_num, token_embedding_size)
        self.embedding = embedding_tensor

    def to_input_variable(self, sequences, return_mask=False):
        """
        given a list of sequences,
        return a tensor of shape (max_sent_len, batch_size)
        """
        word_ids = nn_utils.word2id(sequences, self.tokens)
        sents_t, masks = nn_utils.input_transpose(word_ids, pad_token=0)
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=self.embedding.device)

        if return_mask:
            mask_var = torch.tensor(masks, dtype=torch.long, device=self.embedding.device)
            return sents_var, mask_var

        return sents_var

    def get_embed_for_token_sequences(self, sequences):
        # (max_sent_len, batch_size)
        input_var = self.to_input_variable(sequences)

        # (max_sent_len, batch_size, embed_size)
        seq_embed = self.embedding[input_var]

        return seq_embed


class SyntaxTreeEmbedder(nn.Embedding, Embedder):
    def __init__(self, embedding_size, vocab: VocabEntry, grammar: ASDLGrammar, node_embed_method: str='type'):
        if node_embed_method == 'type':
            node_embed_num = len(grammar.types)
        elif node_embed_method == 'type_and_field':
            node_embed_num = len(grammar.types) + len(grammar.prod_field2id) + 1  # 1 for root field
        else:
            raise ValueError

        nn.Embedding.__init__(self, len(vocab) + node_embed_num, embedding_size)
        Embedder.__init__(self, vocab)

        if node_embed_method == 'type_and_field':
            self.combine_type_and_field_embed = nn.Linear(embedding_size * 2, embedding_size)
            self.field_embed_offset = len(vocab) + len(grammar.types)

        self.node_embed_method = node_embed_method

        nn.init.xavier_normal_(self.weight.data)
        
        self.grammar = grammar

    @property
    def device(self):
        return self.weight.device

    def embed_syntax_tree(self,
                          batch_syntax_trees: List[AbstractSyntaxTree],
                          batch_graph_node2example_node: Dict,
                          prev_code_token_encoding: torch.FloatTensor,
                          bool_use_position=True):
        indices = []
        field_indices = []
        token_indices = []
        batch_ids_for_token = []
        token_pos = []
        for i, (batch_node_id, (e_id, e_node_id)) in enumerate(batch_graph_node2example_node.items()):
            node = batch_syntax_trees[e_id].id2node[e_node_id]
            if isinstance(node, AbstractSyntaxNode):
                idx = self.grammar.type2id[node.production.type] + len(self.vocab)
                if self.node_embed_method == 'type_and_field':
                    field_idx = self.field_embed_offset if node.parent_field is None \
                        else self.field_embed_offset + 1 + self.grammar.prod_field2id[(node.parent_field.parent_node.production, node.parent_field.field)]
            elif isinstance(node, SyntaxToken):
                if bool_use_position and node.position >= 0:
                    batch_ids_for_token.append(e_id)
                    token_pos.append(node.position)
                    token_indices.append(i)

                    idx = 0
                else:
                    idx = self.vocab[node.value]

                field_idx = 0
            else:
                raise ValueError('unknown node type %s' % node)

            indices.append(idx)
            if self.node_embed_method == 'type_and_field':
                field_indices.append(field_idx)

        # (all_node_num, embedding_dim)
        node_embedding = super(SyntaxTreeEmbedder, self).forward(torch.tensor(indices, dtype=torch.long, device=self.device))

        if self.node_embed_method == 'type_and_field':
            node_field_embedding = super(SyntaxTreeEmbedder, self).forward(torch.tensor(field_indices, dtype=torch.long, device=self.device))
            node_embedding = self.combine_type_and_field_embed(torch.cat([node_embedding, node_field_embedding], dim=-1))

        syntax_token_encodings = prev_code_token_encoding[batch_ids_for_token, token_pos]
        node_embedding[token_indices] = syntax_token_encodings

        return node_embedding

    def populate_embedding_table(self, embedding_table):
        tokens = list(embedding_table.tokens.keys())
        indices = [self.vocab[token] for token in tokens]
        token_embedding = super(SyntaxTreeEmbedder, self).forward(torch.tensor(indices, dtype=torch.long, device=self.device))

        embedding_table.init_with_embeddings(token_embedding)

    def get_embed_for_token_sequences(self, sequences):
        # (max_sent_len, batch_size)
        input_var = self.to_input_variable(sequences)

        # (max_sent_len, batch_size, embed_size)
        seq_embed = super(SyntaxTreeEmbedder, self).forward(input_var)

        return seq_embed
