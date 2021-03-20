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
from edit_model.edit_encoder.bag_of_edits_change_encoder import BagOfEditsChangeEncoder
from edit_model.edit_encoder.graph_change_encoder import GraphChangeEncoder
from edit_model.edit_encoder.hybrid_change_encoder import HybridChangeEncoder
from edit_model.edit_encoder.sequential_change_encoder import SequentialChangeEncoder
from edit_model.edit_encoder.tree_diff_encoder import TreeDiffEncoder


class EditEncoder(nn.Module):
    def __init__(self):
        super(EditEncoder, self).__init__()

    @staticmethod
    def build(args, vocab, embedder=None, **kwargs):
        if args['edit_encoder']['type'] == 'sequential':
            edit_encoder = SequentialChangeEncoder(args['encoder']['token_encoding_size'],
                                                   args['edit_encoder']['edit_encoding_size'],
                                                   args['edit_encoder']['change_tag_embed_size'],
                                                   vocab,
                                                   no_unchanged_token_encoding_in_diff_seq=args['edit_encoder']['no_unchanged_token_encoding_in_diff_seq'])
        elif args['edit_encoder']['type'] == 'graph':
            edit_encoder = GraphChangeEncoder(args['edit_encoder']['edit_encoding_size'],
                                              syntax_tree_embedder=embedder,
                                              layer_time_steps=args['edit_encoder']['layer_timesteps'],
                                              dropout=args['edit_encoder']['dropout'],
                                              gnn_use_bias_for_message_linear=args['edit_encoder']['use_bias_for_message_linear'],
                                              master_node_option=args['edit_encoder']['master_node_option'],
                                              connections=args['edit_encoder']['connections'])
        elif args['edit_encoder']['type'] == 'hybrid':
            edit_encoder = HybridChangeEncoder(token_encoding_size=args['encoder']['token_encoding_size'],
                                               change_vector_dim=args['edit_encoder']['edit_encoding_size'],
                                               syntax_tree_embedder=embedder,
                                               layer_timesteps=args['edit_encoder']['layer_timesteps'],
                                               dropout=args['edit_encoder']['dropout'],
                                               vocab=vocab,
                                               gnn_use_bias_for_message_linear=args['edit_encoder']['no_unchanged_token_encoding_in_diff_seq'])
        elif args['edit_encoder']['type'] == 'bag':
            edit_encoder = BagOfEditsChangeEncoder(embedder, vocab)
        elif args['edit_encoder']['type'] == 'treediff':
            edit_encoder = TreeDiffEncoder(graph_encoding_size=args['encoder']['token_encoding_size'],
                                           input_size=args['edit_encoder']['input_size'],
                                           change_vector_size=args['edit_encoder']['edit_encoding_size'],
                                           operators=kwargs['operators'],
                                           operator_embedding=kwargs['operator_embedding'],
                                           production_embedding=kwargs['production_embedding'],
                                           field_embedding=kwargs['field_embedding'],
                                           token_embedding=kwargs['token_embedding'],
                                           copy_syntax_token=args['edit_encoder']['copy_token'])
        else:
            raise ValueError('unknown code change encoder type [%s]' % args['edit_encoder']['type'])

        return edit_encoder
