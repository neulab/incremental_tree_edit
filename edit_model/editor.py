import string
from itertools import chain
from tqdm import tqdm
import sys
import numpy as np
import random
from collections import defaultdict

import torch
import torch.nn as nn

# programming languages
from asdl.transition_system import TransitionSystem
from asdl.lang.csharp.csharp_grammar import CSharpASDLGrammar
from asdl.lang.csharp.csharp_transition import CSharpTransitionSystem

# modules
from edit_model import utils, nn_utils
from edit_model.edit_encoder.edit_encoder import EditEncoder
from edit_model.encdec.sequential_encoder import SequentialEncoder
from edit_components.vocab import VocabEntry
from edit_model.embedder import CodeTokenEmbedder, SyntaxTreeEmbedder, EmbeddingTable, ConvolutionalCharacterEmbedder
from edit_model.encdec.graph_encoder import SyntaxTreeEncoder, TreeEncodingResult
from edit_model.encdec.encoder import EncodingResult
from edit_model.encdec.decoder import Decoder
from edit_model.encdec.sequential_decoder import SequentialDecoder
from edit_model.encdec.transition_decoder import TransitionDecoder
from edit_model.encdec.edit_decoder import IterativeDecoder

# tree edit related
from trees.substitution_system import SubstitutionSystem, AbstractSyntaxTree
from trees.substitution_system import Delete, Add, AddSubtree, Stop, ApplyRuleAction, GenTokenAction
from trees.hypothesis import Hypothesis


def _prepare_edit_encoding(model, edit_encoding):
    if isinstance(edit_encoding, np.ndarray):
        edit_encoding = torch.from_numpy(edit_encoding).to(model.device)
    if len(edit_encoding.size()) == 1:
        edit_encoding = edit_encoding.unsqueeze(0)

    return edit_encoding


class NeuralEditor(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 context_encoder,
                 edit_encoder: EditEncoder,
                 args):
        super(NeuralEditor, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.context_encoder = context_encoder
        self.edit_encoder = edit_encoder
        self.args = args

    @property
    def vocab(self):
        return self.encoder.vocab

    @property
    def device(self):
        return self.encoder.device

    def forward(self, examples, return_change_vectors=False, **kwargs):
        """
        Given a batch of examples, compute the edit encodings, and use the edit encodings to predict the target data,
        return the log-likelihood of generating the target data
        """

        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]
        context_data = [e.context for e in examples]

        # prepare shared encodings
        prev_data_encoding = self.encoder(prev_data)

        context_encoding = self.context_encoder(context_data)

        updated_data_encoding = self.encoder(updated_data)

        # compute edit encoding
        edit_encoding = self.edit_encoder(examples, prev_data_encoding, updated_data_encoding)

        results = self.decoder(examples, prev_data_encoding, context_encoding, edit_encoding)

        if return_change_vectors:
            results['edit_encoding'] = edit_encoding

        return results

    def get_edit_encoding(self, examples):
        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]

        prev_data_encoding = self.encoder(prev_data)
        updated_data_encoding = self.encoder(updated_data)

        edit_encoding = self.edit_encoder(examples, prev_data_encoding, updated_data_encoding)

        return edit_encoding

    def get_edit_encoding_by_batch(self, examples, batch_size=32, quiet=False):
        sorted_example_ids, example_old2new_pos = nn_utils.get_sort_map([len(c.change_seq) for c in examples])
        sorted_examples = [examples[i] for i in sorted_example_ids]

        change_vecs = []

        for batch_examples in tqdm(nn_utils.batch_iter(sorted_examples, batch_size), file=sys.stdout,
                                   total=len(examples) // batch_size, disable=quiet):

            batch_change_vecs = self.get_edit_encoding(batch_examples)
            change_vecs.append(batch_change_vecs)

        change_vecs = torch.cat(change_vecs, dim=0)
        # unsort all batch_examples
        change_vecs = change_vecs[example_old2new_pos]

        return change_vecs

    def decode_updated_data(self, example, edit_encoding=None, with_change_vec=False, beam_size=5, debug=False, **kwargs):
        prev_data = [example.prev_data]
        updated_data = [example.updated_data]
        context_data = [example.context]

        # prepare shared encodings
        prev_data_encoding = self.encoder(prev_data)

        context_encoding = self.context_encoder(context_data)

        updated_data_encoding = self.encoder(updated_data)

        if edit_encoding is not None:
            edit_encoding = _prepare_edit_encoding(self, edit_encoding)
        elif with_change_vec:
            edit_encoding = self.edit_encoder([example], prev_data_encoding, updated_data_encoding)
        else:
            edit_encoding = torch.zeros(1, self.args['edit_encoder']['edit_encoding_size'], device=self.device)

        hypotheses = self.decoder.beam_search_with_source_encodings(example.prev_data, prev_data_encoding,
                                                                    example.context, context_encoding,
                                                                    edit_encoding,
                                                                    beam_size=beam_size,
                                                                    max_decoding_time_step=70,
                                                                    debug=debug)

        return hypotheses

    @staticmethod
    def build(args, vocab=None, **kwargs):
        if not vocab:
            vocab = VocabEntry.load(args['dataset']['vocab_path'])

        language = args['lang']
        if language == 'csharp':
            grammar = CSharpASDLGrammar.from_roslyn_xml(open(args['dataset']['grammar_path']).read(),
                                                        pruning=args['dataset']['prune_grammar'])
            transition_system = CSharpTransitionSystem(grammar)

        else: # todo: add other PLs
            raise Exception("language %s is not supported by NeuralEditor!" % language)

        if args['mode'] == 'seq2seq':
            return Seq2SeqEditor.build(args, vocab=vocab, grammar=grammar, transition_system=transition_system)
        elif args['mode'] == 'graph2tree':
            return Graph2TreeEditor.build(args, vocab=vocab, grammar=grammar, transition_system=transition_system)
        elif args['mode'] == 'graph2iteredit':
            return Graph2IterEditEditor.build(args, vocab=vocab, grammar=grammar, transition_system=transition_system)

    def save(self, model_path, kwargs=None):
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }

        torch.save(params, model_path)

    @staticmethod
    def load(model_path, use_cuda=True):
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        kwargs = params['kwargs'] if params['kwargs'] is not None else dict()

        model = NeuralEditor.build(args, vocab=params['vocab'], **kwargs)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)
        model.eval()

        return model


class Seq2SeqEditor(NeuralEditor):
    # def __init__(self, *args, **kwargs):
        # super(Seq2SeqEditor, self).__init__(*args, **kwargs)
    def __init__(self, encoder, decoder, context_encoder, edit_encoder, args, **kwargs):
        super().__init__(encoder, decoder, context_encoder, edit_encoder, args)
        self.transition_system = kwargs.pop('transition_system', None)

    @staticmethod
    def build(args, vocab=None, grammar=None, transition_system=None):
        if args['edit_encoder']['type'] == 'graph':
            embedder = SyntaxTreeEmbedder(args['embedder']['token_embed_size'],
                                          vocab,
                                          grammar,
                                          node_embed_method=args['embedder']['node_embed_method'])
        else:
            embedder = CodeTokenEmbedder(args['embedder']['token_embed_size'], vocab)

        edit_encoder = EditEncoder.build(args, vocab=vocab, embedder=embedder)

        encoder = SequentialEncoder(args['embedder']['token_embed_size'],
                                    args['encoder']['token_encoding_size'],
                                    token_embedder=embedder,
                                    vocab=vocab)

        decoder = SequentialDecoder(args['embedder']['token_embed_size'],
                                    args['encoder']['token_encoding_size'],
                                    args['edit_encoder']['edit_encoding_size'],
                                    args['decoder']['hidden_size'],
                                    dropout=args['decoder']['dropout'],
                                    init_decode_vec_encoder_state_dropout=args['decoder']['init_decode_vec_encoder_state_dropout'],
                                    code_token_embedder=embedder,
                                    vocab=vocab,
                                    no_copy=not args['decoder']['copy_token'])

        editor = Seq2SeqEditor(encoder, decoder, context_encoder=encoder, edit_encoder=edit_encoder, args=args,
                               transition_system=transition_system)

        return editor


class Graph2TreeEditor(NeuralEditor):
    def __init__(self,
                 sequential_token_encoder,
                 graph_encoder,
                 decoder,
                 context_encoder,
                 edit_encoder: EditEncoder,
                 transition_system: TransitionSystem,
                 args):
        super().__init__(graph_encoder, decoder, context_encoder, edit_encoder, args)

        self.sequential_token_encoder = sequential_token_encoder
        self.transition_system = transition_system

    def forward(self, examples, change_vectors=None, return_change_vectors=False, **kwargs):
        """
        Given a batch of examples, compute the edit encodings, and use the edit encodings to predict the target data,
        return the log-likelihood of generating the target data
        """

        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]
        context_data = [e.context for e in examples]

        # prepare shared encodings
        prev_data_encoding = self.sequential_token_encoder(prev_data)

        context_encoding = self.context_encoder(context_data)

        # updated_data_encoding = self.sequential_token_encoder(updated_data)

        # compute edit encoding
        if change_vectors is not None:
            edit_encoding = change_vectors
        else:
            # edit_encoding = self.edit_encoder(examples, prev_data_encoding, updated_data_encoding)
            edit_encoding = self.get_edit_encoding(examples, prev_data_encoding=prev_data_encoding)

        prev_data_graph_encoding = self.encoder([e.prev_code_ast for e in examples], prev_data_encoding.encoding)

        results = self.decoder(examples, prev_data_graph_encoding, context_encoding, edit_encoding)

        if return_change_vectors:
            results['edit_encoding'] = edit_encoding

        return results

    def get_edit_encoding(self, examples, prev_data_encoding=None):
        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]

        if prev_data_encoding is None:
            prev_data_encoding = self.sequential_token_encoder(prev_data)

        if self.args['edit_encoder']['type'] == 'treediff':
            batch_edit_lengths = [len(e.tgt_edits) for e in examples]
            max_iteration_step = max(batch_edit_lengths)

            batch_edits_list, batch_inputs_list = [], []
            for t in range(max_iteration_step):
                batch_edits_list.append(
                    [e.tgt_edits[t] if t < len(e.tgt_edits) else e.tgt_edits[-1] for e in examples])
                batch_inputs_list.append(
                    [e.tgt_edits[t].meta['tree'] if t < len(e.tgt_edits) else e.tgt_edits[-1].meta['tree']
                     for e in examples])

            # graph encodings in each step
            cur_input_encodings_list = []
            for t in range(max_iteration_step):
                cur_input_encodings_list.append(self.encoder(batch_inputs_list[t], prev_data_encoding.encoding))
            init_input_encodings = cur_input_encodings_list[0]

            # context encoding
            context_data = [e.context for e in examples]
            context_encodings = self.context_encoder(context_data)

            # memory encoding
            batch_full_memory_encodings = init_input_encodings.encoding  # (batch_size, batch_max_node_num, hid)
            batch_memory_subtrees_size = [len(e.tgt_edits[-1].meta['memory']) for e in examples]
            max_memory_size = max(batch_memory_subtrees_size)
            batch_memory_encodings = []
            for e_idx, e in enumerate(examples):
                _valid_subtree_idx = [subtree.root_node.id for subtree in e.tgt_edits[-1].meta['memory']] + \
                                     [0] * (max_memory_size - batch_memory_subtrees_size[e_idx])
                batch_memory_encodings.append(batch_full_memory_encodings[e_idx][_valid_subtree_idx, :])
            batch_memory_encodings = torch.stack(batch_memory_encodings, dim=0)

            # compute decoder masks
            batch_max_node_num_over_time = max(cur_input_encodings.encoding.size(1)
                                               for cur_input_encodings in cur_input_encodings_list)
            batch_init_code_asts = batch_inputs_list[0]
            masks_cache = Graph2IterEditEditor.get_gen_and_copy_index_and_mask_over_time(
                batch_init_code_asts, context_data, batch_edits_list, context_encodings,
                init_input_encodings, batch_memory_encodings,
                batch_max_node_num_over_time,
                grammar=self.transition_system.grammar,
                vocab=self.vocab,
                operators=self.edit_encoder.operators,
                copy_syntax_token=self.decoder.copy_syntax_token,
                device=self.device)

            _cur_input_encodings_list = [cur_input_encodings.encoding for cur_input_encodings in cur_input_encodings_list]
            edit_encoding = self.edit_encoder(batch_edits_list, batch_edit_lengths, masks_cache,
                                              context_encodings.encoding, init_input_encodings.encoding,
                                              _cur_input_encodings_list, batch_memory_encodings)

        else:
            updated_data_encoding = self.sequential_token_encoder(updated_data)
            edit_encoding = self.edit_encoder(examples, prev_data_encoding, updated_data_encoding)

        return edit_encoding

    def decode_updated_data(self, example, edit_encoding=None, with_change_vec=False, beam_size=5, length_norm=False,
                            debug=False):
        prev_data = [example.prev_data]
        updated_data = [example.updated_data]
        context_data = [example.context]

        # prepare shared encodings
        prev_data_encoding = self.sequential_token_encoder(prev_data)

        context_encoding = self.context_encoder(context_data)

        # updated_data_encoding = self.sequential_token_encoder(updated_data)

        if edit_encoding is not None:
            edit_encoding = _prepare_edit_encoding(self, edit_encoding)
        elif with_change_vec:
            # edit_encoding = self.edit_encoder([example], prev_data_encoding, updated_data_encoding)
            edit_encoding = self.get_edit_encoding([example], prev_data_encoding=prev_data_encoding)
        else:
            edit_encoding = torch.zeros(1, self.args['edit_encoder']['edit_encoding_size'], device=self.device)

        prev_data_graph_encoding = self.encoder([example.prev_code_ast], prev_data_encoding.encoding)

        hypotheses = self.decoder.beam_search_with_source_encodings(example.prev_code_ast, prev_data_graph_encoding,
                                                                    example.context, context_encoding,
                                                                    edit_encoding,
                                                                    beam_size=beam_size,
                                                                    max_decoding_time_step=70,
                                                                    transition_system=self.transition_system,
                                                                    debug=debug)

        return hypotheses

    @staticmethod
    def build(args, vocab=None, grammar=None, transition_system=None):
        embedder = SyntaxTreeEmbedder(args['embedder']['token_embed_size'],
                                      vocab,
                                      grammar,
                                      node_embed_method=args['embedder']['node_embed_method'])
        seq_token_encoder = SequentialEncoder(args['embedder']['token_embed_size'],
                                              args['encoder']['token_encoding_size'],
                                              token_embedder=embedder,
                                              vocab=vocab)

        graph_encoder = SyntaxTreeEncoder(hidden_size=args['encoder']['token_encoding_size'],
                                          syntax_tree_embedder=embedder,
                                          layer_timesteps=args['encoder']['layer_timesteps'],
                                          residual_connections=args['encoder']['residual_connections'],
                                          connections=args['encoder']['connections'],
                                          gnn_use_bias_for_message_linear=args['encoder']['use_bias_for_message_linear'],
                                          dropout=args['encoder']['dropout'],
                                          vocab=vocab,
                                          grammar=grammar)

        transition_decoder = TransitionDecoder(args['encoder']['token_encoding_size'],
                                               args['edit_encoder']['edit_encoding_size'],
                                               args['decoder']['hidden_size'],
                                               args['decoder']['action_embed_size'],
                                               args['decoder']['field_embed_size'],
                                               dropout=args['decoder']['dropout'],
                                               init_decode_vec_encoder_state_dropout=args['decoder']['init_decode_vec_encoder_state_dropout'],
                                               vocab=vocab, grammar=grammar,
                                               syntax_tree_encoder=graph_encoder,
                                               use_syntax_token_rnn=args['decoder']['use_syntax_token_rnn'],
                                               no_penalize_apply_tree_when_copy_subtree=args['decoder']['no_penalize_apply_tree_when_copy_subtree'],
                                               encode_change_vec_in_syntax_token_rnn=args['decoder']['encode_change_vec_in_syntax_token_rnn'],
                                               feed_in_token_rnn_state_to_rule_rnn=args['decoder']['feed_in_token_rnn_state_to_rule_rnn'],
                                               fuse_rule_and_token_rnns=args['decoder']['fuse_rule_and_token_rnns'],
                                               decoder_init_method=args['decoder']['init_method'],
                                               copy_identifier_node=args['decoder']['copy_identifier_node'],
                                               copy_syntax_token=args['decoder']['copy_token'],
                                               copy_sub_tree=args['decoder']['copy_subtree'])

        if args['edit_encoder']['type'] == 'treediff':
            operators = ['delete', 'add', 'add_subtree', 'stop']
            operator_embedding = nn.Embedding(len(operators), args['edit_encoder']['operator_embed_size'])
            edit_encoder = EditEncoder.build(args, vocab=vocab, embedder=embedder,
                                             operators=operators,
                                             operator_embedding=operator_embedding,
                                             production_embedding=transition_decoder.production_embedding,
                                             field_embedding=transition_decoder.field_embedding,
                                             token_embedding=transition_decoder.token_embedding)
        else:
            edit_encoder = EditEncoder.build(args, vocab=vocab, embedder=embedder)

        editor = Graph2TreeEditor(seq_token_encoder, graph_encoder, transition_decoder,
                                  context_encoder=seq_token_encoder,
                                  edit_encoder=edit_encoder,
                                  transition_system=transition_system,
                                  args=args)

        return editor

    def save(self, model_path):
        NeuralEditor.save(self, model_path, kwargs=dict(grammar=self.decoder.grammar))


class Graph2IterEditEditor(NeuralEditor):
    def __init__(self,
                 controller,
                 sequential_token_encoder,
                 graph_encoder,
                 decoder,
                 context_encoder,
                 edit_encoder: EditEncoder,
                 substitution_system: SubstitutionSystem,
                 args):
        super().__init__(graph_encoder, decoder, context_encoder, edit_encoder, args)

        self.controller = controller # LSTMCell
        self.sequential_token_encoder = sequential_token_encoder
        self.substitution_system = substitution_system
        self.transition_system = substitution_system.transition_system

    def init_global_states(self, batch_examples):
        return torch.zeros(len(batch_examples), self.controller.hidden_size, dtype=torch.float).to(self.device), \
               torch.zeros(len(batch_examples), self.controller.hidden_size, dtype=torch.float).to(self.device)

    def input_aggregation(self, input_encoding, input_mask):
        """
        Mean pooling aggregation on input graph.
        :param input_encoding: (batch_size, num_of_elements, hidden_size).
        :param input_mask: (batch_size, num_of_elements), with 1 for padding.
        :return: (batch_size, hidden_size)
        """
        return input_encoding.sum(dim=1) / ((1. - input_mask.float()).sum(dim=-1, keepdim=True))

    def forward(self, examples, change_vectors=None, return_change_vectors=False, train_components=None, with_gold_edits=False):
        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]
        context_data = [e.context for e in examples]

        batch_edit_lengths = [len(e.tgt_actions) for e in examples]
        max_iteration_step = max(batch_edit_lengths)
        batch_edit_mask = torch.zeros(len(examples), max_iteration_step, dtype=torch.float).to(self.device)
        for example_id, edit_length in enumerate(batch_edit_lengths):
            if hasattr(examples[example_id], 'tgt_actions_weight'):
                weights = examples[example_id].tgt_actions_weight
                batch_edit_mask[example_id, :edit_length] = torch.Tensor(weights).to(self.device)
            else:
                batch_edit_mask[example_id, :edit_length] = 1.
        batch_edit_mask = batch_edit_mask.transpose(1, 0) # (max_iteration_step, batch_size)

        if with_gold_edits:  # check gold_edits (which should be used to calculate edit enc)
            replaced_example_idxs = [_idx for _idx, e in enumerate(examples) if hasattr(e, 'gold_edits')]
            extended_max_iteration_step = max([len(examples[_idx].gold_edits) for _idx in replaced_example_idxs] +
                                              [max_iteration_step])

        batch_edits_list, batch_inputs_list = [], []
        for t in range(max_iteration_step):
            batch_edits_list.append([e.tgt_actions[t if t < len(e.tgt_actions) else -1] for e in examples])
            batch_inputs_list.append([e.tgt_actions[t if t < len(e.tgt_actions) else -1].meta['tree'] for e in examples])

        # prepare shared encodings
        prev_data_encodings = self.sequential_token_encoder(prev_data)
        init_input_encodings = self.encoder(batch_inputs_list[0], prev_data_encodings.encoding)
        context_encodings = self.context_encoder(context_data)

        # graph encodings in each step
        cur_input_encodings_list = [init_input_encodings]
        replaced_cur_input_encodings_list = [None]
        for t in range(1, max_iteration_step):
            if with_gold_edits and len(replaced_example_idxs):
                replaced_batch_inputs_t = [examples[_idx].gold_edits[t if t < len(examples[_idx].gold_edits) else -1].meta['tree']
                                           for _idx in replaced_example_idxs]
                extended_prev_data_encodings = torch.cat([prev_data_encodings.encoding,
                                                          prev_data_encodings.encoding[replaced_example_idxs]])
                extended_cur_input_encodings = self.encoder(batch_inputs_list[t] + replaced_batch_inputs_t,
                                                            extended_prev_data_encodings)
                cur_input_encodings = TreeEncodingResult(extended_cur_input_encodings.data[:len(examples)],
                                                         extended_cur_input_encodings.encoding[:len(examples)],
                                                         extended_cur_input_encodings.mask[:len(examples)],
                                                         extended_cur_input_encodings.syntax_token_mask[:len(examples)])
                cur_input_encodings_list.append(cur_input_encodings)

                replaced_cur_input_encodings = []
                _replaced_pointer = len(examples)
                for _idx in range(len(examples)):
                    if _idx in replaced_example_idxs:
                        replaced_cur_input_encodings.append(extended_cur_input_encodings.encoding[_replaced_pointer])
                        _replaced_pointer += 1
                    else:
                        replaced_cur_input_encodings.append(extended_cur_input_encodings.encoding[_idx])
                replaced_cur_input_encodings_list.append(torch.stack(replaced_cur_input_encodings))
            else:
                cur_input_encodings_list.append(self.encoder(batch_inputs_list[t], prev_data_encodings.encoding))
        if with_gold_edits and len(replaced_example_idxs) and extended_max_iteration_step > max_iteration_step:
            for t in range(max_iteration_step, extended_max_iteration_step):
                replaced_batch_inputs_t = []
                for _idx in range(len(examples)):
                    e = examples[_idx]
                    if _idx in replaced_example_idxs:
                        replaced_batch_inputs_t.append(e.gold_edits[t if t < len(e.gold_edits) else -1].meta['tree'])
                    else:
                        replaced_batch_inputs_t.append(e.tgt_actions[-1].meta['tree'])
                replaced_cur_input_encodings = self.encoder(replaced_batch_inputs_t, prev_data_encodings.encoding)
                replaced_cur_input_encodings_list.append(replaced_cur_input_encodings.encoding)

        # memory encoding
        batch_full_memory_encodings = init_input_encodings.encoding # (batch_size, batch_max_node_num, hid)
        batch_memory_subtrees_size = [len(e.tgt_actions[-1].meta['memory']) for e in examples]
        max_memory_size = max(batch_memory_subtrees_size)
        batch_memory_encodings = []
        for e_idx, e in enumerate(examples):
            _valid_subtree_idx = [subtree.root_node.id for subtree in e.tgt_actions[-1].meta['memory']] + \
                                 [0] * (max_memory_size - batch_memory_subtrees_size[e_idx])
            batch_memory_encodings.append(batch_full_memory_encodings[e_idx][_valid_subtree_idx, :])
        batch_memory_encodings = torch.stack(batch_memory_encodings, dim=0)

        # compute decoder masks
        batch_max_node_num_over_time = max(cur_input_encodings.encoding.size(1)
                                           for cur_input_encodings in cur_input_encodings_list)
        batch_init_code_asts = batch_inputs_list[0]
        masks_cache = Graph2IterEditEditor.get_gen_and_copy_index_and_mask_over_time(
            batch_init_code_asts, context_data, batch_edits_list, context_encodings,
            init_input_encodings, batch_memory_encodings,
            batch_max_node_num_over_time,
            grammar=self.transition_system.grammar,
            vocab=self.vocab,
            operators=self.decoder.operators,
            copy_syntax_token=self.decoder.copy_syntax_token,
            device=self.device)

        # compute edit encoding
        if change_vectors is not None:
            edit_encodings = change_vectors
        else:
            if self.args['edit_encoder']['type'] == 'treediff':
                if with_gold_edits and len(replaced_example_idxs):
                    _batch_edits_list = []
                    actual_gold_edits_seq = [e.gold_edits if hasattr(e, 'gold_edits') else e.tgt_actions for e in examples]
                    _batch_edit_lengths = [len(actual_gold_edits) for actual_gold_edits in actual_gold_edits_seq]
                    for t in range(extended_max_iteration_step):
                        _batch_edits_list.append([actual_gold_edits[t if t < len(actual_gold_edits) else -1]
                                                 for actual_gold_edits in actual_gold_edits_seq])
                    _cur_input_encodings_list = [init_input_encodings.encoding] + replaced_cur_input_encodings_list[1:]
                    _batch_max_node_num_over_time = max(_cur_input_encodings.size(1) for _cur_input_encodings in _cur_input_encodings_list)

                    _masks_cache = Graph2IterEditEditor.get_gen_and_copy_index_and_mask_over_time(
                        batch_init_code_asts, context_data, _batch_edits_list, context_encodings,
                        init_input_encodings, batch_memory_encodings,
                        _batch_max_node_num_over_time,
                        grammar=self.transition_system.grammar,
                        vocab=self.vocab,
                        operators=self.decoder.operators,
                        copy_syntax_token=self.decoder.copy_syntax_token,
                        device=self.device)
                else:
                    _cur_input_encodings_list = [cur_input_encodings.encoding for cur_input_encodings in
                                                 cur_input_encodings_list]
                    _masks_cache = masks_cache
                    _batch_edits_list = batch_edits_list
                    _batch_edit_lengths = batch_edit_lengths

                edit_encodings = self.edit_encoder(_batch_edits_list, _batch_edit_lengths, _masks_cache,
                                                   context_encodings.encoding, init_input_encodings.encoding,
                                                   _cur_input_encodings_list, batch_memory_encodings)
            else:
                updated_data_encodings = self.sequential_token_encoder(updated_data)
                edit_encodings = self.edit_encoder(examples, prev_data_encodings, updated_data_encodings)

        # compute edit sequence inputs
        last_global_hidden_states, last_global_cell_states = self.init_global_states(examples)
        global_hidden_states_list = []
        for t in range(max_iteration_step):
            global_hidden_states, global_cell_states = self.controller(
                torch.cat([edit_encodings, self.input_aggregation(cur_input_encodings_list[t].encoding,
                                                                  cur_input_encodings_list[t].mask)], dim=1),
                (last_global_hidden_states, last_global_cell_states))
            global_hidden_states_list.append(global_hidden_states)

            last_global_hidden_states = global_hidden_states
            last_global_cell_states = global_cell_states

        returns = self.decoder(examples, batch_edits_list, global_hidden_states_list, context_encodings,
                               init_input_encodings, batch_memory_encodings, cur_input_encodings_list,
                               masks_cache, train_components=train_components)

        log_probs = returns['log_probs'] # (max_iteration_step, batch_size)
        if self.args['trainer']['loss_avg_step']:
            gated_log_probs = torch.sum(log_probs * batch_edit_mask, dim=0) / torch.sum(batch_edit_mask, dim=0)
        else:
            gated_log_probs = torch.sum(log_probs * batch_edit_mask, dim=0)

        results = {'log_probs': gated_log_probs,
                   'ungated_log_probs': log_probs,
                   'batch_edit_mask': batch_edit_mask}  # (batch_size)
        for key, value in returns.items():
            if key not in results:
                results[key] = value

        if return_change_vectors:
            results['edit_encoding'] = edit_encodings

        return results

    def get_edit_encoding(self, examples, prev_data_encodings=None, updated_data_encodings=None):
        prev_data = [e.prev_data for e in examples]
        updated_data = [e.updated_data for e in examples]

        if prev_data_encodings is None:
            prev_data_encodings = self.sequential_token_encoder(prev_data)

        if self.args['edit_encoder']['type'] == 'treediff':
            actual_tgt_actions_list = [e.tgt_actions for e in examples]
            # for e in examples:
            #     if hasattr(e, 'gold_edits'):
            #         actual_tgt_actions_list.append(e.gold_edits)
            #     else:
            #         actual_tgt_actions_list.append(e.tgt_actions)

            batch_edit_lengths = [len(tgt_actions) for tgt_actions in actual_tgt_actions_list]
            max_iteration_step = max(batch_edit_lengths)

            batch_edits_list, batch_inputs_list = [], []
            for t in range(max_iteration_step):
                batch_edits_list.append([tgt_actions[t] if t < len(tgt_actions) else tgt_actions[-1]
                                         for tgt_actions in actual_tgt_actions_list])
                batch_inputs_list.append([tgt_actions[t].meta['tree'] if t < len(tgt_actions) else tgt_actions[-1].meta['tree']
                                          for tgt_actions in actual_tgt_actions_list])
                # batch_edits_list.append(
                #     [e.tgt_actions[t] if t < len(e.tgt_actions) else e.tgt_actions[-1] for e in examples])
                # batch_inputs_list.append(
                #     [e.tgt_actions[t].meta['tree'] if t < len(e.tgt_actions) else e.tgt_actions[-1].meta['tree']
                #      for e in examples])

            # graph encodings in each step
            cur_input_encodings_list = []
            for t in range(max_iteration_step):
                cur_input_encodings_list.append(self.encoder(batch_inputs_list[t], prev_data_encodings.encoding))
            init_input_encodings = cur_input_encodings_list[0]

            # context encoding
            context_data = [e.context for e in examples]
            context_encodings = self.context_encoder(context_data)

            # memory encoding
            batch_full_memory_encodings = init_input_encodings.encoding  # (batch_size, batch_max_node_num, hid)
            # batch_memory_subtrees_size = [len(e.tgt_actions[-1].meta['memory']) for e in examples]
            batch_memory_subtrees_size = [len(tgt_actions[-1].meta['memory']) for tgt_actions in actual_tgt_actions_list]
            max_memory_size = max(batch_memory_subtrees_size)
            batch_memory_encodings = []
            for e_idx, e in enumerate(examples):
                # _valid_subtree_idx = [subtree.root_node.id for subtree in e.tgt_actions[-1].meta['memory']] + \
                #                      [0] * (max_memory_size - batch_memory_subtrees_size[e_idx])
                _valid_subtree_idx = [subtree.root_node.id for subtree in actual_tgt_actions_list[e_idx][-1].meta['memory']] + \
                                     [0] * (max_memory_size - batch_memory_subtrees_size[e_idx])
                batch_memory_encodings.append(batch_full_memory_encodings[e_idx][_valid_subtree_idx, :])
            batch_memory_encodings = torch.stack(batch_memory_encodings, dim=0)

            # compute decoder masks
            batch_max_node_num_over_time = max(cur_input_encodings.encoding.size(1)
                                               for cur_input_encodings in cur_input_encodings_list)
            batch_init_code_asts = batch_inputs_list[0]
            masks_cache = Graph2IterEditEditor.get_gen_and_copy_index_and_mask_over_time(
                batch_init_code_asts, context_data, batch_edits_list, context_encodings,
                init_input_encodings, batch_memory_encodings,
                batch_max_node_num_over_time,
                grammar=self.transition_system.grammar,
                vocab=self.vocab,
                operators=self.decoder.operators,
                copy_syntax_token=self.decoder.copy_syntax_token,
                device=self.device)

            _cur_input_encodings_list = [cur_input_encodings.encoding for cur_input_encodings in cur_input_encodings_list]
            edit_encodings = self.edit_encoder(batch_edits_list, batch_edit_lengths, masks_cache,
                                               context_encodings.encoding, init_input_encodings.encoding,
                                               _cur_input_encodings_list, batch_memory_encodings)
        else:
            if updated_data_encodings is None:
                updated_data_encodings = self.sequential_token_encoder(updated_data)
            edit_encodings = self.edit_encoder(examples, prev_data_encodings, updated_data_encodings)

        return edit_encodings

    def decode_updated_data(self, example, edit_encoding=None, with_change_vec=False, beam_size=5, length_norm=False,
                            debug=False):
        prev_data = [example.prev_data]
        updated_data = [example.updated_data]
        context_data = [example.context]

        # prepare shared encodings
        prev_data_encodings = self.sequential_token_encoder(prev_data)

        context_encodings = self.context_encoder(context_data)

        # updated_data_encodings = self.sequential_token_encoder(updated_data)

        if edit_encoding is not None:
            edit_encodings = _prepare_edit_encoding(self, edit_encoding)
        elif with_change_vec:
            # edit_encodings = self.edit_encoder([example], prev_data_encodings, updated_data_encodings)
            edit_encodings = self.get_edit_encoding([example], prev_data_encodings=prev_data_encodings)
        else:
            edit_encodings = torch.zeros(1, self.args['edit_encoder']['edit_encoding_size'], device=self.device)

        hypotheses = self.decoder.beam_search_with_source_encodings(self.controller, self.encoder, example.prev_code_ast,
                                                                    context_data[0], self.init_global_states([example]),
                                                                    prev_data_encodings, context_encodings,
                                                                    edit_encodings, self.input_aggregation,
                                                                    self.substitution_system,
                                                                    beam_size=beam_size,
                                                                    max_iteration_step=70,
                                                                    length_norm=length_norm)

        return hypotheses

    def decode_updated_data_in_batch(self, examples, max_trajectory_length=70, edit_encodings=None,
                                     meta_tree=False, return_root_node=True):
        """ Decoding in batch. Batchwise graph encoding and global state progresses.
            This function restricts: (1) memory setup = ('all_init', 'joint');
            (2) beam size=1. Length normalization is not supported. """

        def _slice_EncodingResult(EncodingResult_instance, idx):
            if isinstance(EncodingResult_instance, TreeEncodingResult):
                # ['data', 'encoding', 'mask', 'syntax_token_mask']
                return TreeEncodingResult(EncodingResult_instance.data[idx:idx+1],
                                          EncodingResult_instance.encoding[idx:idx+1],
                                          EncodingResult_instance.mask[idx:idx+1],
                                          EncodingResult_instance.syntax_token_mask[idx:idx+1])
            elif isinstance(EncodingResult_instance, EncodingResult):
                # ['data', 'encoding', 'last_state', 'last_cell', 'mask']
                return EncodingResult(EncodingResult_instance.data[idx:idx+1],
                                      EncodingResult_instance.encoding[idx:idx+1],
                                      None, None,
                                      EncodingResult_instance.mask[idx:idx+1])
            else:
                raise Exception('Undetected EncodingResult type.')

        prev_data = [example.prev_data for example in examples]
        context_data = [example.context for example in examples]

        # prepare shared encodings
        prev_data_encodings = self.sequential_token_encoder(prev_data)
        context_encodings = self.context_encoder(context_data)
        if edit_encodings is not None:
            edit_encodings = _prepare_edit_encoding(self, edit_encodings)
        else:
            edit_encodings = self.get_edit_encoding(examples, prev_data_encodings=prev_data_encodings)

        # initial setup
        # init_code_ast_list = [example.prev_code_ast.copy_and_reindex_w_dummy_reduce() for example in examples]
        assert hasattr(examples[0].prev_code_ast, 'dummy_node_ids') and examples[0].prev_code_ast.dummy_node_ids is not None
        init_code_ast_list = [example.prev_code_ast for example in examples]
        init_input_encodings = self.encoder(init_code_ast_list, prev_data_encodings.encoding)
        hypotheses = []
        for e_idx, init_code_ast in enumerate(init_code_ast_list):
            hyp = Hypothesis(init_code_ast, bool_copy_subtree=self.decoder.copy_subtree,
                             memory=[subtree.root_node for subtree in examples[e_idx].tgt_actions[-1].meta['memory']])
            hyp.meta = {'idx': e_idx}
            if meta_tree:
                hyp.meta['tree'] = [hyp.tree.copy()]
            hypotheses.append(hyp)

        # memory setup: initial_memory_encodings of (1, memory_size, hidden_size)
        batch_full_memory_encodings = init_input_encodings.encoding  # (batch_size, batch_max_node_num, hid)
        # batch_memory_subtrees_size = [len(e.tgt_actions[-1].meta['memory']) for e in examples]
        batch_memory_encodings = []
        for e_idx, e in enumerate(examples):
            _valid_subtree_idx = [subtree.root_node.id for subtree in e.tgt_actions[-1].meta['memory']]
            batch_memory_encodings.append(batch_full_memory_encodings[e_idx][_valid_subtree_idx, :])
        # batch_memory_encodings = torch.stack(batch_memory_encodings, dim=0)

        last_global_hidden_states, last_global_cell_states = self.init_global_states(examples)
        origin_idx2hyp = [None for _ in range(len(examples))]

        t = 0
        cur_prev_data_encodings = prev_data_encodings.encoding
        cur_edit_encodings = edit_encodings
        while t < max_trajectory_length:
            batch_cur_inputs = [hyp.tree for hyp in hypotheses]
            cur_input_encodings = self.encoder(batch_cur_inputs, cur_prev_data_encodings)

            global_hidden_states, global_cell_states = self.controller(
                torch.cat([cur_edit_encodings, self.input_aggregation(cur_input_encodings.encoding,
                                                                      cur_input_encodings.mask)], dim=1),
                (last_global_hidden_states, last_global_cell_states))

            # sample the next action
            new_hypotheses = []
            new_hypotheses_h_idx = []
            for h_idx, hyp in enumerate(hypotheses):
                origin_e_idx = hypotheses[h_idx].meta['idx']

                top_partial_hypotheses = self.decoder.one_step_beam_search_with_source_encodings(
                    [hyp], init_code_ast_list[origin_e_idx], context_data[origin_e_idx],
                    global_hidden_states[h_idx:(h_idx + 1)],
                    _slice_EncodingResult(context_encodings, origin_e_idx),
                    _slice_EncodingResult(init_input_encodings, origin_e_idx),
                    batch_memory_encodings[origin_e_idx].unsqueeze(0),
                    _slice_EncodingResult(cur_input_encodings, h_idx), self.substitution_system,
                    beam_size=1, time_step=t)
                if top_partial_hypotheses:
                    new_hyp = top_partial_hypotheses[0]
                    if not new_hyp.stopped:  # the last edit is Stop
                        if meta_tree:
                            new_hyp.meta['tree'].append(new_hyp.tree.copy())
                        new_hypotheses.append(new_hyp)
                        new_hypotheses_h_idx.append(h_idx)
                    else:
                        if return_root_node:
                            new_hyp.tree = new_hyp.tree.root_node
                        origin_idx2hyp[origin_e_idx] = new_hyp
                else:
                    if return_root_node:
                        hyp.tree = hyp.tree.root_node
                    origin_idx2hyp[origin_e_idx] = hyp

            if len(new_hypotheses) == 0:
                hypotheses.clear()
                break # all hypotheses have ended

            # update inputs for surviving hypotheses
            last_global_hidden_states, last_global_cell_states = [], []
            cur_prev_data_encodings, cur_edit_encodings = [], []
            for valid_h_idx, valid_hyp in zip(new_hypotheses_h_idx, new_hypotheses):
                origin_e_idx = valid_hyp.meta['idx']

                last_global_hidden_states.append(global_hidden_states[valid_h_idx])
                last_global_cell_states.append(global_cell_states[valid_h_idx])

                cur_prev_data_encodings.append(prev_data_encodings.encoding[origin_e_idx])
                cur_edit_encodings.append(edit_encodings[origin_e_idx])

            last_global_hidden_states = torch.stack(last_global_hidden_states)
            last_global_cell_states = torch.stack(last_global_cell_states)
            cur_prev_data_encodings = torch.stack(cur_prev_data_encodings)
            cur_edit_encodings = torch.stack(cur_edit_encodings)

            hypotheses = new_hypotheses

            t += 1

        if len(hypotheses):
            for h_idx, hyp in enumerate(hypotheses):
                origin_e_idx = hypotheses[h_idx].meta['idx']
                if return_root_node:
                    hyp.tree = hyp.tree.root_node
                origin_idx2hyp[origin_e_idx] = hyp

        return origin_idx2hyp

    def interactive_decode_updated_data(self, example, edit_encoding=None, with_change_vec=False,
                                        beam_size=5, length_norm=False, max_interaction=5):
        prev_data = [example.prev_data]
        context_data = [example.context]

        # prepare shared encodings
        prev_data_encodings = self.sequential_token_encoder(prev_data)

        context_encodings = self.context_encoder(context_data)

        if edit_encoding is not None:
            edit_encodings = _prepare_edit_encoding(self, edit_encoding)
        elif with_change_vec:
            # edit_encodings = self.edit_encoder([example], prev_data_encodings, updated_data_encodings)
            edit_encodings = self.get_edit_encoding([example], prev_data_encodings=prev_data_encodings)
        else:
            edit_encodings = torch.zeros(1, self.args['edit_encoder']['edit_encoding_size'], device=self.device)

        interaction_hyp_list, gold_edit_list, gold_steps = [], [], []
        for interaction_round in range(0, max_interaction + 1):
            if interaction_round == 0:
                hypotheses = self.decoder.beam_search_with_source_encodings(self.controller, self.encoder,
                                                                            example.prev_code_ast,
                                                                            context_data[0],
                                                                            self.init_global_states([example]),
                                                                            prev_data_encodings, context_encodings,
                                                                            edit_encodings, self.input_aggregation,
                                                                            self.substitution_system,
                                                                            beam_size=beam_size,
                                                                            max_iteration_step=70,
                                                                            length_norm=length_norm,
                                                                            return_ast=True)
            else:
                # update preset_hyp with gold edit
                new_partial_tgt_edits = self.substitution_system.get_decoding_edits_fast(
                    hyp.tree, example.updated_code_ast,
                    bool_copy_subtree=self.args['decoder']['copy_subtree'],
                    preset_memory=hyp.memory)
                gold_edit = new_partial_tgt_edits[0]
                assert not isinstance(gold_edit, Stop)
                gold_edit_list.append(gold_edit)
                gold_steps.append(len(hyp.edits))

                hyp.edits.append(gold_edit)
                hyp.score_per_edit.append(1.0)
                hyp.tree = new_partial_tgt_edits[1].meta['tree']
                hyp.stop_t = None
                hyp.update_frontier_info()

                hypotheses = self.decoder.beam_search_with_source_encodings(self.controller, self.encoder,
                                                                            example.prev_code_ast,
                                                                            context_data[0],
                                                                            None,
                                                                            prev_data_encodings, context_encodings,
                                                                            edit_encodings, self.input_aggregation,
                                                                            self.substitution_system,
                                                                            beam_size=beam_size,
                                                                            max_iteration_step=70,
                                                                            length_norm=length_norm,
                                                                            preset_hyp=hyp,
                                                                            return_ast=True)

            if hypotheses:
                interaction_hyp_list.append(hypotheses[0].tree.root_node.to_string())
                if hypotheses[0].tree.root_node == example.updated_code_ast.root_node:
                    del hypotheses[0].meta
                    setattr(hypotheses[0], 'gold_steps', gold_steps)
                    return 1., interaction_round, hypotheses[0], interaction_hyp_list, gold_edit_list
            else:
                interaction_hyp_list.append(None)
                return 0., interaction_round, None, interaction_hyp_list, gold_edit_list # not valid hyp to continue

            hyp = hypotheses[0]

        del hyp.meta
        setattr(hyp, 'gold_steps', gold_steps)
        return 0., max_interaction, hyp, interaction_hyp_list, gold_edit_list

    def decode_with_gold_sample(self, example, sampling_probability, max_trajectory_length,
                                edit_encoding=None, existing_non_gold_edit_seqs=None, prioritize_last_edit=True,
                                extend_stop=False, debug=False):
        """
        Decoding for imitation learning.
        :param example: input example.
        :param sampling_probability: probability of sampling from the model policy.
        :param max_trajectory_length: maximum sampling length.
        :param edit_encoding: pre-computed edit representation.
        :param existing_non_gold_edit_seqs: a list of non-gold edit sequences.
        :param prioritize_last_edit: whether to prioritize gold edits around the last actual edit action.
        :param extend_stop: whether to include gold edit seq after the model stops.
        :return: gold_edits, actual_edits, actual_decoded_tree.
        """
        def _compare_two_trees(tree1, tree2):
            return tree1.root_node.to_string() == tree2.root_node.to_string()

        def _refresh_tgt_edits(cur_tree, cur_partial_tgt_edits, tgt_code_ast, preset_memory,
                               last_edit_field_node=None):
            if cur_partial_tgt_edits is not None and \
                    _compare_two_trees(cur_tree, cur_partial_tgt_edits[0].meta['tree']):
                return cur_partial_tgt_edits
            else:
                if existing_non_gold_edit_seqs is not None:
                    # check if the same code happened in history
                    for existing_edit_seq in existing_non_gold_edit_seqs:
                        for existing_edit_idx, existing_edit in enumerate(existing_edit_seq):
                            if _compare_two_trees(cur_tree, existing_edit.meta['tree']):
                                return existing_edit_seq[existing_edit_idx:]

                new_partial_tgt_edits = self.substitution_system.get_decoding_edits_fast(
                    cur_tree, tgt_code_ast,
                    bool_copy_subtree=self.args['decoder']['copy_subtree'],
                    preset_memory=preset_memory,
                    last_edit_field_node=last_edit_field_node,
                    bool_debug=debug)
                return new_partial_tgt_edits

        prev_data = [example.prev_data]
        updated_data = [example.updated_data]
        context_data = [example.context]

        # prepare shared encodings
        prev_data_encodings = self.sequential_token_encoder(prev_data)
        context_encodings = self.context_encoder(context_data)
        if edit_encoding is not None:
            edit_encodings = _prepare_edit_encoding(self, edit_encoding)
        else:
            edit_encodings = self.get_edit_encoding([example], prev_data_encodings=prev_data_encodings)

        # initial setup
        init_code_ast = example.prev_code_ast
        init_code_ast = init_code_ast.copy_and_reindex_w_dummy_reduce()
        init_input_encodings = self.encoder([init_code_ast], prev_data_encodings.encoding)
        hyp = Hypothesis(init_code_ast, bool_copy_subtree=self.decoder.copy_subtree,
                         memory=[subtree.root_node for subtree in example.tgt_actions[-1].meta['memory']])
        hyp.meta = {}

        # memory setup: initial_memory_encodings of (1, memory_size, hidden_size)
        initial_full_memory_encodings = init_input_encodings.encoding
        valid_subtree_idx = [subtree.id for subtree in hyp.memory]
        initial_memory_encodings = initial_full_memory_encodings[:1, valid_subtree_idx, :]
        memory_encodings = initial_memory_encodings

        last_global_hidden_states, last_global_cell_states = self.init_global_states([example])
        gold_edits, actual_edits = [], []

        t = 0
        cur_partial_tgt_edits = example.tgt_actions
        actual_max_trajectory_length = max(max_trajectory_length, len(example.tgt_actions))
        while t < actual_max_trajectory_length:
            cur_input = hyp.tree
            cur_input_encodings = self.encoder([cur_input], prev_data_encodings.encoding)

            global_hidden_states, global_cell_states = self.controller(
                torch.cat([edit_encodings, self.input_aggregation(cur_input_encodings.encoding,
                                                                  cur_input_encodings.mask)], dim=1),
                (last_global_hidden_states, last_global_cell_states))
            last_global_hidden_states = global_hidden_states
            last_global_cell_states = global_cell_states

            new_partial_tgt_edits = _refresh_tgt_edits(cur_input, cur_partial_tgt_edits,
                                                       example.updated_code_ast,
                                                       preset_memory=hyp.memory,
                                                       last_edit_field_node=hyp.last_edit_field_node if prioritize_last_edit else None)
            gold_edit = new_partial_tgt_edits[0] # this should already include all meta data necessary for re-training.
            gold_edits.append(gold_edit)

            # sample the next action
            p = random.random()
            if p < sampling_probability: # sample from own policy
                top_partial_hypotheses = self.decoder.one_step_beam_search_with_source_encodings(
                    [hyp], init_code_ast, context_data[0], global_hidden_states, context_encodings,
                    init_input_encodings, memory_encodings, cur_input_encodings, self.substitution_system,
                    beam_size=1, time_step=t)
                if top_partial_hypotheses:
                    hyp = top_partial_hypotheses[0]
                    actual_edits.append(hyp.edits[-1])
                    if extend_stop and (hyp.stopped or len(hyp.edits) >= actual_max_trajectory_length) and \
                            not isinstance(gold_edit, Stop):
                        gold_edits.extend(new_partial_tgt_edits[1:])
                    if hyp.stopped: # the last edit is Stop
                        break
                else: # failed
                    break
            else:
                actual_edits.append(gold_edit)
                if len(new_partial_tgt_edits) > 1:
                    hyp.apply_edit(gold_edit)
                else:
                    assert isinstance(gold_edit, Stop)
                    break

            if len(new_partial_tgt_edits) > 1:
                cur_partial_tgt_edits = new_partial_tgt_edits[1:]
            else:
                cur_partial_tgt_edits = None
            t += 1

        # return training examples
        if 'memory' not in gold_edits[-1].meta:
            gold_edits[-1].meta['memory'] = example.tgt_actions[-1].meta['memory']
        return gold_edits, actual_edits, hyp.tree.root_node

    def decode_with_gold_sample_in_batch(self, examples, sampling_probability, max_trajectory_length,
                                         edit_encodings=None, existing_non_gold_edit_seqs_list=None,
                                         prioritize_last_edit=True, extend_stop=False, debug=False):

        def _compare_two_trees(tree1, tree2):
            return tree1.root_node == tree2.root_node

        def _refresh_tgt_edits(cur_tree, cur_partial_tgt_edits, tgt_code_ast, preset_memory,
                               existing_non_gold_edit_seqs=None, last_edit_field_node=None):
            if cur_partial_tgt_edits is not None and \
                    _compare_two_trees(cur_tree, cur_partial_tgt_edits[0].meta['tree']):
                return cur_partial_tgt_edits
            else:
                if existing_non_gold_edit_seqs is not None:
                    # check if the same code happened in history
                    for existing_edit_seq in existing_non_gold_edit_seqs:
                        for existing_edit_idx, existing_edit in enumerate(existing_edit_seq):
                            if _compare_two_trees(cur_tree, existing_edit.meta['tree']):
                                return existing_edit_seq[existing_edit_idx:]

                new_partial_tgt_edits = self.substitution_system.get_decoding_edits_fast(
                    cur_tree, tgt_code_ast,
                    bool_copy_subtree=self.args['decoder']['copy_subtree'],
                    preset_memory=preset_memory,
                    last_edit_field_node=last_edit_field_node,
                    bool_debug=debug)
                return new_partial_tgt_edits

        def _slice_EncodingResult(EncodingResult_instance, idx):
            if isinstance(EncodingResult_instance, TreeEncodingResult):
                # ['data', 'encoding', 'mask', 'syntax_token_mask']
                return TreeEncodingResult(EncodingResult_instance.data[idx:idx+1],
                                          EncodingResult_instance.encoding[idx:idx+1],
                                          EncodingResult_instance.mask[idx:idx+1],
                                          EncodingResult_instance.syntax_token_mask[idx:idx+1])
            elif isinstance(EncodingResult_instance, EncodingResult):
                # ['data', 'encoding', 'last_state', 'last_cell', 'mask']
                return EncodingResult(EncodingResult_instance.data[idx:idx+1],
                                      EncodingResult_instance.encoding[idx:idx+1],
                                      None, None,
                                      EncodingResult_instance.mask[idx:idx+1])
            else:
                raise Exception('Undetected EncodingResult type.')

        prev_data = [example.prev_data for example in examples]
        context_data = [example.context for example in examples]

        # prepare shared encodings
        prev_data_encodings = self.sequential_token_encoder(prev_data)
        context_encodings = self.context_encoder(context_data)
        if edit_encodings is not None:
            edit_encodings = _prepare_edit_encoding(self, edit_encodings)
        else:
            edit_encodings = self.get_edit_encoding(examples, prev_data_encodings=prev_data_encodings)

        # initial setup
        # init_code_ast_list = [example.prev_code_ast.copy_and_reindex_w_dummy_reduce() for example in examples]
        init_code_ast_list = [example.prev_code_ast for example in examples]
        init_input_encodings = self.encoder(init_code_ast_list, prev_data_encodings.encoding)
        hypotheses = []
        for e_idx, init_code_ast in enumerate(init_code_ast_list):
            hyp = Hypothesis(init_code_ast, bool_copy_subtree=self.decoder.copy_subtree,
                             memory=[subtree.root_node for subtree in examples[e_idx].tgt_actions[-1].meta['memory']])
            hyp.meta = {'idx': e_idx}
            hypotheses.append(hyp)

        # memory setup: initial_memory_encodings of (1, memory_size, hidden_size)
        batch_full_memory_encodings = init_input_encodings.encoding  # (batch_size, batch_max_node_num, hid)
        batch_memory_subtrees_size = [len(e.tgt_actions[-1].meta['memory']) for e in examples]
        max_memory_size = max(batch_memory_subtrees_size)
        batch_memory_encodings = []
        for e_idx, e in enumerate(examples):
            _valid_subtree_idx = [subtree.root_node.id for subtree in e.tgt_actions[-1].meta['memory']] + \
                                 [0] * (max_memory_size - batch_memory_subtrees_size[e_idx])
            batch_memory_encodings.append(batch_full_memory_encodings[e_idx][_valid_subtree_idx, :])
        batch_memory_encodings = torch.stack(batch_memory_encodings, dim=0)

        last_global_hidden_states, last_global_cell_states = self.init_global_states(examples)
        origin_idx2gold_edits = [[] for _ in range(len(examples))]
        origin_idx2actual_edits = [[] for _ in range(len(examples))]
        origin_idx2actual_decoded_tree = [None for _ in range(len(examples))]

        t = 0
        cur_partial_tgt_edits_list = [example.tgt_actions for example in examples]
        cur_prev_data_encodings = prev_data_encodings.encoding
        cur_edit_encodings = edit_encodings
        actual_max_trajectory_length = max(max_trajectory_length, max(len(example.tgt_actions) for example in examples))
        while t < actual_max_trajectory_length:
            batch_cur_inputs = [hyp.tree for hyp in hypotheses]
            cur_input_encodings = self.encoder(batch_cur_inputs, cur_prev_data_encodings)

            global_hidden_states, global_cell_states = self.controller(
                torch.cat([cur_edit_encodings, self.input_aggregation(cur_input_encodings.encoding,
                                                                      cur_input_encodings.mask)], dim=1),
                (last_global_hidden_states, last_global_cell_states))

            if t == 0:
                new_partial_tgt_edits_list = cur_partial_tgt_edits_list
            else:
                new_partial_tgt_edits_list = []
                for h_idx, cur_input in enumerate(batch_cur_inputs):
                    origin_e_idx = hypotheses[h_idx].meta['idx']
                    existing_non_gold_edit_seqs = None
                    if existing_non_gold_edit_seqs_list is not None:
                        existing_non_gold_edit_seqs = existing_non_gold_edit_seqs_list[origin_e_idx]
                    new_partial_tgt_edits = _refresh_tgt_edits(cur_input, cur_partial_tgt_edits_list[h_idx],
                                                               examples[origin_e_idx].updated_code_ast,
                                                               preset_memory=hypotheses[h_idx].memory,
                                                               existing_non_gold_edit_seqs=existing_non_gold_edit_seqs,
                                                               last_edit_field_node=hypotheses[h_idx].last_edit_field_node if prioritize_last_edit else None)
                    new_partial_tgt_edits_list.append(new_partial_tgt_edits)

            for h_idx, new_partial_tgt_edits in enumerate(new_partial_tgt_edits_list):
                origin_e_idx = hypotheses[h_idx].meta['idx']
                gold_edit = new_partial_tgt_edits[0]  # this should already include all meta data necessary for re-training.
                origin_idx2gold_edits[origin_e_idx].append(gold_edit)

            # sample the next action
            new_hypotheses = []
            new_hypotheses_h_idx = []
            for h_idx, hyp in enumerate(hypotheses):
                origin_e_idx = hypotheses[h_idx].meta['idx']

                p = random.random()
                if p < sampling_probability:  # sample from own policy
                    top_partial_hypotheses = self.decoder.one_step_beam_search_with_source_encodings(
                        [hyp], init_code_ast_list[origin_e_idx], context_data[origin_e_idx],
                        global_hidden_states[h_idx:(h_idx+1)],
                        _slice_EncodingResult(context_encodings, origin_e_idx),
                        _slice_EncodingResult(init_input_encodings, origin_e_idx),
                        batch_memory_encodings[origin_e_idx:(origin_e_idx+1)],
                        _slice_EncodingResult(cur_input_encodings, h_idx), self.substitution_system,
                        beam_size=1, time_step=t)
                    if top_partial_hypotheses:
                        new_hyp = top_partial_hypotheses[0]
                        origin_idx2actual_edits[origin_e_idx].append(new_hyp.edits[-1])
                        if not new_hyp.stopped:  # the last edit is Stop
                            new_hypotheses.append(new_hyp)
                            new_hypotheses_h_idx.append(h_idx)
                        if new_hyp.stopped or len(new_hyp.edits) >= actual_max_trajectory_length: # will stop
                            origin_idx2actual_decoded_tree[origin_e_idx] = new_hyp.tree.root_node
                            if extend_stop and not isinstance(origin_idx2gold_edits[origin_e_idx][-1], Stop):
                                origin_idx2gold_edits[origin_e_idx].extend(new_partial_tgt_edits_list[h_idx][1:])
                    else: # will stop
                        origin_idx2actual_decoded_tree[origin_e_idx] = hyp.tree.root_node
                        if extend_stop and not isinstance(origin_idx2gold_edits[origin_e_idx][-1], Stop):
                            origin_idx2gold_edits[origin_e_idx].extend(new_partial_tgt_edits_list[h_idx][1:])

                else:
                    gold_edit = origin_idx2gold_edits[origin_e_idx][-1]
                    new_partial_tgt_edits = new_partial_tgt_edits_list[h_idx]
                    origin_idx2actual_edits[origin_e_idx].append(gold_edit)

                    if len(new_partial_tgt_edits) > 1:
                        hyp.apply_edit(gold_edit)

                        new_hypotheses.append(hyp)
                        new_hypotheses_h_idx.append(h_idx)
                    else: # will stop
                        origin_idx2actual_decoded_tree[origin_e_idx] = hyp.tree.root_node

            if len(new_hypotheses) == 0:
                break # all hypotheses have ended

            last_global_hidden_states, last_global_cell_states = [], []
            cur_partial_tgt_edits_list = []
            cur_prev_data_encodings, cur_edit_encodings = [], []
            for valid_h_idx, valid_hyp in zip(new_hypotheses_h_idx, new_hypotheses):
                origin_e_idx = valid_hyp.meta['idx']

                if len(new_partial_tgt_edits_list[valid_h_idx]) > 1:
                    cur_partial_tgt_edits_list.append(new_partial_tgt_edits_list[valid_h_idx][1:])
                else:
                    cur_partial_tgt_edits_list.append(None)

                last_global_hidden_states.append(global_hidden_states[valid_h_idx])
                last_global_cell_states.append(global_cell_states[valid_h_idx])

                cur_prev_data_encodings.append(prev_data_encodings.encoding[origin_e_idx])
                cur_edit_encodings.append(edit_encodings[origin_e_idx])

            last_global_hidden_states = torch.stack(last_global_hidden_states)
            last_global_cell_states = torch.stack(last_global_cell_states)
            cur_prev_data_encodings = torch.stack(cur_prev_data_encodings)
            cur_edit_encodings = torch.stack(cur_edit_encodings)

            hypotheses = new_hypotheses

            t += 1

        # return training examples
        for origin_e_idx, gold_edits in enumerate(origin_idx2gold_edits):
            if 'memory' not in gold_edits[-1].meta:
                gold_edits[-1].meta['memory'] = examples[origin_e_idx].tgt_actions[-1].meta['memory']

        return origin_idx2gold_edits, origin_idx2actual_edits, origin_idx2actual_decoded_tree

    def decode_with_extend_correction_in_batch(self, examples, max_trajectory_length,
                                               prioritize_last_edit=True, id2decoded_tree_string=None):
        """ Return inference decoding sequence + remaining gold edits until the editing succeeds. """
        hypotheses = self.decode_updated_data_in_batch(examples, max_trajectory_length,
                                                       meta_tree=True, return_root_node=False)

        gold_edits_list, edits_weight_list, actual_decoded_trees = [], [], []
        for e_idx, hyp in enumerate(hypotheses):
            if hyp.tree.root_node == examples[e_idx].updated_code_ast.root_node:
                gold_edits_list.append(None)
                edits_weight_list.append(None)
                actual_decoded_trees.append(None)
                continue

            if id2decoded_tree_string is not None and examples[e_idx].id in id2decoded_tree_string and \
                    hyp.tree.root_node.to_string() in id2decoded_tree_string[examples[e_idx].id]:
                gold_edits_list.append(None)
                edits_weight_list.append(None)
                actual_decoded_trees.append(hyp.tree.root_node)
                continue

            tree_list = hyp.meta['tree']
            assert len(tree_list) >= len(hyp.edits)

            fake_gold_edits = []
            for edit_idx, edit in enumerate(hyp.edits):
                fake_gold_edit = Stop(meta={'tree': tree_list[edit_idx]})
                fake_gold_edits.append(fake_gold_edit)
            edits_weight = [0.] * len(fake_gold_edits)

            remaining_gold_edits = self.substitution_system.get_decoding_edits_fast(
                hyp.tree, examples[e_idx].updated_code_ast,
                bool_copy_subtree=self.args['decoder']['copy_subtree'],
                preset_memory=hyp.memory,
                last_edit_field_node=hyp.last_edit_field_node if prioritize_last_edit else None)
            edits_weight += [1.] * len(remaining_gold_edits)

            gold_edits_list.append(fake_gold_edits + remaining_gold_edits)
            edits_weight_list.append(edits_weight)
            actual_decoded_trees.append(hyp.tree.root_node)

        return gold_edits_list, edits_weight_list, actual_decoded_trees

    @staticmethod
    def build(args, vocab=None, grammar=None, transition_system=None):
        controller = nn.LSTMCell(input_size=args['encoder']['token_encoding_size'] +
                                            args['edit_encoder']['edit_encoding_size'],
                                 hidden_size=args['controller']['hidden_size'])

        embedder = SyntaxTreeEmbedder(args['embedder']['token_embed_size'],
                                      vocab,
                                      grammar,
                                      node_embed_method=args['embedder']['node_embed_method'])

        seq_token_encoder = SequentialEncoder(args['embedder']['token_embed_size'],
                                              args['encoder']['token_encoding_size'],
                                              token_embedder=embedder,
                                              vocab=vocab)

        graph_encoder = SyntaxTreeEncoder(hidden_size=args['encoder']['token_encoding_size'],
                                          syntax_tree_embedder=embedder,
                                          layer_timesteps=args['encoder']['layer_timesteps'],
                                          residual_connections=args['encoder']['residual_connections'],
                                          connections=args['encoder']['connections'],
                                          gnn_use_bias_for_message_linear=args['encoder'][
                                              'use_bias_for_message_linear'],
                                          dropout=args['encoder']['dropout'],
                                          vocab=vocab,
                                          grammar=grammar)

        edit_decoder = IterativeDecoder(global_state_hidden_size=args['controller']['hidden_size'],
                                        source_element_encoding_size=args['encoder']['token_encoding_size'],
                                        hidden_size=args['decoder']['hidden_size'],
                                        operator_embed_size=args['decoder']['operator_embed_size'],
                                        action_embed_size=args['decoder']['action_embed_size'],
                                        field_embed_size=args['decoder']['field_embed_size'],
                                        dropout=args['decoder']['dropout'],
                                        vocab=vocab,
                                        grammar=grammar,
                                        copy_syntax_token=args['decoder']['copy_token'],
                                        copy_sub_tree=args['decoder']['copy_subtree'],
                                        local_feed_anchor_node=args['decoder']['local_feed_anchor_node'],
                                        local_feed_siblings=args['decoder']['local_feed_siblings'],
                                        local_feed_parent_node=args['decoder']['local_feed_parent_node'])
        print(edit_decoder, file=sys.stderr)

        edit_encoder = EditEncoder.build(args, vocab=vocab, embedder=embedder,
                                         operators=edit_decoder.operators,
                                         operator_embedding=edit_decoder.operator_embedding,
                                         production_embedding=edit_decoder.production_embedding,
                                         field_embedding=edit_decoder.field_embedding,
                                         token_embedding=edit_decoder.token_embedding)

        substitution_system = SubstitutionSystem(transition_system)
        editor = Graph2IterEditEditor(controller, seq_token_encoder, graph_encoder, edit_decoder,
                                      context_encoder=seq_token_encoder,
                                      edit_encoder=edit_encoder,
                                      substitution_system=substitution_system,
                                      args=args)

        return editor

    def save(self, model_path):
        NeuralEditor.save(self, model_path, kwargs=dict(grammar=self.decoder.grammar))

    @staticmethod
    def get_gen_and_copy_index_and_mask(batch_init_code_asts, batch_contexts, batch_edits, context_encodings,
                                        init_input_encodings, memory_encodings, batch_max_node_num_in_cur_code,
                                        grammar, vocab, operators, copy_syntax_token, device):
        batch_size = len(batch_edits)

        operator_selection_idx = torch.zeros(batch_size, dtype=torch.long)

        node_selection_idx = torch.zeros(batch_size, dtype=torch.long)
        node_selection_mask = torch.zeros(batch_size, dtype=torch.long)
        node_cand_mask = torch.zeros(batch_size, batch_max_node_num_in_cur_code, dtype=torch.float)
        parent_field_idx = torch.zeros(batch_size, dtype=torch.long)

        tgt_apply_rule_idx = torch.zeros(batch_size, dtype=torch.long)
        tgt_apply_rule_mask = torch.zeros(batch_size, dtype=torch.long)
        apply_rule_cand_mask = torch.zeros(batch_size, len(grammar) + 1, dtype=torch.float)

        # get the maximum number of candidate subtree nodes to copy in an action
        max_cand_subtree_node_num = max([len(edit.meta['tree_node_ids_to_copy']) for edit in batch_edits
                                         if isinstance(edit, AddSubtree)] + [1])
        tgt_apply_subtree_idx = torch.zeros(batch_size, max_cand_subtree_node_num, dtype=torch.long)
        tgt_apply_subtree_idx_mask = torch.zeros(batch_size, max_cand_subtree_node_num, dtype=torch.float)
        tgt_apply_subtree_mask = torch.zeros(batch_size, dtype=torch.long)
        apply_subtree_cand_mask = torch.zeros(batch_size, memory_encodings.size(1) if memory_encodings is not None else 1,
                                              dtype=torch.float)

        tgt_gen_token_idx = torch.zeros(batch_size, dtype=torch.long)
        tgt_gen_token_mask = torch.zeros(batch_size, dtype=torch.float)

        tgt_copy_ctx_token_idx_mask = torch.zeros(batch_size, context_encodings.encoding.size(1), dtype=torch.float)
        tgt_copy_ctx_token_mask = torch.zeros(batch_size, dtype=torch.float)

        tgt_copy_init_token_idx_mask = torch.zeros(batch_size, init_input_encodings.encoding.size(1), dtype=torch.float)
        tgt_copy_init_token_mask = torch.zeros(batch_size, dtype=torch.float)

        for example_id, edit in enumerate(batch_edits):
            if isinstance(edit, Delete):
                operator_selection_idx[example_id] = operators.index("delete")

                node_selection_idx[example_id] = edit.node.id
                node_selection_mask[example_id] = 1
                valid_cont_node_ids = edit.meta['valid_cont_node_ids']
                node_cand_mask[example_id, valid_cont_node_ids] = 1.

                parent_node = edit.field.parent_node
                parent_field = edit.field
                parent_field_idx[example_id] = grammar.prod_field2id[
                    (parent_node.production, parent_field.field)]

            elif isinstance(edit, Add):
                operator_selection_idx[example_id] = operators.index("add")

                node_selection_idx[example_id] = edit.field.as_value_list[edit.value_idx].id
                node_selection_mask[example_id] = 1
                valid_cont_node_ids = edit.meta['valid_cont_node_ids']
                node_cand_mask[example_id, valid_cont_node_ids] = 1.

                parent_node = edit.field.parent_node
                parent_field = edit.field
                parent_field_idx[example_id] = grammar.prod_field2id[
                    (parent_node.production, parent_field.field)]

                action = edit.action
                if isinstance(action, ApplyRuleAction):
                    app_rule_idx = grammar.prod2id[action.production]
                    tgt_apply_rule_idx[example_id] = app_rule_idx
                    tgt_apply_rule_mask[example_id] = 1

                    valid_cont_prod_ids = edit.meta['valid_cont_prod_ids']
                    apply_rule_cand_mask[example_id, valid_cont_prod_ids] = 1
                else:
                    assert isinstance(action, GenTokenAction)
                    tgt_token = action.token # SyntaxToken type
                    if SequentialDecoder._can_only_generate_this_token(tgt_token.value):
                        tgt_gen_token_mask[example_id] = 1
                        tgt_gen_token_idx[example_id] = vocab[tgt_token.value]
                    else:
                        # init_code_ast = batch_examples[example_id].tgt_actions[0].meta['tree']
                        # context = batch_examples[example_id].context
                        init_code_ast = batch_init_code_asts[example_id]
                        context = batch_contexts[example_id]
                        copied = False
                        if copy_syntax_token:
                            if tgt_token in init_code_ast.syntax_tokens_set:
                                token_pos_list = [node_id for node_id, syntax_token in init_code_ast.syntax_tokens_and_ids if
                                                  syntax_token == tgt_token]
                                tgt_copy_init_token_mask[example_id] = 1
                                tgt_copy_init_token_idx_mask[example_id, token_pos_list] = 1
                                copied = True
                            if tgt_token.value in context:
                                token_pos_list = [pos for pos, token in enumerate(context) if token == tgt_token.value]
                                tgt_copy_ctx_token_idx_mask[example_id, token_pos_list] = 1
                                tgt_copy_ctx_token_mask[example_id] = 1
                                copied = True

                        tgt_gen_token_idx[example_id] = vocab[tgt_token.value] # always get a vocab embedding
                        if not copied or tgt_token.value in vocab:
                            # if the token is not copied, we can only generate this token from the vocabulary,
                            # even if it is a <unk>.
                            # otherwise, we can still generate it from the vocabulary
                            tgt_gen_token_mask[example_id] = 1
                            # tgt_gen_token_idx[example_id] = vocab[tgt_token.value]

            elif isinstance(edit, AddSubtree):
                operator_selection_idx[example_id] = operators.index("add_subtree")

                node_selection_idx[example_id] = edit.field.as_value_list[edit.value_idx].id
                node_selection_mask[example_id] = 1
                valid_cont_node_ids = edit.meta['valid_cont_node_ids']
                node_cand_mask[example_id, valid_cont_node_ids] = 1.

                parent_node = edit.field.parent_node
                parent_field = edit.field
                parent_field_idx[example_id] = grammar.prod_field2id[
                    (parent_node.production, parent_field.field)]

                tree_node_ids_to_copy = edit.meta['tree_node_ids_to_copy']

                tgt_apply_subtree_idx[example_id, :len(tree_node_ids_to_copy)] = torch.tensor(
                    tree_node_ids_to_copy, dtype=torch.long, device=device)
                tgt_apply_subtree_idx_mask[example_id, :len(tree_node_ids_to_copy)] = 1
                tgt_apply_subtree_mask[example_id] = 1

                valid_cont_subtree_ids = edit.meta['valid_cont_subtree_ids']
                apply_subtree_cand_mask[example_id, valid_cont_subtree_ids] = 1.

            else:
                assert isinstance(edit, Stop)
                operator_selection_idx[example_id] = operators.index("stop")

        operator_selection_idx = operator_selection_idx.to(device)
        node_selection_idx = node_selection_idx.to(device)
        node_selection_mask = node_selection_mask.to(device)
        node_cand_mask = node_cand_mask.to(device)
        parent_field_idx = parent_field_idx.to(device)
        tgt_apply_rule_idx = tgt_apply_rule_idx.to(device)
        tgt_apply_rule_mask = tgt_apply_rule_mask.to(device)
        apply_rule_cand_mask = apply_rule_cand_mask.to(device)
        tgt_apply_subtree_idx = tgt_apply_subtree_idx.to(device)
        tgt_apply_subtree_idx_mask = tgt_apply_subtree_idx_mask.to(device)
        tgt_apply_subtree_mask = tgt_apply_subtree_mask.to(device)
        apply_subtree_cand_mask = apply_subtree_cand_mask.to(device)
        tgt_gen_token_idx = tgt_gen_token_idx.to(device)
        tgt_gen_token_mask = tgt_gen_token_mask.to(device)
        tgt_copy_ctx_token_idx_mask = tgt_copy_ctx_token_idx_mask.to(device)
        tgt_copy_ctx_token_mask = tgt_copy_ctx_token_mask.to(device)
        tgt_copy_init_token_idx_mask = tgt_copy_init_token_idx_mask.to(device)
        tgt_copy_init_token_mask = tgt_copy_init_token_mask.to(device)

        return operator_selection_idx, \
               node_selection_idx, node_selection_mask, node_cand_mask, parent_field_idx, \
               tgt_apply_rule_idx, tgt_apply_rule_mask, apply_rule_cand_mask, \
               tgt_apply_subtree_idx, tgt_apply_subtree_idx_mask, tgt_apply_subtree_mask, apply_subtree_cand_mask, \
               tgt_gen_token_idx, tgt_gen_token_mask, tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
               tgt_copy_init_token_idx_mask, tgt_copy_init_token_mask

    @staticmethod
    def get_gen_and_copy_index_and_mask_over_time(batch_init_code_asts, batch_contexts, batch_edits_list,
                                                  context_encodings, init_input_encodings, memory_encodings,
                                                  batch_max_node_num_over_time,
                                                  grammar, vocab, operators, copy_syntax_token, device):
        max_iteration_step = len(batch_edits_list)
        batch_size = len(batch_init_code_asts)

        operator_selection_idx_over_time = []

        node_selection_idx_over_time, node_selection_mask_over_time, node_cand_mask_over_time = [], [], []

        parent_field_idx_over_time = []

        tgt_apply_rule_idx_over_time, tgt_apply_rule_mask_over_time, apply_rule_cand_mask_over_time = [], [], []

        # get the maximum number of candidate subtree nodes to copy in an action
        max_cand_subtree_node_num_over_time = max([len(edit.meta['tree_node_ids_to_copy'])
                                                   for batch_edits in batch_edits_list
                                                   for edit in batch_edits if isinstance(edit, AddSubtree)] + [1])
        tgt_apply_subtree_idx_over_time = torch.zeros(max_iteration_step, batch_size, max_cand_subtree_node_num_over_time,
                                                      dtype=torch.long).to(device)
        tgt_apply_subtree_idx_mask_over_time = torch.zeros(max_iteration_step, batch_size, max_cand_subtree_node_num_over_time,
                                                           dtype=torch.float).to(device)
        tgt_apply_subtree_mask_over_time, apply_subtree_cand_mask_over_time = [], []

        tgt_gen_token_idx_over_time, tgt_gen_token_mask_over_time = [], []
        tgt_copy_ctx_token_idx_mask_over_time, tgt_copy_ctx_token_mask_over_time = [], []
        tgt_copy_init_token_idx_mask_over_time, tgt_copy_init_token_mask_over_time = [], []

        for t in range(max_iteration_step):
            operator_selection_idx, \
            node_selection_idx, node_selection_mask, node_cand_mask, parent_field_idx, \
            tgt_apply_rule_idx, tgt_apply_rule_mask, apply_rule_cand_mask, \
            tgt_apply_subtree_idx, tgt_apply_subtree_idx_mask, tgt_apply_subtree_mask, apply_subtree_cand_mask, \
            tgt_gen_token_idx, tgt_gen_token_mask, tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
            tgt_copy_init_token_idx_mask, tgt_copy_init_token_mask = Graph2IterEditEditor.get_gen_and_copy_index_and_mask(
                batch_init_code_asts, batch_contexts, batch_edits_list[t], context_encodings, init_input_encodings,
                memory_encodings, batch_max_node_num_over_time, grammar, vocab, operators, copy_syntax_token, device)

            operator_selection_idx_over_time.append(operator_selection_idx)
            node_selection_idx_over_time.append(node_selection_idx)
            node_selection_mask_over_time.append(node_selection_mask)
            node_cand_mask_over_time.append(node_cand_mask)
            parent_field_idx_over_time.append(parent_field_idx)
            tgt_apply_rule_idx_over_time.append(tgt_apply_rule_idx)
            tgt_apply_rule_mask_over_time.append(tgt_apply_rule_mask)
            apply_rule_cand_mask_over_time.append(apply_rule_cand_mask)
            tgt_apply_subtree_idx_over_time[t, :, :tgt_apply_subtree_idx.size(1)] = tgt_apply_subtree_idx
            tgt_apply_subtree_idx_mask_over_time[t, :, :tgt_apply_subtree_idx_mask.size(1)] = tgt_apply_subtree_idx_mask
            tgt_apply_subtree_mask_over_time.append(tgt_apply_subtree_mask)
            apply_subtree_cand_mask_over_time.append(apply_subtree_cand_mask)
            tgt_gen_token_idx_over_time.append(tgt_gen_token_idx)
            tgt_gen_token_mask_over_time.append(tgt_gen_token_mask)
            tgt_copy_ctx_token_idx_mask_over_time.append(tgt_copy_ctx_token_idx_mask)
            tgt_copy_ctx_token_mask_over_time.append(tgt_copy_ctx_token_mask)
            tgt_copy_init_token_idx_mask_over_time.append(tgt_copy_init_token_idx_mask)
            tgt_copy_init_token_mask_over_time.append(tgt_copy_init_token_mask)

        return torch.stack(operator_selection_idx_over_time, dim=0), torch.stack(node_selection_idx_over_time, dim=0), \
               torch.stack(node_selection_mask_over_time, dim=0), torch.stack(node_cand_mask_over_time, dim=0), \
               torch.stack(parent_field_idx_over_time, dim=0), \
               torch.stack(tgt_apply_rule_idx_over_time, dim=0), torch.stack(tgt_apply_rule_mask_over_time, dim=0), \
               torch.stack(apply_rule_cand_mask_over_time, dim=0), tgt_apply_subtree_idx_over_time, \
               tgt_apply_subtree_idx_mask_over_time, torch.stack(tgt_apply_subtree_mask_over_time, dim=0), \
               torch.stack(apply_subtree_cand_mask_over_time, dim=0), torch.stack(tgt_gen_token_idx_over_time, dim=0), \
               torch.stack(tgt_gen_token_mask_over_time, dim=0), torch.stack(tgt_copy_ctx_token_idx_mask_over_time, dim=0), \
               torch.stack(tgt_copy_ctx_token_mask_over_time, dim=0), torch.stack(tgt_copy_init_token_idx_mask_over_time, dim=0), \
               torch.stack(tgt_copy_init_token_mask_over_time, dim=0)


class SequentialAutoEncoder(nn.Module):
    def __init__(self,
                 token_embed_size, token_encoding_size, change_vector_size, change_tag_embed_size,
                 decoder_hidden_size, decoder_dropout, init_decode_vec_encoder_state_dropout,
                 vocab,
                 no_change_vector=False,
                 no_unchanged_token_encoding_in_diff_seq=False,
                 no_copy=False,
                 change_encoder_type='word',
                 token_embedder='word'):

        self.args = utils.get_method_args_dict(self.__init__, locals())
        super(SequentialAutoEncoder, self).__init__()

        if token_embedder == 'word':
            self.syntax_token_embedder = CodeTokenEmbedder(token_embed_size, vocab)
        elif token_embedder == 'char':
            self.syntax_token_embedder = ConvolutionalCharacterEmbedder(token_embed_size, max_character_size=20)

        self.sequential_code_encoder = SequentialCodeEncoder(token_embed_size, token_encoding_size,
                                                             code_token_embedder=self.syntax_token_embedder,
                                                             vocab=vocab)

        if change_encoder_type == 'word':
            self.code_change_encoder = SequentialChangeEncoder(token_encoding_size, change_vector_size,
                                                               change_tag_embed_size,
                                                               vocab,
                                                               no_unchanged_token_encoding_in_diff_seq=no_unchanged_token_encoding_in_diff_seq)
        elif change_encoder_type == 'bag':
            self.code_change_encoder = BagOfEditsChangeEncoder(self.syntax_token_embedder.weight,
                                                               vocab)

        self.decoder = SequentialDecoder(token_embed_size, token_encoding_size, change_vector_size, decoder_hidden_size,
                                         dropout=decoder_dropout,
                                         init_decode_vec_encoder_state_dropout=init_decode_vec_encoder_state_dropout,
                                         code_token_embedder=self.syntax_token_embedder,
                                         vocab=vocab,
                                         no_copy=no_copy)

        self.vocab = vocab

    @property
    def device(self):
        return self.code_change_encoder.device

    def forward(self, examples, return_change_vectors=False):
        previous_code_chunk_list = [e.previous_code_chunk for e in examples]
        updated_code_chunk_list = [e.updated_code_chunk for e in examples]
        context_list = [e.context for e in examples]

        embedding_cache = EmbeddingTable(
            chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
        self.syntax_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = self.sequential_code_encoder.encode(previous_code_chunk_list,
                                                                embedding_cache=embedding_cache)
        batched_updated_code = self.sequential_code_encoder.encode(updated_code_chunk_list,
                                                                   embedding_cache=embedding_cache)
        batched_context = self.sequential_code_encoder.encode(context_list, embedding_cache=embedding_cache)

        if self.args['no_change_vector'] is False:
            change_vectors = self.code_change_encoder(examples, batched_prev_code, batched_updated_code)
        else:
            change_vectors = torch.zeros(batched_updated_code.batch_size, self.args['change_vector_size'], device=self.device)

        scores = self.decoder(examples, batched_prev_code, batched_context, change_vectors, embedding_cache=embedding_cache)

        if return_change_vectors:
            return scores, change_vectors
        else:
            return scores

    def decode_updated_code(self, example, with_change_vec=False, change_vec=None, beam_size=5, debug=False):
        previous_code_chunk_list = [example.previous_code_chunk]
        updated_code_chunk_list = [example.updated_code_chunk]
        context_list = [example.context]

        embedding_cache = EmbeddingTable(
            chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
        self.syntax_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = self.sequential_code_encoder.encode(previous_code_chunk_list,
                                                                embedding_cache=embedding_cache)
        batched_updated_code = self.sequential_code_encoder.encode(updated_code_chunk_list,
                                                                   embedding_cache=embedding_cache)
        batched_context = self.sequential_code_encoder.encode(context_list, embedding_cache=embedding_cache)

        if change_vec is not None:
            change_vectors = torch.from_numpy(change_vec).to(self.device)
            if len(change_vectors.size()) == 1:
                change_vectors = change_vectors.unsqueeze(0)
        elif with_change_vec:
            change_vectors = self.code_change_encoder([example], batched_prev_code, batched_updated_code)
        else:
            change_vectors = torch.zeros(batched_updated_code.batch_size, self.args['change_vector_size'],
                                         device=self.device)

        hypotheses = self.decoder.beam_search_with_source_encodings(example.previous_code_chunk, batched_prev_code,
                                                                    example.context, batched_context,
                                                                    change_vectors,
                                                                    beam_size=beam_size, max_decoding_time_step=70,
                                                                    debug=debug)

        return hypotheses

    def save(self, model_path):
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, model_path)

    @staticmethod
    def load(model_path, use_cuda=True):
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = SequentialAutoEncoder(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)

        return model


class TreeBasedAutoEncoderWithGraphEncoder(nn.Module):
    def __init__(self,
                 token_embed_size, token_encoding_size, change_vector_size, change_tag_embed_size,
                 action_embed_size, field_embed_size,
                 decoder_hidden_size, decoder_dropout, init_decode_vec_encoder_state_dropout,
                 gnn_layer_timesteps, gnn_residual_connections, gnn_dropout,
                 vocab,
                 grammar,
                 mode,
                 no_change_vector=False,
                 no_unchanged_token_encoding_in_diff_seq=False,
                 use_syntax_token_rnn=False,
                 change_encoder_type='word',
                 token_embedder='word',
                 node_embed_method='type',
                 no_penalize_apply_tree_when_copy_subtree=False,
                 encode_change_vec_in_syntax_token_rnn=False,
                 feed_in_token_rnn_state_to_rule_rnn=False,
                 fuse_rule_and_token_rnns=False,
                 gnn_no_token_connection=False,
                 gnn_no_top_down_connection=False,
                 gnn_no_bottom_up_connection=False,
                 gnn_prev_sibling_connection=False,
                 gnn_next_sibling_connection=False,
                 copy_identifier=True,
                 decoder_init_method='avg_pooling',
                 gnn_use_bias_for_message_linear=True,
                 change_encoder_master_node_option=None,
                 no_copy=False):

        self.args = utils.get_method_args_dict(self.__init__, locals())
        super(TreeBasedAutoEncoderWithGraphEncoder, self).__init__()

        self.syntax_tree_node_embedder = SyntaxTreeEmbedder(token_embed_size, vocab, grammar, node_embed_method=node_embed_method)

        if token_embedder == 'word':
            self.syntax_token_embedder = self.syntax_tree_node_embedder
        elif token_embedder == 'char':
            self.syntax_token_embedder = ConvolutionalCharacterEmbedder(token_embed_size, max_character_size=20)

        self.sequential_code_encoder = SequentialCodeEncoder(token_embed_size, token_encoding_size,
                                                             code_token_embedder=self.syntax_token_embedder,
                                                             vocab=vocab)

        if change_encoder_type == 'word':
            self.code_change_encoder = SequentialChangeEncoder(token_encoding_size, change_vector_size, change_tag_embed_size,
                                                               vocab,
                                                               no_unchanged_token_encoding_in_diff_seq=no_unchanged_token_encoding_in_diff_seq)
        elif change_encoder_type == 'graph':
            self.code_change_encoder = GraphChangeEncoder(change_vector_size, syntax_tree_embedder=self.syntax_tree_node_embedder,
                                                          layer_time_steps=gnn_layer_timesteps,
                                                          dropout=gnn_dropout,
                                                          gnn_use_bias_for_message_linear=gnn_use_bias_for_message_linear,
                                                          master_node_option=change_encoder_master_node_option)
        elif change_encoder_type == 'hybrid':
            self.code_change_encoder = HybridChangeEncoder(token_encoding_size=token_encoding_size,
                                                           change_vector_dim=change_vector_size,
                                                           syntax_tree_embedder=self.syntax_tree_node_embedder,
                                                           layer_timesteps=gnn_layer_timesteps,
                                                           dropout=gnn_dropout,
                                                           vocab=vocab,
                                                           gnn_use_bias_for_message_linear=gnn_use_bias_for_message_linear)
        elif change_encoder_type == 'bag':
            self.code_change_encoder = BagOfEditsChangeEncoder(self.syntax_token_embedder.weight,
                                                               vocab)

        else:
            raise ValueError('unknown code change encoder type %s' % change_encoder_type)

        self.prev_ast_encoder = GraphCodeEncoder(hidden_size=token_encoding_size,
                                                 syntax_tree_embedder=self.syntax_tree_node_embedder,
                                                 layer_timesteps=gnn_layer_timesteps, residual_connections=gnn_residual_connections, dropout=gnn_dropout,
                                                 vocab=vocab, grammar=grammar,
                                                 token_bidirectional_connection=not gnn_no_token_connection,
                                                 top_down_connection=not gnn_no_top_down_connection,
                                                 bottom_up_connection=not gnn_no_bottom_up_connection,
                                                 prev_sibling_connection=gnn_prev_sibling_connection,
                                                 next_sibling_connection=gnn_next_sibling_connection,
                                                 gnn_use_bias_for_message_linear=gnn_use_bias_for_message_linear)

        if '2tree' in mode:
            self.decoder = TransitionDecoderWithGraphEncoder(node_encoding_size=token_encoding_size,
                                                             change_vector_size=change_vector_size,
                                                             hidden_size=decoder_hidden_size,
                                                             action_embed_size=action_embed_size,
                                                             field_embed_size=field_embed_size,
                                                             dropout=decoder_dropout,
                                                             init_decode_vec_encoder_state_dropout=init_decode_vec_encoder_state_dropout,
                                                             vocab=vocab, grammar=grammar, mode=mode,
                                                             syntax_tree_embedder=self.syntax_tree_node_embedder,
                                                             use_syntax_token_rnn=use_syntax_token_rnn,
                                                             no_penalize_apply_tree_when_copy_subtree=no_penalize_apply_tree_when_copy_subtree,
                                                             encode_change_vec_in_syntax_token_rnn=encode_change_vec_in_syntax_token_rnn,
                                                             feed_in_token_rnn_state_to_rule_rnn=feed_in_token_rnn_state_to_rule_rnn,
                                                             fuse_rule_and_token_rnns=fuse_rule_and_token_rnns,
                                                             decoder_init_method=decoder_init_method,
                                                             copy_identifier=copy_identifier,
                                                             no_copy=no_copy)
        else:
            self.decoder = SequentialDecoderWithTreeEncoder(token_embed_size, token_encoding_size, change_vector_size,
                                                            decoder_hidden_size,
                                                            dropout=decoder_dropout,
                                                            init_decode_vec_encoder_state_dropout=init_decode_vec_encoder_state_dropout,
                                                            code_token_embedder=self.syntax_token_embedder,
                                                            vocab=vocab,
                                                            decoder_init_method=decoder_init_method)

        self.vocab = vocab
        self.grammar = grammar

    @property
    def device(self):
        return self.code_change_encoder.device

    def forward(self, examples, return_change_vectors=False, **kwargs):
        previous_code_chunk_list = [e.previous_code_chunk for e in examples]
        updated_code_chunk_list = [e.updated_code_chunk for e in examples]
        context_list = [e.context for e in examples]

        embedding_cache = EmbeddingTable(chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
        self.syntax_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = self.sequential_code_encoder.encode(previous_code_chunk_list, embedding_cache=embedding_cache)
        batched_updated_code = self.sequential_code_encoder.encode(updated_code_chunk_list, embedding_cache=embedding_cache)
        batched_context = self.sequential_code_encoder.encode(context_list, embedding_cache=embedding_cache)

        if self.args['no_change_vector'] is False:
            change_vectors = self.code_change_encoder(examples, batched_prev_code, batched_updated_code)
        else:
            change_vectors = torch.zeros(batched_updated_code.batch_size, self.args['change_vector_size'],
                                         device=self.device)

        batched_prev_ast_node_encoding, \
        batched_prev_ast_node_mask, \
        batched_prev_ast_syntax_token_mask = self.prev_ast_encoder([e.prev_code_ast for e in examples], batched_prev_code.encoding)

        batched_prev_asts = type('BatchedDatum', (object,), {'encoding': batched_prev_ast_node_encoding,
                                                             'mask': batched_prev_ast_node_mask,
                                                             'syntax_token_mask': batched_prev_ast_syntax_token_mask})

        results = self.decoder(examples, batched_prev_asts, batched_context, change_vectors, embedding_cache=embedding_cache, **kwargs)

        if return_change_vectors:
            return results, change_vectors
        else:
            return results

    def decode_updated_code(self, example, transition_system, with_change_vec=False, change_vec=None, beam_size=5, debug=False):
        previous_code_chunk_list = [example.previous_code_chunk]
        updated_code_chunk_list = [example.updated_code_chunk]
        context_list = [example.context]

        embedding_cache = EmbeddingTable(
            chain.from_iterable(previous_code_chunk_list + updated_code_chunk_list + context_list))
        self.syntax_token_embedder.populate_embedding_table(embedding_cache)

        batched_prev_code = self.sequential_code_encoder.encode(previous_code_chunk_list,
                                                                embedding_cache=embedding_cache)
        batched_updated_code = self.sequential_code_encoder.encode(updated_code_chunk_list,
                                                                   embedding_cache=embedding_cache)
        batched_context = self.sequential_code_encoder.encode(context_list, embedding_cache=embedding_cache)

        if change_vec is not None:
            change_vectors = torch.from_numpy(change_vec).to(self.device)
            if len(change_vectors.size()) == 1:
                change_vectors = change_vectors.unsqueeze(0)
        elif with_change_vec:
            change_vectors = self.code_change_encoder([example], batched_prev_code, batched_updated_code)
        else:
            change_vectors = torch.zeros(batched_updated_code.batch_size, self.args['change_vector_size'],
                                         device=self.device)

        batched_prev_ast_node_encoding, \
        batched_prev_ast_node_mask, \
        batched_prev_ast_syntax_token_mask = self.prev_ast_encoder([example.prev_code_ast],
                                                                   batched_prev_code.encoding)

        batched_prev_asts = type('BatchedDatum', (object,), {'encoding': batched_prev_ast_node_encoding,
                                                             'mask': batched_prev_ast_node_mask,
                                                             'syntax_token_mask': batched_prev_ast_syntax_token_mask})

        hypotheses = self.decoder.beam_search_with_source_encodings(example.prev_code_ast, batched_prev_asts,
                                                                    example.context, batched_context,
                                                                    change_vectors,
                                                                    beam_size=beam_size, max_decoding_time_step=70,
                                                                    transition_system=transition_system, debug=debug)

        return hypotheses

    def save(self, model_path):
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'grammar': self.grammar,
            'state_dict': self.state_dict()
        }

        torch.save(params, model_path)

    @staticmethod
    def load(model_path, use_cuda=True):
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = TreeBasedAutoEncoderWithGraphEncoder(vocab=params['vocab'], grammar=params['grammar'], **args)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)

        return model


class WordPredictionMultiTask(nn.Module):
    def __init__(self, change_vector_size, vocab, device):
        super(WordPredictionMultiTask, self).__init__()

        self.vocab = vocab
        self.device = device
        self.change_vec_to_vocab = nn.Linear(change_vector_size, len(vocab))
        self.words_to_discard = {'VAR0', 'int', 'long', 'string', 'float', 'LITERAL', 'var'}

    def forward(self, examples, change_vecs):
        # change_vecs: (batch_size, change_vec_size)

        # (batch_size, max_word_num)
        tgt_word_ids, tgt_word_mask = self.get_word_ids_to_predict(examples)

        # (batch_size, vocab_size)
        log_probs = F.log_softmax(self.change_vec_to_vocab(change_vecs), dim=-1)

        tgt_log_probs = torch.gather(log_probs, 1, tgt_word_ids)
        tgt_log_probs = (tgt_log_probs * tgt_word_mask).sum(dim=-1)
        tgt_log_probs = tgt_log_probs / (tgt_word_mask.sum(dim=-1) + 1e-7)  # to avoid underflow

        return tgt_log_probs

    def get_word_ids_to_predict(self, examples):
        tgt_words = []
        for example in examples:
            example_tgt_words = []

            example_tgt_words.extend(filter(lambda x: x not in self.words_to_discard and not all(c in string.punctuation for c in x), example.previous_code_chunk))
            example_tgt_words.extend(filter(lambda x: x not in self.words_to_discard and not all(c in string.punctuation for c in x), example.updated_code_chunk))

            tgt_words.append(example_tgt_words)
            # if len(example_tgt_words) == 0:
            #     print(example.prev_data)
            #     print(example.updated_data)

        max_word_num = max(len(x) for x in tgt_words)
        tgt_word_ids = torch.zeros(len(examples), max_word_num, dtype=torch.long, device=self.device)
        tgt_word_mask = torch.zeros(len(examples), max_word_num, dtype=torch.float, device=self.device)

        for batch_id, example_words in enumerate(tgt_words):
            tgt_word_ids[batch_id, :len(example_words)] = torch.LongTensor([self.vocab[word] for word in example_words], device=self.device)
            tgt_word_mask[batch_id, :len(example_words)] = 1

        return tgt_word_ids, tgt_word_mask


class ChangedWordPredictionMultiTask(nn.Module):
    def __init__(self, change_vector_size, vocab, device):
        super(ChangedWordPredictionMultiTask, self).__init__()

        self.vocab = vocab
        self.device = device
        self.change_vec_to_vocab = nn.Linear(change_vector_size, len(vocab) * 2)
        self.offset = len(vocab)
        self.words_to_discard = {'VAR', 'LITERAL', 'var'}  # 'int', 'long', 'string', 'float',

    def forward(self, examples, change_vecs):
        # change_vecs: (batch_size, change_vec_size)

        # (batch_size, max_word_num)
        tgt_word_ids, tgt_word_mask = self.get_word_ids_to_predict(examples)

        if len(tgt_word_ids.size()) == 1:
            return None

        # (batch_size, vocab_size)
        log_probs = F.log_softmax(self.change_vec_to_vocab(change_vecs), dim=-1)

        tgt_log_probs = torch.gather(log_probs, 1, tgt_word_ids)
        tgt_log_probs = (tgt_log_probs * tgt_word_mask).sum(dim=-1)
        tgt_log_probs = tgt_log_probs / (tgt_word_mask.sum(dim=-1) + 1e-7)  # to avoid underflow

        return tgt_log_probs

    def get_changed_words_from_change_seq(self, change_seq):
        add_del_words = []
        for entry in change_seq:
            tag, token = entry

            if tag == 'ADD':
                add_del_words.append(('ADD', token))
            elif tag == 'DEL':
                add_del_words.append(('DEL', token))
            elif tag == 'REPLACE':
                add_del_words.append(('DEL', token[0]))
                add_del_words.append(('ADD', token[1]))

        add_del_words = list(filter(lambda t: t[1] not in self.words_to_discard and \
                                                   not t[1].startswith('VAR') and \
                                                   not all(c in string.punctuation for c in t[1]), add_del_words))

        return add_del_words

    def get_word_ids_to_predict(self, examples):
        tgt_words = []
        for example in examples:
            example_tgt_words = self.get_changed_words_from_change_seq(example.change_seq)
            tgt_words.append(example_tgt_words)

        max_word_num = max(len(x) for x in tgt_words)
        tgt_word_ids = torch.zeros(len(examples), max_word_num, dtype=torch.long, device=self.device)
        tgt_word_mask = torch.zeros(len(examples), max_word_num, dtype=torch.float, device=self.device)

        for batch_id, example_words in enumerate(tgt_words):
            if len(example_words) > 0:
                tgt_word_ids[batch_id, :len(example_words)] = torch.LongTensor([self.vocab[word] if tag == 'ADD' else (self.offset + self.vocab[word])
                                                                                for tag, word in example_words],
                                                                            device=self.device)
                tgt_word_mask[batch_id, :len(example_words)] = 1

        return tgt_word_ids, tgt_word_mask