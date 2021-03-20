# coding=utf-8
from collections import OrderedDict
from itertools import chain
import re

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from asdl.transition_system import ApplyRuleAction, ReduceAction, ApplySubTreeAction, GenTokenAction
from asdl.hypothesis import Hypothesis
from asdl.lang.csharp.csharp_hypothesis import CSharpHypothesis
from asdl.asdl_ast import SyntaxToken

from edit_model import nn_utils
from edit_model.encdec.sequential_decoder import SequentialDecoder, Decoder
from edit_model.pointer_net import PointerNet

import numpy as np


class TransitionDecoder(Decoder):
    """tree-based decoder that constructs ASTs using a sequence of transition actions"""

    def __init__(self, source_element_encoding_size, change_vector_size, hidden_size,
                 action_embed_size, field_embed_size,
                 dropout,
                 init_decode_vec_encoder_state_dropout,
                 syntax_tree_encoder,
                 vocab,
                 grammar,
                 **kwargs):
        super(TransitionDecoder, self).__init__()

        self.source_element_encoding_size = source_element_encoding_size
        self.hidden_size = hidden_size
        self.action_embed_size = action_embed_size
        self.field_embed_size = field_embed_size
        self.change_vector_size = change_vector_size

        self.vocab = vocab
        self.grammar = grammar
        if self.grammar.language == 'csharp':
            self.lambda_hypothesis = CSharpHypothesis
        else:
            self.lambda_hypothesis = Hypothesis

        self.no_penalize_apply_tree_when_copy_subtree = kwargs.pop('no_penalize_apply_tree_when_copy_subtree', False)
        self.use_syntax_token_rnn = kwargs.pop('use_syntax_token_rnn', False)
        self.encode_change_vec_in_syntax_token_rnn = kwargs.pop('encode_change_vec_in_syntax_token_rnn', False)
        self.feed_in_token_rnn_state_to_rule_rnn = kwargs.pop('feed_in_token_rnn_state_to_rule_rnn', False)
        self.fuse_rule_and_token_rnns = kwargs.pop('fuse_rule_and_token_rnns', False)
        self.copy_syntax_token = kwargs.pop('copy_syntax_token', True)
        self.copy_subtree = kwargs.pop('copy_subtree', True)
        self.copy_identifier_node = kwargs.pop('copy_identifier_node', True)
        self.decoder_init_method = kwargs.pop('decoder_init_method', 'avg_pooling')

        decoder_input_dim = action_embed_size  # previous action embedding
        decoder_input_dim += hidden_size  # parent hidden state
        decoder_input_dim += field_embed_size
        decoder_input_dim += change_vector_size

        if self.use_syntax_token_rnn:
            if self.feed_in_token_rnn_state_to_rule_rnn:
                decoder_input_dim += self.hidden_size

            syntax_token_rnn_input_dim = source_element_encoding_size
            if self.encode_change_vec_in_syntax_token_rnn:
                assert self.use_syntax_token_rnn
                syntax_token_rnn_input_dim += change_vector_size

            self.syntax_token_rnn = nn.LSTM(syntax_token_rnn_input_dim, self.hidden_size, bidirectional=False)

            if self.fuse_rule_and_token_rnns:
                self.fuse_rule_and_syntax_token_states = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        else:
            assert self.feed_in_token_rnn_state_to_rule_rnn is False and \
                   self.fuse_rule_and_token_rnns is False and \
                   self.encode_change_vec_in_syntax_token_rnn is False

        self.syntax_tree_encoder = syntax_tree_encoder

        self.decoder_lstm = nn.LSTMCell(decoder_input_dim, hidden_size)
        self.decoder_cell_init = nn.Linear(source_element_encoding_size + change_vector_size, hidden_size)

        self.syntax_token_copy_ptr_net = PointerNet(src_encoding_size=source_element_encoding_size,
                                                    query_vec_size=hidden_size)

        # switch probability between copy and generation
        self.copy_gen_switch = nn.Linear(hidden_size, 3)

        if self.copy_subtree:
            self.sub_tree_copy_ptr_net = PointerNet(src_encoding_size=source_element_encoding_size, query_vec_size=hidden_size)

        self.attention_linear = nn.Linear(source_element_encoding_size, hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(source_element_encoding_size + hidden_size, hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.code_token_readout = nn.Linear(hidden_size, len(self.vocab), bias=False)

        self.dropout = nn.Dropout(dropout)
        if init_decode_vec_encoder_state_dropout > 0.:
            self.init_decode_vec_encoder_state_dropout = nn.Dropout(init_decode_vec_encoder_state_dropout)

        self.production_embedding = nn.Embedding(len(self.grammar) + 1, action_embed_size)
        self.field_embedding = nn.Embedding(len(self.grammar.prod_field2id), field_embed_size)
        self.token_embedding = nn.Embedding(len(vocab), action_embed_size)

        nn.init.xavier_normal_(self.production_embedding.weight.data)
        nn.init.xavier_normal_(self.field_embedding.weight.data)
        nn.init.xavier_normal_(self.token_embedding.weight.data)

        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(self.grammar) + 1).zero_())
        self.token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab)).zero_())

        self.att_vec_to_embed = nn.Linear(hidden_size, action_embed_size)

        self.production_readout = lambda q: F.linear(F.tanh(self.att_vec_to_embed(q)), self.production_embedding.weight,
                                                     bias=self.production_readout_b)

        self.terminal_token_readout = lambda q: F.linear(F.tanh(self.att_vec_to_embed(q)), self.token_embedding.weight,
                                                         bias=self.token_readout_b)

    @property
    def device(self):
        return self.production_embedding.weight.device

    def decode(self, examples, batched_prev_code, change_vectors, dec_init_vec, updated_syntax_token_encoding=None, embedding_cache=None, debug=False):
        batch_size = len(examples)

        # (batch_size, prev_code_len, encode_size)
        prev_code_att_linear = self.attention_linear(batched_prev_code.encoding)

        h_tm1 = dec_init_vec

        att_vecs = []
        log_att_weights = []
        history_states = []
        log_entries = []

        new_float_tensor = prev_code_att_linear.new

        att_tm1 = new_float_tensor(batch_size, self.hidden_size).zero_()
        zero_action_embed = new_float_tensor(self.action_embed_size).zero_()

        if self.use_syntax_token_rnn:
            zero_syntax_token_rnn_state = torch.zeros(self.hidden_size, device=self.device)

        max_time_step = max(len(e.tgt_actions) for e in examples)
        for t in range(max_time_step):
            # x = [prev_action, parent_hidden_state, frontier_field_embed, change_vector]
            actions_t = [e.tgt_actions[t] if t < len(e.tgt_actions) else None for e in examples]

            if t == 0:
                zero_input_size = self.decoder_lstm.input_size - self.change_vector_size
                if self.feed_in_token_rnn_state_to_rule_rnn:
                    zero_input_size -= self.hidden_size
                x = new_float_tensor(batch_size, zero_input_size).zero_()
                inputs = [x]
            else:
                actions_tm1 = [e.tgt_actions[t - 1] if t < len(e.tgt_actions) else None for e in examples]
                a_tm1_embeds = []
                for batch_id, action_tm1 in enumerate(actions_tm1):
                    if action_tm1:
                        if isinstance(action_tm1, ApplyRuleAction):
                            a_tm1_embed = self.production_embedding.weight[self.grammar.prod2id[action_tm1.production]]
                        elif isinstance(action_tm1, ApplySubTreeAction):
                            # embedding of the sub-tree from the GNN encoder
                            a_tm1_embed = batched_prev_code.encoding[batch_id, action_tm1.tree_node_ids[0]]
                        elif isinstance(action_tm1, ReduceAction):
                            a_tm1_embed = self.production_embedding.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.syntax_tree_encoder.syntax_tree_embedder.weight[self.vocab[action_tm1.token.value]]

                        a_tm1_embeds.append(a_tm1_embed)
                    else:
                        a_tm1_embeds.append(zero_action_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                parent_states = torch.stack([history_states[p_t][0][batch_id]
                                             for batch_id, p_t in
                                             enumerate(a_t.parent_t if a_t else 0 for a_t in actions_t)])

                frontier_field_idx = [self.grammar.prod_field2id[(action_t.frontier_prod, action_t.frontier_field)] if action_t else 0 for action_t in actions_t]
                frontier_field_embed = self.field_embedding(torch.tensor(frontier_field_idx, dtype=torch.long, device=self.device))

                inputs = [a_tm1_embeds, parent_states, frontier_field_embed]

            if self.use_syntax_token_rnn:
                syntax_token_states = []
                for batch_id, action_t in enumerate(actions_t):
                    if action_t:
                        token_rnn_state = updated_syntax_token_encoding[
                            batch_id, action_t.preceding_syntax_token_index + 1]
                    else:
                        token_rnn_state = zero_syntax_token_rnn_state

                    syntax_token_states.append(token_rnn_state)

                syntax_token_states = torch.stack(syntax_token_states)

                if self.feed_in_token_rnn_state_to_rule_rnn:
                    inputs.append(syntax_token_states)
            else:
                syntax_token_states = None

            inputs.append(change_vectors)
            x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, log_att_weight = self.step(x,
                                                             h_tm1,
                                                             syntax_token_states,
                                                             batched_prev_code.encoding, batched_prev_code.mask,
                                                             prev_code_att_linear)

            if debug:
                log_entry = {'x': x, 'h_tm1': h_tm1[0], 'c_tm1': h_tm1[1], 'h_t': h_t,
                             'att_t': att_t,
                             'prev_code_encoding': batched_prev_code.encoding}
                if self.use_syntax_token_rnn:
                    log_entry['syntax_token_rnn_h'] = syntax_token_states
                log_entries.append(log_entry)

            history_states.append((h_t, cell_t))
            att_vecs.append(att_t)
            log_att_weights.append(log_att_weight)

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        att_vecs = torch.stack(att_vecs)
        log_att_weights = torch.stack(log_att_weights)

        if debug:
            return att_vecs, log_att_weights, log_entries
        else:
            return att_vecs, log_att_weights

    def forward(self, batch_examples, batched_prev_code, batched_context, change_vectors, embedding_cache=None, debug=False):
        # (batch_size, hidden_size)
        dec_init_vec = self.get_init_hidden_state(batched_prev_code, batched_context, change_vectors)

        updated_syntax_token_encoding = None
        if self.use_syntax_token_rnn:
            updated_syntax_token_encoding = self.run_syntax_token_rnn(batch_examples, dec_init_vec, change_vectors)

        # att_vecs: (tgt_action_len, batch_size, hidden_size)
        # att_weights: (tgt_action_len, batch_size, src_ast_node_num)

        decode_returns = self.decode(batch_examples, batched_prev_code, change_vectors, dec_init_vec,
                                            updated_syntax_token_encoding=updated_syntax_token_encoding,
                                            embedding_cache=embedding_cache, debug=debug)
        if debug:
            att_vecs, att_weights, log_entries = decode_returns
        else:
            att_vecs, att_weights = decode_returns

        # prepare masks, target indices
        tgt_apply_rule_idx, tgt_apply_rule_mask, apply_rule_cand_mask, \
        tgt_apply_subtree_idx, tgt_apply_subtree_idx_mask, tgt_apply_subtree_mask, apply_subtree_cand_mask, \
        tgt_gen_token_idx, tgt_gen_token_mask, \
        tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
        tgt_copy_prev_token_idx_mask, tgt_copy_prev_token_mask = self.get_gen_and_copy_index_and_mask(
            batch_examples,
            batched_prev_code,
            batched_context,
            no_penalize_apply_tree_when_copy_subtree=self.no_penalize_apply_tree_when_copy_subtree and self.training)

        # ApplyRule action probability
        # (tgt_action_len, batch_size, grammar_size + 1)
        apply_rule_logits = self.production_readout(att_vecs)

        if self.copy_subtree:
            # ApplySubTree action probability
            # (tgt_action_len, batch_size, src_ast_node_num)
            apply_subtree_cand_mask[apply_subtree_cand_mask.sum(dim=-1).eq(0), :] = 1.    # safeguard mask
            apply_subtree_logits = self.sub_tree_copy_ptr_net(batched_prev_code.encoding, None,
                                                              att_vecs,
                                                              return_logits=True)
            apply_rule_and_subtree_logits = torch.cat([apply_rule_logits, apply_subtree_logits], dim=-1)
            apply_rule_and_subtree_cand_mask = torch.cat([apply_rule_cand_mask, apply_subtree_cand_mask], dim=-1)
        else:
            apply_rule_and_subtree_logits = apply_rule_logits
            apply_rule_and_subtree_cand_mask = apply_rule_cand_mask

        # (tgt_action_len, batch_size, grammar_size + 1 + src_ast_node_num)
        apply_rule_and_subtree_log_prob = F.log_softmax(apply_rule_and_subtree_logits + (apply_rule_and_subtree_cand_mask + 1e-45).log(), dim=-1)

        # (tgt_action_len, batch_size, terminal_vocab_size)
        gen_terminal_token_log_prob = F.log_softmax(self.terminal_token_readout(att_vecs), dim=-1)

        # (tgt_action_len, batch_size, ctx_len)
        copy_ctx_token_prob = self.syntax_token_copy_ptr_net(batched_context.encoding, batched_context.mask, att_vecs)

        # (tgt_action_len, batch_size, ctx_len)
        copy_prev_token_prob = self.syntax_token_copy_ptr_net(batched_prev_code.encoding,
                                                              batched_prev_code.syntax_token_mask,
                                                              att_vecs)

        # (tgt_action_len, batch_size, [COPY_FROM_PREV, COPY_FROM_CONTEXT, GEN])
        token_copy_gen_switch = F.log_softmax(self.copy_gen_switch(att_vecs), dim=-1)

        # get the target probabilities

        # ApplyRule (tgt_action_len, batch_size)
        tgt_apply_rule_prob = torch.gather(apply_rule_and_subtree_log_prob, dim=-1,
                                           index=tgt_apply_rule_idx.unsqueeze(2)).squeeze(2)

        tgt_gated_apply_rule_prob = tgt_apply_rule_prob
        tgt_gated_apply_rule_prob = tgt_gated_apply_rule_prob * tgt_apply_rule_mask

        if self.copy_subtree:
            # ApplySubTree (tgt_action_len, batch_size, max_cand_node_num)
            tgt_apply_subtree_prob = torch.gather(apply_rule_and_subtree_log_prob, dim=-1,
                                                 index=tgt_apply_subtree_idx + len(self.grammar) + 1)
            tgt_apply_subtree_prob = nn_utils.log_sum_exp(tgt_apply_subtree_prob, mask=tgt_apply_subtree_idx_mask)
            tgt_apply_subtree_prob[tgt_apply_subtree_prob == -float('inf')] = 0.

            tgt_gated_apply_subtree_prob = tgt_apply_subtree_prob

            tgt_gated_apply_subtree_prob = tgt_gated_apply_subtree_prob * tgt_apply_subtree_mask

        # GenToken (tgt_action_len, batch_size)
        tgt_gen_token_prob = torch.gather(gen_terminal_token_log_prob, dim=-1,
                                          index=tgt_gen_token_idx.unsqueeze(2)).squeeze(2)
        tgt_gen_selection_prob = token_copy_gen_switch[:, :, 2]
        gated_tgt_gen_token_prob = tgt_gen_token_prob + tgt_gen_selection_prob

        tgt_copy_ctx_token_prob = (torch.sum(copy_ctx_token_prob * tgt_copy_ctx_token_idx_mask, dim=-1) + 1.e-15).log()

        tgt_copy_ctx_selection_prob = token_copy_gen_switch[:, :, 1]
        gated_tgt_copy_ctx_token_prob = tgt_copy_ctx_token_prob + tgt_copy_ctx_selection_prob

        tgt_copy_prev_token_prob = (torch.sum(copy_prev_token_prob * tgt_copy_prev_token_idx_mask, dim=-1) + 1.e-15).log()

        tgt_copy_prev_selection_prob = token_copy_gen_switch[:, :, 0]
        gated_tgt_copy_prev_token_prob = tgt_copy_prev_token_prob + tgt_copy_prev_selection_prob

        # tgt_gen_and_copy_token_prob = tgt_gen_token_prob + tgt_copy_ctx_token_prob + tgt_copy_prev_token_prob
        # token_mask = tgt_gen_token_mask + tgt_copy_ctx_token_mask + tgt_copy_prev_token_mask
        # token_mask = torch.ge(token_mask, 1).float()

        tgt_gen_and_copy_token_prob = nn_utils.log_sum_exp(torch.stack([gated_tgt_gen_token_prob, gated_tgt_copy_ctx_token_prob, gated_tgt_copy_prev_token_prob], dim=-1),
                                                           mask=torch.stack([tgt_gen_token_mask, tgt_copy_ctx_token_mask, tgt_copy_prev_token_mask], dim=-1))
        tgt_gen_and_copy_token_prob[tgt_gen_and_copy_token_prob == -float('inf')] = 0.

        if self.copy_subtree:
            tgt_actions_prob = tgt_gated_apply_rule_prob + tgt_gated_apply_subtree_prob + tgt_gen_and_copy_token_prob
        else:
            tgt_actions_prob = tgt_gated_apply_rule_prob + tgt_gen_and_copy_token_prob

        if debug:
            debug_info = OrderedDict()

            for batch_id, example in enumerate(batch_examples):
                action_trace = []
                log_p = 0.0
                for t, action_t in enumerate(example.tgt_actions):
                    if isinstance(action_t, (ApplyRuleAction, ReduceAction)):
                        p_t = tgt_gated_apply_rule_prob[t, batch_id].item()
                        entry = {'t': t,
                                 'action': repr(action_t),
                                 'apply_rule_prob': tgt_apply_rule_prob[t, batch_id].item(),
                                 'p_t': p_t}
                    elif isinstance(action_t, ApplySubTreeAction):
                        p_t = tgt_gated_apply_subtree_prob[t, batch_id].item()
                        entry = {'t': t,
                                 'action': repr(action_t),
                                 'apply_subtree_prob': tgt_apply_subtree_prob[t, batch_id].item(),
                                 'p_t': p_t}
                    elif isinstance(action_t, GenTokenAction):
                        p_t = tgt_gen_and_copy_token_prob[t, batch_id].item()
                        entry = {'t': t,
                                 'copy_gen_switch': token_copy_gen_switch[t, batch_id].cpu().numpy(),
                                 'tgt_gen_token_prob': tgt_gen_token_prob[t, batch_id].item() if tgt_gen_token_mask[t, batch_id].item() else 'n/a',
                                 'tgt_copy_prev_token_prob': tgt_copy_prev_token_prob[t, batch_id].item() if tgt_copy_prev_token_mask[t, batch_id].item() else 'n/a',
                                 'tgt_copy_ctx_token_prob': tgt_copy_ctx_token_prob[t, batch_id].item() if tgt_copy_ctx_token_mask[t, batch_id].item() else 'n/a',
                                 'action': repr(action_t),
                                 'p_t': p_t}
                    else:
                        raise RuntimeError('unknown action!')

                    entry.update({k: v[batch_id] for k, v in log_entries[t].items()})

                    log_p += p_t
                    action_trace.append(entry)

                debug_info[example.id] = dict(action_trace=action_trace, log_p=log_p)

        # (batch_size)
        tgt_actions_prob = tgt_actions_prob.sum(dim=0)

        returns = {'log_probs': tgt_actions_prob}

        if debug:
            returns['debug_info'] = debug_info

        return returns

    def run_syntax_token_rnn(self, examples, dev_init_vec=None, change_vectors=None):
        syntax_tokens = [e.updated_code_ast.syntax_tokens for e in examples]
        syntax_token_values = [['<s>'] + [token.value for token in tokens] for tokens in syntax_tokens]
        # (seq_len, batch_size, encoding_size)
        inputs = self.syntax_tree_encoder.syntax_tree_embedder.get_embed_for_token_sequences(syntax_token_values)

        if self.encode_change_vec_in_syntax_token_rnn:
            inputs = torch.cat([inputs, change_vectors.unsqueeze(0).expand(inputs.size(0), change_vectors.size(0), -1)], dim=-1)

        # (token_num, batch_size, hidden_size)
        token_encodings, (_, _) = self.syntax_token_rnn(inputs, (dev_init_vec[0].unsqueeze(0), dev_init_vec[1].unsqueeze(0)))
        token_encodings = token_encodings.permute(1, 0, 2)

        return token_encodings

    def syntax_token_rnn_step(self, tokens, h_tm1=None, change_vectors=None):
        if h_tm1 is not None:
            h_tm1 = (h_tm1[0].unsqueeze(0), h_tm1[1].unsqueeze(0))

        if any(isinstance(x, list) for x in tokens):
            lens = torch.tensor([len(x) for x in tokens], dtype=torch.long, device=self.device)
            lens_sorted, sorted_idx = lens.sort(descending=True)

            # (batch_size, max_seq_len)
            word_ids_padded = nn_utils.to_input_variable(tokens, vocab=self.vocab, device=self.device, batch_first=True)
            word_ids_sorted = word_ids_padded[sorted_idx]
            # (batch_size, max_seq_len, embed_size)
            inputs = self.syntax_tree_encoder.syntax_tree_embedder(torch.tensor(word_ids_sorted, dtype=torch.long, device=self.device))

            if self.encode_change_vec_in_syntax_token_rnn:
                inputs = torch.cat(
                    [inputs, change_vectors.unsqueeze(1).expand(inputs.size(0), inputs.size(1), -1)], dim=-1)

            packed_word_embeds = pack_padded_sequence(inputs, lens_sorted.tolist(), batch_first=True)
            h_tm1_sorted = (h_tm1[0][:, sorted_idx], h_tm1[1][:, sorted_idx])
            _, (h_t, c_t) = self.syntax_token_rnn(packed_word_embeds, h_tm1_sorted)
            h_t = h_t.squeeze(0)
            c_t = c_t.squeeze(0)

            idx = sorted_idx.unsqueeze(1).expand(-1, h_t.size(1))
            unsorted_h_t = torch.zeros_like(h_t).scatter_(0, idx, h_t)
            unsorted_c_t = torch.zeros_like(c_t).scatter_(0, idx, c_t)

            h_t = (unsorted_h_t, unsorted_c_t)
        else:
            word_ids = [self.vocab[token] for token in tokens]
            inputs = self.syntax_tree_encoder.syntax_tree_embedder(torch.tensor(word_ids, dtype=torch.long, device=self.device))

            if self.encode_change_vec_in_syntax_token_rnn:
                inputs = torch.cat([inputs, change_vectors], dim=-1)

            inputs = inputs.unsqueeze(0)

            _, (h_t, c_t) = self.syntax_token_rnn(inputs, h_tm1)
            h_t = (h_t.squeeze(0), c_t.squeeze(0))

        return h_t

    def step(self, x, h_tm1, syntax_token_states, batched_prev_code_encoding, batched_prev_code_mask, prev_code_att_linear):
        """
        a single LSTM decoding step
        """

        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        if self.fuse_rule_and_token_rnns:
            att_query_vec = F.tanh(self.fuse_rule_and_syntax_token_states(torch.cat([h_t, syntax_token_states], dim=-1)))
        else:
            att_query_vec = h_t

        ctx_t, alpha_t, log_alpha_t = nn_utils.dot_prod_attention(att_query_vec,
                                                                  batched_prev_code_encoding, prev_code_att_linear,
                                                                  mask=batched_prev_code_mask,
                                                                  return_log_att_weight=True)

        att_t = F.tanh(self.att_vec_linear(torch.cat([att_query_vec, ctx_t], 1)))
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t, log_alpha_t

    def get_gen_and_copy_index_and_mask(self, examples, batched_prev_code, batched_context, no_penalize_apply_tree_when_copy_subtree=False):
        batch_size = len(examples)
        max_seq_len = max(len(e.tgt_actions) for e in examples)

        tgt_apply_rule_idx = torch.zeros(max_seq_len, batch_size, dtype=torch.long)
        tgt_apply_rule_mask = torch.zeros(max_seq_len, batch_size, dtype=torch.float)
        apply_rule_cand_mask = torch.zeros(max_seq_len, batch_size, len(self.grammar) + 1, dtype=torch.float)

        # get the maximum number of candidate subtree nodes to copy in an action
        max_cand_subtree_node_num = max(list(chain.from_iterable(
            [len(a_t.tree_node_ids) for a_t in e.tgt_actions if isinstance(a_t, ApplySubTreeAction)] for e in
            examples)) + [1])
        tgt_apply_subtree_idx = torch.zeros(max_seq_len, batch_size, max_cand_subtree_node_num, dtype=torch.long)
        tgt_apply_subtree_idx_mask = torch.zeros(max_seq_len, batch_size, max_cand_subtree_node_num, dtype=torch.float)
        tgt_apply_subtree_mask = torch.zeros(max_seq_len, batch_size, dtype=torch.float)
        apply_subtree_cand_mask = torch.zeros(max_seq_len, batch_size, batched_prev_code.encoding.size(1), dtype=torch.float)

        tgt_gen_token_idx = torch.zeros(max_seq_len, batch_size, dtype=torch.long)
        tgt_gen_token_mask = torch.zeros(max_seq_len, batch_size, dtype=torch.float)

        tgt_copy_ctx_token_idx_mask = torch.zeros(max_seq_len, batch_size, batched_context.encoding.size(1), dtype=torch.float)
        tgt_copy_ctx_token_mask = torch.zeros(max_seq_len, batch_size, dtype=torch.float)

        tgt_copy_prev_token_idx_mask = torch.zeros(max_seq_len, batch_size, batched_prev_code.encoding.size(1), dtype=torch.float)
        tgt_copy_prev_token_mask = torch.zeros(max_seq_len, batch_size, dtype=torch.float)

        for batch_id in range(batch_size):
            prev_code_ast = examples[batch_id].prev_code_ast
            context = examples[batch_id].context
            example = examples[batch_id]
            tgt_actions = example.tgt_actions
            for t, action_t in enumerate(tgt_actions):
                if isinstance(action_t, (ApplyRuleAction, ReduceAction)):
                    app_rule_idx = self.grammar.prod2id[action_t.production] if isinstance(action_t, ApplyRuleAction) else len(self.grammar)
                    tgt_apply_rule_idx[t, batch_id] = app_rule_idx
                    tgt_apply_rule_mask[t, batch_id] = 1
                elif isinstance(action_t, ApplySubTreeAction):
                    tgt_apply_subtree_idx[t, batch_id, :len(action_t.tree_node_ids)] = torch.tensor(
                        action_t.tree_node_ids, dtype=torch.long, device=self.device)
                    tgt_apply_subtree_idx_mask[t, batch_id, :len(action_t.tree_node_ids)] = 1
                    tgt_apply_subtree_mask[t, batch_id] = 1
                else:
                    tgt_token = action_t.token
                    if SequentialDecoder._can_only_generate_this_token(tgt_token.value):
                        tgt_gen_token_mask[t, batch_id] = 1
                        tgt_gen_token_idx[t, batch_id] = self.vocab[tgt_token.value]
                    else:
                        copied = False
                        if self.copy_syntax_token:
                            if tgt_token in prev_code_ast.syntax_tokens_set:
                                token_pos_list = [pos for pos, syntax_token in prev_code_ast.syntax_tokens_and_ids if syntax_token == tgt_token]
                                tgt_copy_prev_token_mask[t, batch_id] = 1
                                tgt_copy_prev_token_idx_mask[t, batch_id, token_pos_list] = 1
                                copied = True
                            if tgt_token.value in context:
                                token_pos_list = [pos for pos, token in enumerate(context) if token == tgt_token.value]
                                tgt_copy_ctx_token_idx_mask[t, batch_id, token_pos_list] = 1
                                tgt_copy_ctx_token_mask[t, batch_id] = 1
                                copied = True

                        if not copied or tgt_token.value in self.vocab:
                            # if the token is not copied, we can only generate this token from the vocabulary,
                            # even if it is a <unk>.
                            # otherwise, we can still generate it from the vocabulary
                            tgt_gen_token_mask[t, batch_id] = 1
                            tgt_gen_token_idx[t, batch_id] = self.vocab[tgt_token.value]

                if isinstance(action_t, (ApplyRuleAction, ReduceAction, ApplySubTreeAction)):
                    valid_cont_prod_ids = action_t.valid_continuating_production_ids
                    valid_cont_subtree_ids = action_t.valid_continuating_subtree_ids

                    apply_rule_cand_mask[t, batch_id, valid_cont_prod_ids] = 1
                    apply_subtree_cand_mask[t, batch_id, valid_cont_subtree_ids] = 1.

        tgt_apply_rule_idx = tgt_apply_rule_idx.to(self.device)
        tgt_apply_rule_mask = tgt_apply_rule_mask.to(self.device)
        apply_rule_cand_mask = apply_rule_cand_mask.to(self.device)
        tgt_apply_subtree_idx = tgt_apply_subtree_idx.to(self.device)
        tgt_apply_subtree_idx_mask = tgt_apply_subtree_idx_mask.to(self.device)
        tgt_apply_subtree_mask = tgt_apply_subtree_mask.to(self.device)
        apply_subtree_cand_mask = apply_subtree_cand_mask.to(self.device)
        tgt_gen_token_idx = tgt_gen_token_idx.to(self.device)
        tgt_gen_token_mask = tgt_gen_token_mask.to(self.device)
        tgt_copy_ctx_token_idx_mask = tgt_copy_ctx_token_idx_mask.to(self.device)
        tgt_copy_ctx_token_mask = tgt_copy_ctx_token_mask.to(self.device)
        tgt_copy_prev_token_idx_mask = tgt_copy_prev_token_idx_mask.to(self.device)
        tgt_copy_prev_token_mask = tgt_copy_prev_token_mask.to(self.device)

        return tgt_apply_rule_idx, tgt_apply_rule_mask, apply_rule_cand_mask, \
               tgt_apply_subtree_idx, tgt_apply_subtree_idx_mask, tgt_apply_subtree_mask, apply_subtree_cand_mask, \
               tgt_gen_token_idx, tgt_gen_token_mask, \
               tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
               tgt_copy_prev_token_idx_mask, tgt_copy_prev_token_mask

    def get_init_hidden_state(self, batched_prev_code, batched_context, change_vectors):
        if self.decoder_init_method == 'avg_pooling':
            node_state = batched_prev_code.encoding.sum(dim=1) / ((1. - batched_prev_code.mask.float()).sum(dim=-1, keepdim=True))
        elif self.decoder_init_method == 'root_node':
            node_state = batched_prev_code.encoding[:, 0]

        if hasattr(self, 'init_decode_vec_encoder_state_dropout'):
            node_state = self.init_decode_vec_encoder_state_dropout(node_state)

        x = torch.cat([node_state, change_vectors], dim=-1)
        dec_init_cell = self.decoder_cell_init(x)
        dec_init_state = F.tanh(dec_init_cell)

        return dec_init_state, dec_init_cell

    def beam_search_with_source_encodings(self, prev_code, prev_code_encoding, context, context_encoding, change_vector,
                                          transition_system, beam_size=5, max_decoding_time_step=100, debug=False):
        # prev_code_encoding: (1, src_seq_len, encoding_size)
        # change_vec: (1, change_vec_size)
        # dec_init_vec: Tuple[(1, hidden_size)]

        dec_init_vec = self.get_init_hidden_state(prev_code_encoding, context_encoding, change_vector)

        aggregated_prev_code_tokens = OrderedDict()
        # if self.mode.startswith('tree2tree'):
        for token_pos, token in prev_code.syntax_tokens_and_ids:
            aggregated_prev_code_tokens.setdefault(token.value, []).append(token_pos)
        # else:
        #     for token_pos, token in enumerate(prev_code):
        #         aggregated_prev_code_tokens.setdefault(token, []).append(token_pos)

        aggregated_context_tokens = OrderedDict()
        for token_pos, token in enumerate(context):
            aggregated_context_tokens.setdefault(token, []).append(token_pos)

        # (1, prev_code_len, encode_size)
        prev_code_att_linear = self.attention_linear(prev_code_encoding.encoding)

        h_tm1 = dec_init_vec

        t = 0
        # hypotheses = [CSharpHypothesis()]
        hypotheses = [self.lambda_hypothesis()]

        # some meta-info need to be tracked in the hypothesis
        hypotheses[0].action_log = []

        hyp_states = [[]]
        hyp_scores = [hypotheses[0].score]
        completed_hypotheses = []
        # initiate the syntax token rnn states
        if self.use_syntax_token_rnn:
            syntax_token_rnn_states = self.syntax_token_rnn_step(tokens=['<s>'], h_tm1=dec_init_vec, change_vectors=change_vector)

        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = prev_code_encoding.encoding.expand(hyp_num, prev_code_encoding.encoding.size(1), prev_code_encoding.encoding.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = prev_code_att_linear.expand(hyp_num, prev_code_att_linear.size(1), prev_code_att_linear.size(2))
            # (hyp_num, change_vec_size)
            exp_change_vector = change_vector.expand(hyp_num, change_vector.size(1))

            apply_rule_cand_mask = torch.zeros(hyp_num, len(self.grammar) + 1, dtype=torch.float, device=self.device)
            if self.copy_subtree:
                apply_subtree_cand_mask = torch.zeros(hyp_num, prev_code_encoding.encoding.size(1), dtype=torch.float, device=self.device)

            if t == 0:
                zero_input_size = self.decoder_lstm.input_size - self.change_vector_size
                if self.feed_in_token_rnn_state_to_rule_rnn:
                    zero_input_size -= self.hidden_size
                x = torch.zeros(hyp_num, zero_input_size, device=self.device)
                inputs = [x]
            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]
                a_tm1_embeds = []
                for hyp_id, action_tm1 in enumerate(actions_tm1):
                    if isinstance(action_tm1, ApplyRuleAction):
                        a_tm1_embed = self.production_embedding.weight[self.grammar.prod2id[action_tm1.production]]
                    elif isinstance(action_tm1, ApplySubTreeAction):
                        # embedding of the sub-tree from the GNN encoder
                        a_tm1_embed = prev_code_encoding.encoding[0, action_tm1.tree_node_ids[0]]
                    elif isinstance(action_tm1, ReduceAction):
                        a_tm1_embed = self.production_embedding.weight[len(self.grammar)]
                    else:
                        a_tm1_embed = self.syntax_tree_encoder.syntax_tree_embedder.weight[self.vocab[action_tm1.token.value]]

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                # parent states
                p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                parent_states = torch.stack([hyp_states[hyp_id][p_t][0] for hyp_id, p_t in enumerate(p_ts)])

                frontier_field_idx = [
                    self.grammar.prod_field2id[(hyp.frontier_node.production, hyp.frontier_field.field)] for
                    hyp in hypotheses]
                frontier_field_embed = self.field_embedding(torch.tensor(frontier_field_idx, dtype=torch.long, device=self.device))

                inputs = [a_tm1_embeds, parent_states, frontier_field_embed]

            if self.feed_in_token_rnn_state_to_rule_rnn:
                inputs.append(syntax_token_rnn_states[0])

            inputs.append(exp_change_vector)
            x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, log_att_weight = self.step(x,
                                                             h_tm1,
                                                             syntax_token_rnn_states[0] if self.use_syntax_token_rnn else None,
                                                             exp_src_encodings, batched_prev_code_mask=None,
                                                             prev_code_att_linear=exp_src_encodings_att_linear)

            if debug:
                log_entry = {'x': x, 'h_tm1': h_tm1[0], 'c_tm1': h_tm1[1], 'h_t': h_t,
                             'att_t': att_t,
                             'prev_code_encoding': exp_src_encodings}

            # first, for fine grained masking, computing the masks based on action types
            for hyp_id, hyp in enumerate(hypotheses):
                # generate new continuations
                action_types = transition_system.get_valid_continuation_types(hyp)

                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        if hyp.frontier_field:
                            valid_cont_prods = self.grammar[hyp.frontier_field.type]
                        else:
                            valid_cont_prods = [transition_system.starting_production]
                        valid_cont_prod_ids = [self.grammar.prod2id[prod] for prod in valid_cont_prods]
                        apply_rule_cand_mask[hyp_id, valid_cont_prod_ids] = 1
                    elif action_type == ReduceAction:
                        valid_cont_prod_ids = [len(self.grammar)]
                        apply_rule_cand_mask[hyp_id, valid_cont_prod_ids] = 1
                    elif action_type == ApplySubTreeAction and t > 0 and self.copy_subtree:
                        valid_subtree_types = self.grammar.descendant_types[hyp.frontier_field.type]
                        valid_subtree_ids = [node_id for node_id, node in prev_code.descendant_nodes
                                             if node.production.type in valid_subtree_types]
                        apply_subtree_cand_mask[hyp_id, valid_subtree_ids] = 1

            # now let's compute action probabilities

            # ApplyRule action probability
            # (batch_size, grammar_size + 1)
            apply_rule_logits = self.production_readout(att_t)

            if self.copy_subtree:
                # ApplySubTree action probability
                # (batch_size, src_ast_node_num)

                apply_subtree_logits = self.sub_tree_copy_ptr_net(exp_src_encodings, None,
                                                                  att_t.unsqueeze(0),
                                                                  valid_masked_as_one=True,
                                                                  return_logits=True).squeeze(0)

                apply_rule_and_subtree_logits = torch.cat([apply_rule_logits, apply_subtree_logits], dim=-1)
                apply_rule_and_subtree_cand_mask = torch.cat([apply_rule_cand_mask, apply_subtree_cand_mask], dim=-1)
            else:
                apply_rule_and_subtree_logits = apply_rule_logits
                apply_rule_and_subtree_cand_mask = apply_rule_cand_mask

            apply_rule_and_subtree_log_prob = F.log_softmax(apply_rule_and_subtree_logits + (apply_rule_and_subtree_cand_mask + 1e-45).log(), dim=-1)

            gated_apply_rule_log_prob = apply_rule_and_subtree_log_prob[:, :len(self.grammar) + 1]

            if self.copy_subtree:
                gated_apply_subtree_log_prob = apply_rule_and_subtree_log_prob[:, len(self.grammar) + 1:]

            # (batch_size, terminal_vocab_size)
            gen_terminal_token_prob = F.softmax(self.terminal_token_readout(att_t), dim=-1)

            # (batch_size, ctx_len)
            copy_ctx_token_prob = self.syntax_token_copy_ptr_net(context_encoding.encoding.expand(hyp_num,
                                                                                                  context_encoding.encoding.size(1),
                                                                                                  context_encoding.encoding.size(2)),
                                                                 None, att_t.unsqueeze(0)).squeeze(0)

            # (batch_size, ctx_len)
            copy_prev_token_prob = self.syntax_token_copy_ptr_net(prev_code_encoding.encoding,
                                                                  prev_code_encoding.syntax_token_mask.expand(hyp_num, prev_code_encoding.syntax_token_mask.size(1)),
                                                                  # prev_code_encoding.syntax_token_mask.expand(hyp_num, prev_code_encoding.syntax_token_mask.size(1))
                                                                  # if self.mode.startswith('tree2tree') else None,
                                                                  att_t.unsqueeze(0)).squeeze(0)

            # (batch_size, [COPY_FROM_PREV, COPY_FROM_CONTEXT, GEN])
            token_copy_gen_switch = F.softmax(self.copy_gen_switch(att_t), dim=-1)

            terminal_token_prob = token_copy_gen_switch[:, 2].unsqueeze(1) * gen_terminal_token_prob

            new_hyp_meta = []
            gentoken_prev_hyp_ids = []
            for hyp_id, hyp in enumerate(hypotheses):
                # generate new continuations
                action_types = transition_system.get_valid_continuation_types(hyp)

                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            prod_score = gated_apply_rule_log_prob[hyp_id, prod_id]
                            new_hyp_score = hyp.score + prod_score

                            meta_entry = {'action_type': 'apply_rule', 'prod_id': prod_id,
                                          'score': prod_score, 'new_hyp_score': new_hyp_score,
                                          'prev_hyp_id': hyp_id}
                            if debug:
                                meta_entry['apply_rule_prob'] = gated_apply_rule_log_prob[hyp_id, prod_id].item()

                            new_hyp_meta.append(meta_entry)
                    elif action_type == ReduceAction:
                        action_score = gated_apply_rule_log_prob[hyp_id, len(self.grammar)]
                        new_hyp_score = hyp.score + action_score

                        meta_entry = {'action_type': 'apply_rule', 'prod_id': len(self.grammar),
                                      'score': action_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        if debug:
                            meta_entry['apply_rule_prob'] = gated_apply_rule_log_prob[hyp_id, len(self.grammar)].item()

                        new_hyp_meta.append(meta_entry)
                    elif action_type == ApplySubTreeAction:
                        # do not allow ApplySubTree action at the beginning!
                        if not self.copy_subtree or hyp.frontier_field is None:
                            continue

                        # we can only apply sub trees with the same type!!
                        valid_continuating_types = self.grammar.descendant_types[hyp.frontier_field.type]
                        for node_id, node in prev_code.descendant_nodes:
                            # if node.production in valid_continuating_productions:
                            if node.production.type in valid_continuating_types and not (node.production.type.name == 'IdentifierNameSyntax' and not self.copy_identifier_node):
                                subtree_score = gated_apply_subtree_log_prob[hyp_id, node_id]
                                new_hyp_score = hyp.score + subtree_score

                                meta_entry = {'action_type': 'apply_subtree',
                                              'subtree_id': node_id, 'tree': node,
                                              'score': subtree_score, 'new_hyp_score': new_hyp_score,
                                              'prev_hyp_id': hyp_id}
                                if debug:
                                    meta_entry['apply_subtree_prob'] = gated_apply_subtree_log_prob[hyp_id, node_id].item()

                                new_hyp_meta.append(meta_entry)
                    elif action_type == GenTokenAction:
                        gentoken_prev_hyp_ids.append(hyp_id)
                        hyp_unk_copy_score = OrderedDict()

                        if self.copy_syntax_token:
                            for token, token_pos_list in aggregated_prev_code_tokens.items():
                                sum_copy_prob = copy_prev_token_prob[hyp_id, token_pos_list].sum()
                                gated_copy_prob = token_copy_gen_switch[hyp_id, 0] * sum_copy_prob

                                if token in self.vocab:
                                    token_id = self.vocab[token]
                                    terminal_token_prob[hyp_id, token_id] = terminal_token_prob[hyp_id, token_id] + gated_copy_prob
                                else:
                                    if token in hyp_unk_copy_score:
                                        hyp_unk_copy_score[token] = hyp_unk_copy_score[token] + gated_copy_prob
                                    else:
                                        hyp_unk_copy_score[token] = gated_copy_prob

                            for token, token_pos_list in aggregated_context_tokens.items():
                                sum_copy_prob = copy_ctx_token_prob[hyp_id, token_pos_list].sum()
                                gated_copy_prob = token_copy_gen_switch[hyp_id, 1] * sum_copy_prob

                                if token in self.vocab:
                                    token_id = self.vocab[token]
                                    terminal_token_prob[hyp_id, token_id] = terminal_token_prob[hyp_id, token_id] + gated_copy_prob
                                else:
                                    if token in hyp_unk_copy_score:
                                        hyp_unk_copy_score[token] = hyp_unk_copy_score[token] + gated_copy_prob
                                    else:
                                        hyp_unk_copy_score[token] = gated_copy_prob

                        for token, token_score in hyp_unk_copy_score.items():
                            copy_log_prob = token_score.log()
                            new_hyp_meta.append({'action_type': 'gen_token',
                                                 'prev_hyp_id': hyp_id, 'token': token,
                                                 'new_hyp_score': hyp.score + copy_log_prob,
                                                 'copy_log_prob': copy_log_prob})

            new_hyp_scores = None
            if new_hyp_meta:
                new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta])
            if gentoken_prev_hyp_ids:
                terminal_token_prob = terminal_token_prob.log()
                gen_token_new_hyp_scores = (
                            hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + terminal_token_prob[gentoken_prev_hyp_ids, :]).view(-1)

                if new_hyp_scores is not None:
                    new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])
                else:
                    new_hyp_scores = gen_token_new_hyp_scores

            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0),
                                                                   beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            token_rnn_updated_hyp_ids = []
            token_rnn_tokens_to_update = []
            for new_hyp_idx, (new_hyp_score, new_hyp_flattened_pos) in enumerate(zip(top_new_hyp_scores, top_new_hyp_pos)):
                new_hyp_flattened_pos = new_hyp_flattened_pos.cpu().item()
                if new_hyp_flattened_pos < len(new_hyp_meta) and new_hyp_meta[new_hyp_flattened_pos]['action_type'] != 'gen_token':
                    hyp_meta_entry = new_hyp_meta[new_hyp_flattened_pos]
                    action_type_str = hyp_meta_entry['action_type']
                    prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                    prev_hyp = hypotheses[prev_hyp_id]

                    if action_type_str == 'apply_rule':
                        prod_id = hyp_meta_entry['prod_id']
                        if prod_id < len(self.grammar):
                            production = self.grammar.id2prod[prod_id]
                            action = ApplyRuleAction(production)
                        else:
                            action = ReduceAction()

                        if debug:
                            action_log_entry = {'t': t,
                                                'action': repr(action),
                                                'apply_rule_prob': hyp_meta_entry['apply_rule_prob'],
                                                'p_t': hyp_meta_entry['score'].item()}
                    elif action_type_str == 'apply_subtree':
                        sub_tree = hyp_meta_entry['tree']
                        action = ApplySubTreeAction(tree=sub_tree.copy(), tree_node_ids=[hyp_meta_entry['subtree_id']])

                        if debug:
                            action_log_entry = {'t': t,
                                                'action': repr(action),
                                                'apply_subtree_prob': hyp_meta_entry['apply_subtree_prob'],
                                                'p_t': hyp_meta_entry['score'].item()}
                    else:
                        raise ValueError(f'invalid action string {action_type_str}')
                else:
                    # GenToken action
                    if new_hyp_flattened_pos < len(new_hyp_meta):
                        # copying a unk-word from previous code or context
                        hyp_meta_entry = new_hyp_meta[new_hyp_flattened_pos]
                        token = hyp_meta_entry['token']
                        prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                    else:
                        token_id = (new_hyp_flattened_pos - len(new_hyp_meta)) % terminal_token_prob.size(1)
                        token = self.vocab.id2word[token_id]

                        k = (new_hyp_flattened_pos - len(new_hyp_meta)) // terminal_token_prob.size(1)
                        prev_hyp_id = gentoken_prev_hyp_ids[k]

                    prev_hyp = hypotheses[prev_hyp_id]

                    # this special token shall only be used to signal the end of generating a list of terminal tokens
                    if prev_hyp.frontier_field.cardinality == 'single' and \
                            (self.grammar.language != 'csharp' or token == transition_system.END_OF_SYNTAX_TOKEN_LIST_SYMBOL):
                        continue

                    action = GenTokenAction(SyntaxToken(type=prev_hyp.frontier_field.type, value=token))

                    if debug:
                        action_log_entry = {'t': t,
                                            'action': repr(action),
                                            'token_copy_gen_switch': token_copy_gen_switch[prev_hyp_id,
                                                                     :].log().cpu().numpy(),
                                            'in_vocab': token in self.vocab,
                                            'tgt_gen_token_prob': gen_terminal_token_prob[prev_hyp_id, self.vocab[
                                                token]].log().item() if token in self.vocab else 'n/a',
                                            'tgt_copy_prev_token_prob': copy_prev_token_prob[
                                                prev_hyp_id, aggregated_prev_code_tokens[
                                                    token]].sum().log().item() if token in aggregated_prev_code_tokens else 'n/a',
                                            'tgt_copy_ctx_token_prob': copy_ctx_token_prob[
                                                prev_hyp_id, aggregated_context_tokens[
                                                    token]].sum().log().item() if token in aggregated_context_tokens else 'n/a',
                                            'p_t': (new_hyp_score - prev_hyp.score).item()}
                # if debug:
                #     action_log_entry.update({k: v[prev_hyp_id] for k, v in log_entry.items()})
                #     if self.use_syntax_token_rnn:
                #         action_log_entry['syntax_token_rnn_h'] = syntax_token_rnn_states[0][prev_hyp_id]

                new_hyp = prev_hyp.clone_and_apply_action(action)
                new_hyp.score = new_hyp_score
                if debug:
                    new_hyp.action_log = list(prev_hyp.action_log) + [action_log_entry]

                if new_hyp.completed:
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

                    if isinstance(action, GenTokenAction) and action.token.value != '</s>':
                        token_rnn_updated_hyp_ids.append(len(live_hyp_ids) - 1)
                        token_rnn_tokens_to_update.append(action.token.value)
                    elif isinstance(action, ApplySubTreeAction):
                        leaf_tokens = [token.value for token in action.tree.descendant_tokens]
                        if leaf_tokens:
                            token_rnn_updated_hyp_ids.append(len(live_hyp_ids) - 1)
                            token_rnn_tokens_to_update.append(leaf_tokens)

            if live_hyp_ids:
                if self.use_syntax_token_rnn:
                    # update syntax token RNN
                    # (new_hyp_num, hidden_size)
                    syntax_token_rnn_states = (syntax_token_rnn_states[0][live_hyp_ids], syntax_token_rnn_states[1][live_hyp_ids])
                    if token_rnn_updated_hyp_ids:
                        token_rnn_h_tm1 = syntax_token_rnn_states[0][token_rnn_updated_hyp_ids]
                        token_rnn_c_tm1 = syntax_token_rnn_states[1][token_rnn_updated_hyp_ids]

                        if any(isinstance(x, list) for x in token_rnn_tokens_to_update):
                            token_rnn_tokens_to_update = [x if isinstance(x, list) else [x] for x in token_rnn_tokens_to_update]
                        token_rnn_h_t, token_rnn_c_t = self.syntax_token_rnn_step(token_rnn_tokens_to_update,
                                                                                  h_tm1=(token_rnn_h_tm1, token_rnn_c_tm1),
                                                                                  change_vectors=exp_change_vector[live_hyp_ids][token_rnn_updated_hyp_ids])
                        # FIXME: performance improvement of advanced indexing
                        syntax_token_rnn_states[0][token_rnn_updated_hyp_ids] = token_rnn_h_t
                        syntax_token_rnn_states[1][token_rnn_updated_hyp_ids] = token_rnn_c_t

                hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                hyp_scores = torch.tensor([hyp.score for hyp in hypotheses], dtype=torch.float, device=self.device)
                t += 1
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses
