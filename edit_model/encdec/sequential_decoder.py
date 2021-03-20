# coding=utf-8
from collections import OrderedDict, namedtuple
from itertools import chain
import re
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from edit_model import nn_utils
from edit_model.encdec.decoder import Decoder
from edit_model.pointer_net import PointerNet

import numpy as np

SequentialHypothesis = namedtuple('SequentialHypothesis', ['code', 'score', 'action_log'])
SequentialHypothesis.__new__.__defaults__ = (None,)


class SequentialDecoder(Decoder):
    """
    given the context encoding [List[Tokens]], the previous code [List[Tokens]],
    and the change encoding vector, decode the updated code [List[Tokens]]
    """

    def __init__(self,
                 token_embed_size, token_encoding_size, change_vector_size, hidden_size,
                 dropout,
                 init_decode_vec_encoder_state_dropout,
                 code_token_embedder,
                 vocab,
                 no_copy=False):
        super(SequentialDecoder, self).__init__()

        self.vocab = vocab
        self.hidden_size = hidden_size
        self.no_copy = no_copy

        self.code_token_embedder = code_token_embedder
        self.decoder_lstm = nn.LSTMCell(token_embed_size + change_vector_size,
                                        hidden_size)
        self.pointer_net = PointerNet(src_encoding_size=token_encoding_size,
                                      query_vec_size=hidden_size)

        self.decoder_cell_init = nn.Linear(token_encoding_size + change_vector_size, hidden_size)

        self.attention_linear = nn.Linear(token_encoding_size, hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(token_encoding_size + hidden_size, hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.code_token_readout = nn.Linear(hidden_size, len(self.vocab), bias=False)

        # switch probability between copy and generation
        self.copy_gen_switch = nn.Linear(hidden_size, 3)

        self.dropout = nn.Dropout(dropout)

        if init_decode_vec_encoder_state_dropout > 0.:
            self.init_decode_vec_encoder_state_dropout = nn.Dropout(init_decode_vec_encoder_state_dropout)

    @property
    def device(self):
        return self.copy_gen_switch.weight.device

    def forward(self, batch_examples, batched_prev_code, batched_context, change_vectors,
                embedding_cache=None, debug=False):
        """
        compute the probability of generating the target code given context,
        previous code and the change vector

        batched_context: (batch_size, ctx_len, encode_size)
        batched_prev_code: (batch_size, code_len, encode_size)
        change_vector: (batch_size, change_vec_size)
        """

        # (batch_size, hidden_size)
        h_tm1 = self.get_init_hidden_state(batched_prev_code, batched_context, change_vectors)

        batch_size = h_tm1[0].size(0)

        # (batch_size, prev_code_len, encode_size)
        prev_code_att_linear = self.attention_linear(batched_prev_code.encoding)

        # (**updated_code_len**, batch_size, embed_size)
        # pad the target code sequence with boundary symbols
        updated_code_list = [['<s>'] + e.updated_data + ['</s>'] for e in batch_examples]
        updated_code_embed = self.code_token_embedder.get_embed_for_token_sequences(updated_code_list)

        att_vecs = []
        att_tm1 = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)

        # assume the updated code is properly padded by <s> and </s>
        for t, y_tm1_embed in list(enumerate(updated_code_embed.split(split_size=1)))[:-1]:
            y_tm1_embed = y_tm1_embed.squeeze(0)

            x = torch.cat([y_tm1_embed, change_vectors], dim=-1)  # No input feeding

            (h_t, cell_t), att_t = self.step(x,
                                             h_tm1,
                                             batched_prev_code.encoding, batched_prev_code.mask,
                                             prev_code_att_linear)

            att_vecs.append(att_t)

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        # compute copy probabilities and generation probabilities

        # (updated_code_len - 1, batch_size, hidden_size)
        att_vecs = torch.stack(att_vecs)

        # (updated_code_len - 1, batch_size, code_vocab_size)
        gen_code_token_log_prob = F.log_softmax(self.code_token_readout(att_vecs), dim=-1)

        # (updated_code_len - 1, batch_size, ctx_len)
        copy_ctx_token_prob = self.pointer_net(batched_context.encoding, batched_context.mask, att_vecs)

        # (updated_code_len - 1, batch_size, ctx_len)
        copy_prev_token_prob = self.pointer_net(batched_prev_code.encoding, batched_prev_code.mask, att_vecs)

        # (updated_code_len - 1, batch_size, [COPY_FROM_PREV, COPY_FROM_CONTEXT, GEN])
        token_copy_gen_switch = F.log_softmax(self.copy_gen_switch(att_vecs), dim=-1)

        # prepare masks, target indices
        tgt_gen_token_idx, tgt_gen_token_mask, \
        tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
        tgt_copy_prev_token_idx_mask, tgt_copy_prev_token_mask = self.get_gen_and_copy_index_and_mask(batch_examples,
                                                                                                      batched_prev_code,
                                                                                                      batched_context)

        # (updated_code_len - 1, batch_size)
        tgt_gen_token_prob = torch.gather(gen_code_token_log_prob, dim=-1,
                                          index=tgt_gen_token_idx.unsqueeze(2)).squeeze(2)
        tgt_gen_selection_prob = token_copy_gen_switch[:, :, 2]
        tgt_gen_token_prob = tgt_gen_token_prob + tgt_gen_selection_prob

        # (updated_code_len - 1, batch_size)
        tgt_copy_ctx_token_prob = (torch.sum(copy_ctx_token_prob * tgt_copy_ctx_token_idx_mask,
                                             dim=-1) + 1.e-15).log()

        tgt_copy_ctx_selection_prob = token_copy_gen_switch[:, :, 1]
        tgt_copy_ctx_token_prob = tgt_copy_ctx_token_prob + tgt_copy_ctx_selection_prob

        tgt_copy_prev_token_prob = (torch.sum(copy_prev_token_prob * tgt_copy_prev_token_idx_mask,
                                              dim=-1) + 1.e-15).log()

        tgt_copy_prev_selection_prob = token_copy_gen_switch[:, :, 0]
        tgt_copy_prev_token_prob = tgt_copy_prev_token_prob + tgt_copy_prev_selection_prob

        tgt_gen_and_copy_token_prob = nn_utils.log_sum_exp(
            torch.stack([tgt_gen_token_prob, tgt_copy_ctx_token_prob, tgt_copy_prev_token_prob], dim=-1),
            mask=torch.stack([tgt_gen_token_mask, tgt_copy_ctx_token_mask, tgt_copy_prev_token_mask], dim=-1))
        tgt_gen_and_copy_token_prob[tgt_gen_and_copy_token_prob == -float('inf')] = 0.

        # (batch_size)
        tgt_token_prob = tgt_gen_and_copy_token_prob.sum(dim=0)

        return {'log_probs': tgt_token_prob}

    def step(self, x, h_tm1, batched_prev_code_encoding, batched_prev_code_mask, prev_code_att_linear):
        """
        a single LSTM decoding step
        """

        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     batched_prev_code_encoding, prev_code_att_linear,
                                                     mask=batched_prev_code_mask)

        att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t

    def get_init_hidden_state(self, batched_prev_code, batched_context, change_vectors):
        last_cell = batched_prev_code.last_cell

        if hasattr(self, 'init_decode_vec_dropout'):
            last_cell = self.init_decode_vec_encoder_state_dropout(last_cell)

        x = torch.cat([last_cell, change_vectors], dim=-1)
        dec_init_cell = self.decoder_cell_init(x)
        dec_init_state = F.tanh(dec_init_cell)

        return dec_init_state, dec_init_cell

    @staticmethod
    def populate_gen_and_copy_index_and_mask(example, vocab, copy_token=True):
        prev_code = example.prev_data
        updated_code = example.updated_data
        context = example.context

        seq_len = len(example.updated_data) + 1

        tgt_gen_token_idx = torch.zeros(seq_len, dtype=torch.long)
        tgt_gen_token_mask = torch.zeros(seq_len, dtype=torch.float)

        tgt_copy_ctx_token_idx_mask = torch.zeros(seq_len, len(context), dtype=torch.float)
        tgt_copy_ctx_token_mask = torch.zeros(seq_len, dtype=torch.float)

        tgt_copy_prev_token_idx_mask = torch.zeros(seq_len, len(prev_code), dtype=torch.float)
        tgt_copy_prev_token_mask = torch.zeros(seq_len, dtype=torch.float)

        for t, tgt_token in enumerate(updated_code):
            if SequentialDecoder._can_only_generate_this_token(tgt_token):
                tgt_gen_token_mask[t] = 1
                tgt_gen_token_idx[t] = vocab[tgt_token]
            else:
                copied = False
                if copy_token:
                    if tgt_token in prev_code:
                        token_pos_list = [pos for pos, token in enumerate(prev_code) if token == tgt_token]
                        tgt_copy_prev_token_idx_mask[t, token_pos_list] = 1
                        tgt_copy_prev_token_mask[t] = 1
                        copied = True
                    if tgt_token in context:
                        token_pos_list = [pos for pos, token in enumerate(context) if token == tgt_token]
                        tgt_copy_ctx_token_idx_mask[t, token_pos_list] = 1
                        tgt_copy_ctx_token_mask[t] = 1
                        copied = True

                if not copied or tgt_token in vocab:
                    # if the token is not copied, we can only generate this token from the vocabulary,
                    # even if it is a <unk>.
                    # otherwise, we can still generate it from the vocabulary
                    tgt_gen_token_mask[t] = 1
                    tgt_gen_token_idx[t] = vocab[tgt_token]

        # add the index for ending </s>
        tgt_gen_token_mask[len(updated_code)] = 1
        tgt_gen_token_idx[len(updated_code)] = vocab['</s>']

        example.tgt_gen_token_idx = tgt_gen_token_idx
        example.tgt_gen_token_mask = tgt_gen_token_mask
        example.tgt_copy_ctx_token_idx_mask = tgt_copy_ctx_token_idx_mask
        example.tgt_copy_ctx_token_mask = tgt_copy_ctx_token_mask
        example.tgt_copy_prev_token_idx_mask = tgt_copy_prev_token_idx_mask
        example.tgt_copy_prev_token_mask = tgt_copy_prev_token_mask

    def get_gen_and_copy_index_and_mask(self, examples, batched_prev_code, batched_context):
        batch_size = len(examples)
        max_seq_len = max([len(e.updated_data) for e in examples]) + 1

        tgt_gen_token_idx = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        tgt_gen_token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float)

        tgt_copy_ctx_token_idx_mask = torch.zeros(batch_size, max_seq_len, batched_context.encoding.size(1), dtype=torch.float)
        tgt_copy_ctx_token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float)

        tgt_copy_prev_token_idx_mask = torch.zeros(batch_size, max_seq_len, batched_prev_code.encoding.size(1), dtype=torch.float)
        tgt_copy_prev_token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float)

        for batch_id, example in enumerate(examples):
            tgt_gen_token_mask[batch_id, :example.tgt_gen_token_mask.size(0)] = example.tgt_gen_token_mask
            tgt_gen_token_idx[batch_id, :example.tgt_gen_token_idx.size(0)] = example.tgt_gen_token_idx

            tgt_copy_prev_token_idx_mask[batch_id, :example.tgt_copy_prev_token_idx_mask.size(0), :example.tgt_copy_prev_token_idx_mask.size(1)] = example.tgt_copy_prev_token_idx_mask
            tgt_copy_prev_token_mask[batch_id, :example.tgt_copy_prev_token_mask.size(0)] = example.tgt_copy_prev_token_mask

            tgt_copy_ctx_token_idx_mask[batch_id, :example.tgt_copy_ctx_token_idx_mask.size(0), :example.tgt_copy_ctx_token_idx_mask.size(1)] = example.tgt_copy_ctx_token_idx_mask
            tgt_copy_ctx_token_mask[batch_id, :example.tgt_copy_ctx_token_mask.size(0)] = example.tgt_copy_ctx_token_mask

        return tgt_gen_token_idx.permute(1, 0).to(self.device), tgt_gen_token_mask.permute(1, 0).to(self.device), \
               tgt_copy_ctx_token_idx_mask.permute(1, 0, 2).to(self.device), tgt_copy_ctx_token_mask.permute(1, 0).to(self.device), \
               tgt_copy_prev_token_idx_mask.permute(1, 0, 2).to(self.device), tgt_copy_prev_token_mask.permute(1, 0).to(self.device)

    def beam_search_with_source_encodings(self, prev_code, prev_code_encoding, context, context_encoding, change_vector,
                                          beam_size=5, max_decoding_time_step=70, debug=False):
        dec_init_vec = self.get_init_hidden_state(prev_code_encoding, context_encoding, change_vector)

        aggregated_prev_code_tokens = OrderedDict()
        for token_pos, token in enumerate(prev_code):
            aggregated_prev_code_tokens.setdefault(token, []).append(token_pos)

        aggregated_context_tokens = OrderedDict()
        for token_pos, token in enumerate(context):
            aggregated_context_tokens.setdefault(token, []).append(token_pos)

        # (1, prev_code_len, encode_size)
        prev_code_att_linear = self.attention_linear(prev_code_encoding.encoding)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)

        t = 0
        hypotheses = [['<s>']]
        action_logs = [[]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = prev_code_encoding.encoding.expand(hyp_num, prev_code_encoding.encoding.size(1),
                                                                   prev_code_encoding.encoding.size(2))
            exp_src_encodings_att_linear = prev_code_att_linear.expand(hyp_num, prev_code_att_linear.size(1),
                                                                       prev_code_att_linear.size(2))
            # (hyp_num, change_vec_size)
            exp_change_vector = change_vector.expand(hyp_num, change_vector.size(1))

            y_tm1 = torch.tensor([self.vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_tm1_embed = self.code_token_embedder(y_tm1)

            x = torch.cat([y_tm1_embed, exp_change_vector], dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1,
                                             exp_src_encodings, batched_prev_code_mask=None,
                                             prev_code_att_linear=exp_src_encodings_att_linear)

            # (batch_size, code_vocab_size)
            gen_terminal_token_prob = F.softmax(self.code_token_readout(att_t), dim=-1)

            # (batch_size, ctx_len)
            copy_ctx_token_prob = self.pointer_net(context_encoding.encoding, None, att_t.unsqueeze(0)).squeeze(0)

            # (batch_size, ctx_len)
            copy_prev_token_prob = self.pointer_net(prev_code_encoding.encoding, None, att_t.unsqueeze(0)).squeeze(0)

            # (batch_size, [COPY_FROM_PREV, COPY_FROM_CONTEXT, GEN])
            token_copy_gen_switch = F.softmax(self.copy_gen_switch(att_t), dim=-1)

            # (batch_size, code_vocab_size)
            terminal_token_prob = token_copy_gen_switch[:, 2].unsqueeze(1) * gen_terminal_token_prob

            hyp_unk_copy_score_dict: Dict[str, torch.tensor] = OrderedDict()  # Dict[token] = Tensor[hyp_num]
            if self.no_copy is False:
                for token, token_pos_list in aggregated_prev_code_tokens.items():
                    # (hyp_num)
                    sum_copy_prob = copy_prev_token_prob[:, token_pos_list].sum(dim=-1)
                    # (hyp_num)
                    gated_copy_prob = token_copy_gen_switch[:, 0] * sum_copy_prob

                    if token in self.vocab:
                        token_id = self.vocab[token]
                        terminal_token_prob[:, token_id] = terminal_token_prob[:, token_id] + gated_copy_prob
                    else:
                        if token in hyp_unk_copy_score_dict:
                            hyp_unk_copy_score_dict[token] = hyp_unk_copy_score_dict[token] + gated_copy_prob
                        else:
                            hyp_unk_copy_score_dict[token] = gated_copy_prob

                for token, token_pos_list in aggregated_context_tokens.items():
                    # (hyp_num)
                    sum_copy_prob = copy_ctx_token_prob[:, token_pos_list].sum(dim=-1)
                    # (hyp_num)
                    gated_copy_prob = token_copy_gen_switch[:, 1] * sum_copy_prob

                    if token in self.vocab:
                        token_id = self.vocab[token]
                        terminal_token_prob[:, token_id] = terminal_token_prob[:, token_id] + gated_copy_prob
                    else:
                        if token in hyp_unk_copy_score_dict:
                            hyp_unk_copy_score_dict[token] = hyp_unk_copy_score_dict[token] + gated_copy_prob
                        else:
                            hyp_unk_copy_score_dict[token] = gated_copy_prob

            terminal_token_prob = terminal_token_prob.log()
            candidate_hyp_scores = (hyp_scores.unsqueeze(1) + terminal_token_prob).view(-1)
            if len(hyp_unk_copy_score_dict) > 0:
                # (unk_num, hyp_num)
                unk_copy_hyp_scores = torch.cat(
                    [copy_scores.log() + hyp_scores for copy_scores in hyp_unk_copy_score_dict.values()], dim=0)
                # (unk_num * hyp_num)
                unk_copy_hyp_scores = unk_copy_hyp_scores.view(-1)
                candidate_hyp_scores = torch.cat([candidate_hyp_scores, unk_copy_hyp_scores], dim=0)

            top_new_hyp_scores, top_new_hyp_pos = torch.topk(candidate_hyp_scores,
                                                             k=min(candidate_hyp_scores.size(0),
                                                                   beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            new_hypotheses_scores = []
            new_action_logs = []
            id2unk = list(hyp_unk_copy_score_dict.keys())
            vocab_size = terminal_token_prob.size(1)
            vocab_boundary = hyp_num * vocab_size  # hyp_num * vocab_num
            for new_hyp_score, new_hyp_flattened_pos in zip(top_new_hyp_scores, top_new_hyp_pos):
                new_hyp_flattened_pos = new_hyp_flattened_pos.cpu().item()
                new_hyp_score = new_hyp_score.cpu().item()

                if new_hyp_flattened_pos < vocab_boundary:
                    hyp_token_id = new_hyp_flattened_pos % vocab_size
                    tgt_token = self.vocab.id2word[hyp_token_id]
                    prev_hyp_id = new_hyp_flattened_pos // vocab_size
                else:
                    k = new_hyp_flattened_pos - vocab_boundary
                    unk_token_id = k // hyp_num
                    tgt_token = id2unk[unk_token_id]
                    prev_hyp_id = k % hyp_num

                if debug:
                    action_log_entry = {'t': t,
                                        'token': tgt_token,
                                        'token_copy_gen_switch': token_copy_gen_switch[prev_hyp_id,
                                                                 :].log().cpu().numpy(),
                                        'in_vocab': tgt_token in self.vocab,
                                        'tgt_gen_token_prob': gen_terminal_token_prob[prev_hyp_id, self.vocab[
                                            tgt_token]].log().item() if tgt_token in self.vocab else 'n/a',
                                        'tgt_copy_prev_token_prob': copy_prev_token_prob[
                                            prev_hyp_id, aggregated_prev_code_tokens[
                                                tgt_token]].sum().log().item() if tgt_token in aggregated_prev_code_tokens else 'n/a',
                                        'tgt_copy_ctx_token_prob': copy_ctx_token_prob[
                                            prev_hyp_id, aggregated_context_tokens[
                                                tgt_token]].sum().log().item() if tgt_token in aggregated_context_tokens else 'n/a',
                                        'p_t': (new_hyp_score - hyp_scores[prev_hyp_id]).item()
                                        }
                    action_log_list = list(action_logs[prev_hyp_id]) + [action_log_entry]

                if tgt_token == '</s>':
                    hyp_tgt_tokens = hypotheses[prev_hyp_id][1:]
                    completed_hypotheses.append(SequentialHypothesis(hyp_tgt_tokens, new_hyp_score,
                                                                     action_log=action_log_list if debug else None))
                else:
                    hyp_tgt_tokens = hypotheses[prev_hyp_id] + [tgt_token]
                    new_hypotheses.append(hyp_tgt_tokens)

                    live_hyp_ids.append(prev_hyp_id)
                    new_hypotheses_scores.append(new_hyp_score)

                    if debug:
                        new_action_logs.append(action_log_list)

            if len(completed_hypotheses) >= beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hypotheses_scores, dtype=torch.float, device=self.device)
            if debug:
                action_logs = new_action_logs

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses


class SequentialDecoderWithTreeEncoder(SequentialDecoder):
    """
    given the context encoding [List[Tokens]], the previous code [List[Tokens]],
    and the change encoding vector, decode the updated code [List[Tokens]]
    """

    def __init__(self,
                 token_embed_size, token_encoding_size, change_vector_size, hidden_size,
                 dropout,
                 init_decode_vec_encoder_state_dropout,
                 code_token_embedder,
                 vocab,
                 decoder_init_method='avg_pooling'):
        super(SequentialDecoderWithTreeEncoder, self).__init__(token_embed_size, token_encoding_size, change_vector_size, hidden_size,
                                                               dropout,
                                                               init_decode_vec_encoder_state_dropout,
                                                               code_token_embedder,
                                                               vocab)

        self.decoder_init_method = decoder_init_method


    @property
    def device(self):
        return self.copy_gen_switch.weight.device

    def forward(self, batch_examples, batched_prev_code, batched_context, change_vectors,
                embedding_cache=None, debug=False):
        """
        compute the probability of generating the target code given context,
        previous code and the change vector

        batched_context: (batch_size, ctx_len, encode_size)
        batched_prev_code: (batch_size, code_len, encode_size)
        change_vector: (batch_size, change_vec_size)
        """

        # (batch_size, hidden_size)
        h_tm1 = self.get_init_hidden_state(batched_prev_code, batched_context, change_vectors)

        batch_size = h_tm1[0].size(0)

        # (batch_size, prev_code_len, encode_size)
        prev_code_att_linear = self.attention_linear(batched_prev_code.encoding)

        # (**updated_code_len**, batch_size, embed_size)
        # pad the target code sequence with boundary symbols
        updated_code_list = [['<s>'] + e.updated_code_chunk + ['</s>'] for e in batch_examples]
        updated_code_embed = self.code_token_embedder.get_embed_for_token_sequences(updated_code_list)

        att_vecs = []
        att_tm1 = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)

        # assume the updated code is properly padded by <s> and </s>
        for t, y_tm1_embed in list(enumerate(updated_code_embed.split(split_size=1)))[:-1]:
            y_tm1_embed = y_tm1_embed.squeeze(0)

            x = torch.cat([y_tm1_embed, change_vectors], dim=-1)  # No input feeding

            (h_t, cell_t), att_t = self.step(x,
                                             h_tm1,
                                             batched_prev_code.encoding, batched_prev_code.mask,
                                             prev_code_att_linear)

            att_vecs.append(att_t)

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        # compute copy probabilities and generation probabilities

        # (updated_code_len - 1, batch_size, hidden_size)
        att_vecs = torch.stack(att_vecs)

        # (updated_code_len - 1, batch_size, code_vocab_size)
        gen_code_token_log_prob = F.log_softmax(self.code_token_readout(att_vecs), dim=-1)

        # (updated_code_len - 1, batch_size, ctx_len)
        copy_ctx_token_prob = self.pointer_net(batched_context.encoding, batched_context.mask, att_vecs)

        # (updated_code_len - 1, batch_size, ctx_len)
        copy_prev_token_prob = self.pointer_net(batched_prev_code.encoding, batched_prev_code.syntax_token_mask, att_vecs)

        # (updated_code_len - 1, batch_size, [COPY_FROM_PREV, COPY_FROM_CONTEXT, GEN])
        token_copy_gen_switch = F.log_softmax(self.copy_gen_switch(att_vecs), dim=-1)

        # prepare masks, target indices
        tgt_gen_token_idx, tgt_gen_token_mask, \
        tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
        tgt_copy_prev_token_idx_mask, tgt_copy_prev_token_mask = self.get_gen_and_copy_index_and_mask(batch_examples,
                                                                                                      batched_prev_code,
                                                                                                      batched_context)

        # (updated_code_len - 1, batch_size)
        tgt_gen_token_prob = torch.gather(gen_code_token_log_prob, dim=-1,
                                          index=tgt_gen_token_idx.unsqueeze(2)).squeeze(2)
        tgt_gen_selection_prob = token_copy_gen_switch[:, :, 2]
        gated_tgt_gen_token_prob = tgt_gen_token_prob + tgt_gen_selection_prob

        # (updated_code_len - 1, batch_size)
        tgt_copy_ctx_token_prob = (torch.sum(copy_ctx_token_prob * tgt_copy_ctx_token_idx_mask,
                                             dim=-1) + 1.e-15).log()

        tgt_copy_ctx_selection_prob = token_copy_gen_switch[:, :, 1]
        gated_tgt_copy_ctx_token_prob = tgt_copy_ctx_token_prob + tgt_copy_ctx_selection_prob

        tgt_copy_prev_token_prob = (torch.sum(copy_prev_token_prob * tgt_copy_prev_token_idx_mask,
                                              dim=-1) + 1.e-15).log()

        tgt_copy_prev_selection_prob = token_copy_gen_switch[:, :, 0]
        gated_tgt_copy_prev_token_prob = tgt_copy_prev_token_prob + tgt_copy_prev_selection_prob

        tgt_gen_and_copy_token_prob = nn_utils.log_sum_exp(
            torch.stack([gated_tgt_gen_token_prob, gated_tgt_copy_ctx_token_prob, gated_tgt_copy_prev_token_prob], dim=-1),
            mask=torch.stack([tgt_gen_token_mask, tgt_copy_ctx_token_mask, tgt_copy_prev_token_mask], dim=-1))
        tgt_gen_and_copy_token_prob[tgt_gen_and_copy_token_prob == -float('inf')] = 0.

        # (batch_size)
        tgt_token_prob = tgt_gen_and_copy_token_prob.sum(dim=0)

        if debug:
            debug_info = OrderedDict()

            for batch_id, example in enumerate(batch_examples):
                action_trace = []
                log_p = 0.0
                for t in range(len(example.updated_code_chunk) + 1):
                    p_t = tgt_gen_and_copy_token_prob[t, batch_id].item()
                    entry = {'t': t,
                             'token': updated_code_list[batch_id][t + 1],
                             'copy_gen_switch': token_copy_gen_switch[t, batch_id].cpu().numpy(),
                             'tgt_gen_token_prob': tgt_gen_token_prob[t, batch_id].item() if tgt_gen_token_mask[t, batch_id].item() else 'n/a',
                             'tgt_copy_prev_token_prob': tgt_copy_prev_token_prob[t, batch_id].item() if tgt_copy_prev_token_mask[t, batch_id].item() else 'n/a',
                             'tgt_copy_ctx_token_prob': tgt_copy_ctx_token_prob[t, batch_id].item() if tgt_copy_ctx_token_mask[t, batch_id].item() else 'n/a',
                             'p_t': p_t}

                    # entry.update({k: v[batch_id] for k, v in log_entries[t].items()})

                    log_p += p_t
                    action_trace.append(entry)

                debug_info[example.id] = dict(action_trace=action_trace, log_p=log_p)

        if debug:
            return tgt_token_prob, debug_info
        return tgt_token_prob

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

    def get_gen_and_copy_index_and_mask(self, examples, batched_prev_code, batched_context):
        batch_size = len(examples)
        max_seq_len = max([len(e.updated_code_chunk) for e in examples]) + 1

        tgt_gen_token_idx = torch.zeros(max_seq_len, batch_size, dtype=torch.long, device=self.device)
        tgt_gen_token_mask = torch.zeros(max_seq_len, batch_size, dtype=torch.float, device=self.device)

        tgt_copy_ctx_token_idx_mask = torch.zeros(max_seq_len, batch_size, batched_context.encoding.size(1),
                                                  dtype=torch.float, device=self.device)
        tgt_copy_ctx_token_mask = torch.zeros(max_seq_len, batch_size, dtype=torch.float, device=self.device)

        tgt_copy_prev_token_idx_mask = torch.zeros(max_seq_len, batch_size, batched_prev_code.encoding.size(1),
                                                   dtype=torch.float, device=self.device)
        tgt_copy_prev_token_mask = torch.zeros(max_seq_len, batch_size, dtype=torch.float, device=self.device)

        for batch_id in range(batch_size):
            updated_code = examples[batch_id].updated_data
            prev_ast = examples[batch_id].prev_data
            context = batched_context.data[batch_id]
            for t, tgt_token in enumerate(updated_code):
                if SequentialDecoder._can_only_generate_this_token(tgt_token):
                    tgt_gen_token_mask[t, batch_id] = 1
                    tgt_gen_token_idx[t, batch_id] = self.vocab[tgt_token]
                else:
                    copied = False
                    if tgt_token in prev_ast.syntax_token_value2ids:
                        token_pos_list = prev_ast.syntax_token_value2ids[tgt_token]
                        tgt_copy_prev_token_idx_mask[t, batch_id, token_pos_list] = 1
                        tgt_copy_prev_token_mask[t, batch_id] = 1
                        copied = True
                    if tgt_token in context:
                        token_pos_list = [pos for pos, token in enumerate(context) if token == tgt_token]
                        tgt_copy_ctx_token_idx_mask[t, batch_id, token_pos_list] = 1
                        tgt_copy_ctx_token_mask[t, batch_id] = 1
                        copied = True

                    if not copied or tgt_token in self.vocab:
                        # if the token is not copied, we can only generate this token from the vocabulary,
                        # even if it is a <unk>.
                        # otherwise, we can still generate it from the vocabulary
                        tgt_gen_token_mask[t, batch_id] = 1
                        tgt_gen_token_idx[t, batch_id] = self.vocab[tgt_token]

            # add the index for ending </s>
            tgt_gen_token_mask[len(updated_code), batch_id] = 1
            tgt_gen_token_idx[len(updated_code), batch_id] = self.vocab['</s>']

        return tgt_gen_token_idx, tgt_gen_token_mask, \
               tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
               tgt_copy_prev_token_idx_mask, tgt_copy_prev_token_mask

    def beam_search_with_source_encodings(self, prev_code, prev_code_encoding, context, context_encoding, change_vector,
                                          beam_size=5, max_decoding_time_step=70, debug=False, **kwargs):
        dec_init_vec = self.get_init_hidden_state(prev_code_encoding, context_encoding, change_vector)

        aggregated_prev_code_tokens = OrderedDict()
        for token_pos, token in prev_code.syntax_tokens_and_ids:
            aggregated_prev_code_tokens.setdefault(token.value, []).append(token_pos)

        aggregated_context_tokens = OrderedDict()
        for token_pos, token in enumerate(context):
            aggregated_context_tokens.setdefault(token, []).append(token_pos)

        # (1, prev_code_len, encode_size)
        prev_code_att_linear = self.attention_linear(prev_code_encoding.encoding)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)

        t = 0
        hypotheses = [['<s>']]
        action_logs = [[]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = prev_code_encoding.encoding.expand(hyp_num, prev_code_encoding.encoding.size(1),
                                                                   prev_code_encoding.encoding.size(2))
            exp_src_encodings_att_linear = prev_code_att_linear.expand(hyp_num, prev_code_att_linear.size(1),
                                                                       prev_code_att_linear.size(2))
            # (hyp_num, change_vec_size)
            exp_change_vector = change_vector.expand(hyp_num, change_vector.size(1))

            y_tm1 = torch.tensor([self.vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_tm1_embed = self.code_token_embedder(y_tm1)

            x = torch.cat([y_tm1_embed, exp_change_vector], dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1,
                                             exp_src_encodings, batched_prev_code_mask=None,
                                             prev_code_att_linear=exp_src_encodings_att_linear)

            # (batch_size, code_vocab_size)
            gen_terminal_token_prob = F.softmax(self.code_token_readout(att_t), dim=-1)

            # (batch_size, ctx_len)
            copy_ctx_token_prob = self.pointer_net(context_encoding.encoding, None, att_t.unsqueeze(0)).squeeze(0)

            # (batch_size, ctx_len)
            copy_prev_token_prob = self.pointer_net(prev_code_encoding.encoding, prev_code_encoding.syntax_token_mask, att_t.unsqueeze(0)).squeeze(0)

            # (batch_size, [COPY_FROM_PREV, COPY_FROM_CONTEXT, GEN])
            token_copy_gen_switch = F.softmax(self.copy_gen_switch(att_t), dim=-1)

            # (batch_size, code_vocab_size)
            terminal_token_prob = token_copy_gen_switch[:, 2].unsqueeze(1) * gen_terminal_token_prob

            hyp_unk_copy_score_dict: Dict[str, torch.tensor] = OrderedDict()  # Dict[token] = Tensor[hyp_num]
            for token, token_pos_list in aggregated_prev_code_tokens.items():
                # (hyp_num)
                sum_copy_prob = copy_prev_token_prob[:, token_pos_list].sum(dim=-1)
                # (hyp_num)
                gated_copy_prob = token_copy_gen_switch[:, 0] * sum_copy_prob

                if token in self.vocab:
                    token_id = self.vocab[token]
                    terminal_token_prob[:, token_id] = terminal_token_prob[:, token_id] + gated_copy_prob
                else:
                    if token in hyp_unk_copy_score_dict:
                        hyp_unk_copy_score_dict[token] = hyp_unk_copy_score_dict[token] + gated_copy_prob
                    else:
                        hyp_unk_copy_score_dict[token] = gated_copy_prob

            for token, token_pos_list in aggregated_context_tokens.items():
                # (hyp_num)
                sum_copy_prob = copy_ctx_token_prob[:, token_pos_list].sum(dim=-1)
                # (hyp_num)
                gated_copy_prob = token_copy_gen_switch[:, 1] * sum_copy_prob

                if token in self.vocab:
                    token_id = self.vocab[token]
                    terminal_token_prob[:, token_id] = terminal_token_prob[:, token_id] + gated_copy_prob
                else:
                    if token in hyp_unk_copy_score_dict:
                        hyp_unk_copy_score_dict[token] = hyp_unk_copy_score_dict[token] + gated_copy_prob
                    else:
                        hyp_unk_copy_score_dict[token] = gated_copy_prob

            terminal_token_prob = terminal_token_prob.log()
            candidate_hyp_scores = (hyp_scores.unsqueeze(1) + terminal_token_prob).view(-1)
            if len(hyp_unk_copy_score_dict) > 0:
                # (unk_num, hyp_num)
                unk_copy_hyp_scores = torch.cat(
                    [copy_scores.log() + hyp_scores for copy_scores in hyp_unk_copy_score_dict.values()], dim=0)
                # (unk_num * hyp_num)
                unk_copy_hyp_scores = unk_copy_hyp_scores.view(-1)
                candidate_hyp_scores = torch.cat([candidate_hyp_scores, unk_copy_hyp_scores], dim=0)

            top_new_hyp_scores, top_new_hyp_pos = torch.topk(candidate_hyp_scores,
                                                             k=min(candidate_hyp_scores.size(0),
                                                                   beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            new_hypotheses_scores = []
            new_action_logs = []
            id2unk = list(hyp_unk_copy_score_dict.keys())
            vocab_size = terminal_token_prob.size(1)
            vocab_boundary = hyp_num * vocab_size  # hyp_num * vocab_num
            for new_hyp_score, new_hyp_flattened_pos in zip(top_new_hyp_scores, top_new_hyp_pos):
                new_hyp_flattened_pos = new_hyp_flattened_pos.cpu().item()
                new_hyp_score = new_hyp_score.cpu().item()

                if new_hyp_flattened_pos < vocab_boundary:
                    hyp_token_id = new_hyp_flattened_pos % vocab_size
                    tgt_token = self.vocab.id2word[hyp_token_id]
                    prev_hyp_id = new_hyp_flattened_pos // vocab_size
                else:
                    k = new_hyp_flattened_pos - vocab_boundary
                    unk_token_id = k // hyp_num
                    tgt_token = id2unk[unk_token_id]
                    prev_hyp_id = k % hyp_num

                if debug:
                    action_log_entry = {'t': t,
                                        'token': tgt_token,
                                        'token_copy_gen_switch': token_copy_gen_switch[prev_hyp_id,
                                                                 :].log().cpu().numpy(),
                                        'in_vocab': tgt_token in self.vocab,
                                        'tgt_gen_token_prob': gen_terminal_token_prob[prev_hyp_id, self.vocab[
                                            tgt_token]].log().item() if tgt_token in self.vocab else 'n/a',
                                        'tgt_copy_prev_token_prob': copy_prev_token_prob[
                                            prev_hyp_id, aggregated_prev_code_tokens[
                                                tgt_token]].sum().log().item() if tgt_token in aggregated_prev_code_tokens else 'n/a',
                                        'tgt_copy_ctx_token_prob': copy_ctx_token_prob[
                                            prev_hyp_id, aggregated_context_tokens[
                                                tgt_token]].sum().log().item() if tgt_token in aggregated_context_tokens else 'n/a',
                                        'p_t': (new_hyp_score - hyp_scores[prev_hyp_id]).item()
                                        }
                    action_log_list = list(action_logs[prev_hyp_id]) + [action_log_entry]

                if tgt_token == '</s>':
                    hyp_tgt_tokens = hypotheses[prev_hyp_id][1:]
                    completed_hypotheses.append(SequentialHypothesis(hyp_tgt_tokens, new_hyp_score,
                                                                     action_log=action_log_list if debug else None))
                else:
                    hyp_tgt_tokens = hypotheses[prev_hyp_id] + [tgt_token]
                    new_hypotheses.append(hyp_tgt_tokens)

                    live_hyp_ids.append(prev_hyp_id)
                    new_hypotheses_scores.append(new_hyp_score)

                    if debug:
                        new_action_logs.append(action_log_list)

            if len(completed_hypotheses) >= beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hypotheses_scores, dtype=torch.float, device=self.device)
            if debug:
                action_logs = new_action_logs

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses
