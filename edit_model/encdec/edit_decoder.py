# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, OrderedDict

from edit_model.encdec.decoder import Decoder
from edit_model.pointer_net import PointerNet
from edit_model import nn_utils
from trees.substitution_system import Delete, Add, AddSubtree, Stop, ApplyRuleAction, GenTokenAction
from trees.hypothesis import Hypothesis
from trees.utils import find_by_id, get_sibling_ids
from asdl.asdl_ast import SyntaxToken, AbstractSyntaxTree


class IterativeDecoder(Decoder):
    def __init__(self, global_state_hidden_size, source_element_encoding_size, hidden_size,
                 operator_embed_size, action_embed_size, field_embed_size,
                 dropout, vocab, grammar, **kwargs):
        super(IterativeDecoder, self).__init__()

        self.vocab = vocab
        self.grammar = grammar

        self.copy_syntax_token = kwargs.pop('copy_syntax_token', True)
        self.copy_subtree = kwargs.pop('copy_sub_tree', True)
        self.local_feed_anchor_node = kwargs.pop('local_feed_anchor_node', False)
        self.local_feed_siblings = kwargs.pop('local_feed_siblings', False)
        self.local_feed_parent_node = kwargs.pop('local_feed_parent_node', False)
        assert self.local_feed_anchor_node or self.local_feed_siblings

        if self.copy_subtree:
            self.operators = ['delete', 'add', 'add_subtree', 'stop']
        else:
            self.operators = ['delete', 'add', 'stop']

        self.operator_embedding = nn.Embedding(len(self.operators), operator_embed_size)
        self.global_to_op_emb = nn.Linear(global_state_hidden_size, operator_embed_size, bias=True)
        self.operator_readout = lambda h_g: torch.matmul(self.global_to_op_emb(h_g),
                                                         self.operator_embedding.weight.transpose(1, 0))

        # h_g_op: (batch_size, global_state_hidden_size + operator_embed_size)
        # h_node: (batch_size, max_node_num, source_element_encoding_size)
        # readout output: (batch_size, max_node_num)
        self.global_op_to_node_emb = nn.Linear(global_state_hidden_size + operator_embed_size,
                                               source_element_encoding_size, bias=True)
        self.node_readout = lambda h_g_op, h_node: torch.matmul(h_node, F.tanh(self.global_op_to_node_emb(h_g_op)).
                                                                unsqueeze(-1)).squeeze(-1)

        local_proj_input_dim = global_state_hidden_size + field_embed_size
        if self.local_feed_anchor_node:
            local_proj_input_dim += source_element_encoding_size
        if self.local_feed_siblings:
            local_proj_input_dim += source_element_encoding_size * 2
        if self.local_feed_parent_node:
            local_proj_input_dim += source_element_encoding_size
        self.local_proj = nn.Linear(local_proj_input_dim, hidden_size, bias=True)
        self.local_encoder = lambda h_comb: F.tanh(self.local_proj(h_comb))

        self.syntax_token_copy_ptr_net = PointerNet(src_encoding_size=source_element_encoding_size,
                                                    query_vec_size=hidden_size)

        # switch probability between copy and generation
        self.copy_gen_switch = nn.Linear(hidden_size, 3)

        if self.copy_subtree:
            self.sub_tree_copy_ptr_net = PointerNet(src_encoding_size=source_element_encoding_size,
                                                    query_vec_size=hidden_size)

        self.dropout = nn.Dropout(dropout)

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

    def forward(self, batch_examples, batch_edits_list, global_hidden_states_list, context_encodings,
                init_input_encodings, memory_encodings, cur_input_encodings_list, masks_cache, train_components=None):
        """
        Perform a sequence of edit prediction.
        :param batch_examples: a batch of examples.
        :param batch_edits_list: a list of edit batches in each time step.
        :param global_hidden_states_list: a list of (batch_size, global_hidden_size).
        :param context_encodings: encoding result of context.
            Encoding of size (batch_size, num_words_in_context, source_element_encoding_size).
        :param init_input_encodings: encoding result of initial code snippets.
            Encoding of size (batch_size, batch_max_node_num_in_init_code, source_element_encoding_size).
        :param memory_encodings: (batch_size, batch_max_memory_size, source_element_encoding_size) or None,
            encoding vectors of subtrees in memories.
        :param cur_input_encodings_list: a list of encoding results of code snippets.
            Encoding of size (batch_size, batch_max_node_num_in_cur_code, source_element_encoding_size).
        :param masks_cache: masks used for training.
        :param train_components: components to train, choice in ('op', 'node', 'add', 'add_subtree').
            Only components in training have log probs calculated.
        :return:
        """
        if train_components is None:
            train_components = ('op', 'node', 'add', 'add_subtree')

        max_iteration_step = len(batch_edits_list)
        batch_size = len(batch_examples)
        batch_max_node_num_over_time = max(cur_input_encodings.encoding.size(1)
                                           for cur_input_encodings in cur_input_encodings_list)

        returns = {'log_probs': torch.zeros(max_iteration_step, len(batch_examples), dtype=torch.float).to(self.device)}

        # collect gold index and mask
        operator_selection_idx, \
        node_selection_idx, node_selection_mask, node_cand_mask, parent_field_idx, \
        tgt_apply_rule_idx, tgt_apply_rule_mask, apply_rule_cand_mask, \
        tgt_apply_subtree_idx, tgt_apply_subtree_idx_mask, tgt_apply_subtree_mask, apply_subtree_cand_mask, \
        tgt_gen_token_idx, tgt_gen_token_mask, tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
        tgt_copy_init_token_idx_mask, tgt_copy_init_token_mask = masks_cache
        # self.get_gen_and_copy_index_and_mask_over_time(
        #     batch_examples, batch_edits_list, context_encodings,
        #     init_input_encodings, memory_encodings, batch_max_node_num_over_time)

        # (max_iteration_step, batch_size, global_hidden_size)
        global_hidden_states = torch.stack(global_hidden_states_list, dim=0)

        # (max_iteration_step, batch_size, batch_max_node_num_over_time, source_element_encoding_size)
        cur_input_encodings_encoding_over_time = torch.zeros(
            max_iteration_step, batch_size, batch_max_node_num_over_time, cur_input_encodings_list[0].encoding.size(2)).to(self.device)
        for t, cur_input_encodings in enumerate(cur_input_encodings_list):
            cur_input_encodings_encoding_over_time[t, :, :cur_input_encodings.encoding.size(1)] = cur_input_encodings.encoding

        # operator selection: (max_iteration_step, batch_size, len(self.operator))
        op_logits = self.operator_readout(global_hidden_states)
        op_log_probs = F.log_softmax(op_logits, dim=-1)
        tgt_op_log_probs = torch.gather(op_log_probs, dim=-1, index=operator_selection_idx.unsqueeze(-1)).squeeze(-1)
        returns['tgt_op_log_probs'] = tgt_op_log_probs # (max_iteration_step, batch_size)
        returns['log_probs'] += returns['tgt_op_log_probs']

        # node selection: (max_iteration_step, batch_size, batch_max_node_num_in_cur_code)
        if 'node' not in train_components or node_selection_mask.eq(0).all():  # all Stop edits
            return returns

        op_embeds = self.operator_embedding(operator_selection_idx)
        node_logits = self.node_readout(torch.cat([global_hidden_states, op_embeds], dim=-1),
                                        cur_input_encodings_encoding_over_time)
        # apply mask to keep only valid nodes given the gold operators as candidates
        # (max_iteration_step, batch_size, batch_max_node_num_in_cur_code)
        node_cand_mask[node_cand_mask.sum(dim=-1).eq(0), :] = 1.  # safeguard
        node_log_probs = F.log_softmax(node_logits + (node_cand_mask + 1e-45).log(), dim=-1)
        tgt_node_log_probs = torch.gather(node_log_probs, dim=-1, index=node_selection_idx.unsqueeze(-1)).squeeze(-1)
        returns['tgt_node_log_probs'] = tgt_node_log_probs * node_selection_mask
        returns['node_selection_mask'] = node_selection_mask
        returns['log_probs'] += returns['tgt_node_log_probs']

        # for Add and AddSubtree
        if 'add' not in train_components:
            return returns

        tgt_add_operator_mask = operator_selection_idx.eq(self.operators.index("add"))
        if not (tgt_add_operator_mask.any() or tgt_apply_subtree_mask.eq(1).any()):  # all Delete or Stop edits
            return returns

        # encode local hidden states (\tilde{h}_t)
        parent_mask, parent_field_idx, left_sibling_mask, right_sibling_mask = self.get_surrounding_node_index_over_time(
            batch_edits_list, batch_max_node_num_over_time)
        parent_field_encodings = self.field_embedding(parent_field_idx)

        local_feed_inputs = [global_hidden_states, parent_field_encodings]
        if self.local_feed_parent_node:
            parent_encodings = torch.zeros(cur_input_encodings_encoding_over_time.shape, dtype=torch.float).to(self.device)
            parent_encodings[parent_mask.sum(dim=-1).ne(0)] = cur_input_encodings_encoding_over_time[parent_mask.sum(dim=-1).ne(0)]
            parent_encodings = parent_encodings.sum(dim=2)
            local_feed_inputs.append(parent_encodings)
        if self.local_feed_anchor_node:
            self_mask = torch.zeros(max_iteration_step * batch_size, batch_max_node_num_over_time, dtype=torch.long).to(self.device)
            self_mask[torch.arange(0, max_iteration_step * batch_size, dtype=torch.long).to(self.device),
                      node_selection_idx.view(-1)] = 1
            self_mask = self_mask.reshape(max_iteration_step, batch_size, batch_max_node_num_over_time)
            self_encodings = torch.sum(cur_input_encodings_encoding_over_time * self_mask.unsqueeze(-1), dim=2)
            local_feed_inputs.append(self_encodings)
        if self.local_feed_siblings:
            left_sibling_encodings = torch.zeros(cur_input_encodings_encoding_over_time.shape, dtype=torch.float).to(self.device)
            left_sibling_encodings[left_sibling_mask.sum(dim=-1).ne(0)] = cur_input_encodings_encoding_over_time[left_sibling_mask.sum(dim=-1).ne(0)]
            left_sibling_encodings = left_sibling_encodings.sum(dim=2) / (torch.sum(left_sibling_mask, dim=-1, keepdim=True) + 1e-45)
            right_sibling_encodings = torch.zeros(cur_input_encodings_encoding_over_time.shape, dtype=torch.float).to(self.device)
            right_sibling_encodings[right_sibling_mask.sum(dim=-1).ne(0)] = cur_input_encodings_encoding_over_time[right_sibling_mask.sum(dim=-1).ne(0)]
            right_sibling_encodings = right_sibling_encodings.sum(dim=2) / (torch.sum(right_sibling_mask, dim=-1, keepdim=True) + 1e-45)
            local_feed_inputs.append(left_sibling_encodings)
            local_feed_inputs.append(right_sibling_encodings)
        # (max_iteration_step, batch_size, hidden_size)
        local_hidden_states = self.local_encoder(torch.cat(local_feed_inputs, dim=-1))
        local_hidden_states = self.dropout(local_hidden_states)

        if tgt_add_operator_mask.any():
            # for Add edit operator, ApplyRule action
            if tgt_apply_rule_mask.eq(1).any():
                # (max_iteration_step, batch_size, grammar_size + 1)
                apply_rule_logits = self.production_readout(local_hidden_states)
                apply_rule_log_probs = F.log_softmax(apply_rule_logits + (apply_rule_cand_mask + 1e-45).log(), dim=-1)
                tgt_apply_rule_log_probs = torch.gather(apply_rule_log_probs, dim=-1,
                                                        index=tgt_apply_rule_idx.unsqueeze(-1)).squeeze(-1)
                tgt_apply_rule_log_probs = tgt_apply_rule_log_probs * tgt_apply_rule_mask
                returns['tgt_apply_rule_log_probs'] = tgt_apply_rule_log_probs
            else:
                tgt_apply_rule_log_probs = 0.

            # for Add edit operator, GenToken action
            if tgt_gen_token_mask.eq(1).any() or tgt_copy_ctx_token_mask.eq(1).any() or tgt_copy_init_token_mask.eq(1).any():
                # (max_iteration_step, batch_size, terminal_vocab_size)
                gen_terminal_token_log_prob = F.log_softmax(self.terminal_token_readout(local_hidden_states), dim=-1)

                # (max_iteration_step, batch_size, ctx_len)
                copy_ctx_token_prob = self.syntax_token_copy_ptr_net(context_encodings.encoding,
                                                                     context_encodings.mask,
                                                                     local_hidden_states)

                # (max_iteration_step, batch_size, ctx_len)
                copy_init_token_prob = self.syntax_token_copy_ptr_net(init_input_encodings.encoding,
                                                                      init_input_encodings.syntax_token_mask,
                                                                      local_hidden_states)

                # (max_iteration_step, batch_size, [COPY_FROM_PREV, COPY_FROM_CONTEXT, GEN])
                token_copy_gen_switch = F.log_softmax(self.copy_gen_switch(local_hidden_states), dim=-1)

                tgt_gen_token_log_prob = torch.gather(gen_terminal_token_log_prob, dim=-1,
                                                      index=tgt_gen_token_idx.unsqueeze(-1)).squeeze(-1)
                tgt_gen_selection_log_prob = token_copy_gen_switch[:, :, 2]
                gated_tgt_gen_token_log_prob = tgt_gen_token_log_prob + tgt_gen_selection_log_prob

                tgt_copy_ctx_token_log_prob = (
                            torch.sum(copy_ctx_token_prob * tgt_copy_ctx_token_idx_mask, dim=-1) + 1.e-15).log()
                tgt_copy_ctx_selection_log_prob = token_copy_gen_switch[:, :, 1]
                gated_tgt_copy_ctx_token_log_prob = tgt_copy_ctx_token_log_prob + tgt_copy_ctx_selection_log_prob

                tgt_copy_init_token_log_prob = (
                            torch.sum(copy_init_token_prob * tgt_copy_init_token_idx_mask, dim=-1) + 1.e-15).log()
                tgt_copy_init_selection_log_prob = token_copy_gen_switch[:, :, 0]
                gated_tgt_copy_init_token_log_prob = tgt_copy_init_token_log_prob + tgt_copy_init_selection_log_prob

                tgt_gen_and_copy_token_log_probs = nn_utils.log_sum_exp(
                    torch.stack([gated_tgt_gen_token_log_prob, gated_tgt_copy_ctx_token_log_prob,
                                 gated_tgt_copy_init_token_log_prob], dim=-1),
                    mask=torch.stack([tgt_gen_token_mask, tgt_copy_ctx_token_mask, tgt_copy_init_token_mask], dim=-1))
                tgt_gen_and_copy_token_log_probs[tgt_gen_and_copy_token_log_probs == -float('inf')] = 0.
                returns['tgt_gen_and_copy_token_log_probs'] = tgt_gen_and_copy_token_log_probs
            else:
                tgt_gen_and_copy_token_log_probs = 0.

            # (max_iteration_step, batch_size)
            tgt_add_log_probs = tgt_apply_rule_log_probs + tgt_gen_and_copy_token_log_probs
            returns['tgt_add_log_probs'] = tgt_add_log_probs * tgt_add_operator_mask
            returns['tgt_add_operator_mask'] = tgt_add_operator_mask
            returns['log_probs'] += returns['tgt_add_log_probs']

        if not self.copy_subtree or 'add_subtree' not in train_components:
            return returns

        # for AddSubtree edit operator
        if self.copy_subtree and tgt_apply_subtree_mask.eq(1).any():
            assert memory_encodings is not None

            # (max_iteration_step, batch_size, batch_max_memory_size)
            apply_subtree_logits = self.sub_tree_copy_ptr_net(memory_encodings, None,
                                                              local_hidden_states,
                                                              return_logits=True)
            apply_subtree_cand_mask[apply_subtree_cand_mask.sum(dim=-1).eq(0), :] = 1.  # safeguard mask
            apply_subtree_log_probs = F.log_softmax(apply_subtree_logits + (apply_subtree_cand_mask + 1e-45).log(), dim=-1)
            # (max_iteration_step, batch_size, batch_max_cand_subtree_node_num)
            tgt_apply_subtree_log_probs = torch.gather(apply_subtree_log_probs, dim=-1, index=tgt_apply_subtree_idx)

            # marginalize probs since there could be multiple "gold" candidates to copy
            # tgt_apply_subtree_idx_mask_invalid_idx = tgt_apply_subtree_mask.eq(0)
            # tgt_apply_subtree_idx_mask[tgt_apply_subtree_idx_mask_invalid_idx, 0] = 1.  # safeguard
            # tgt_apply_subtree_log_probs = nn_utils.log_sum_exp(tgt_apply_subtree_log_probs,
            #                                                    mask=tgt_apply_subtree_idx_mask)
            # tgt_apply_subtree_log_probs[tgt_apply_subtree_idx_mask_invalid_idx] = 0.
            tgt_apply_subtree_log_probs = nn_utils.log_sum_exp(tgt_apply_subtree_log_probs,
                                                               mask=tgt_apply_subtree_idx_mask)
            tgt_apply_subtree_log_probs[tgt_apply_subtree_log_probs == -float('inf')] = 0.

            returns['tgt_add_subtree_log_probs'] = tgt_apply_subtree_log_probs * tgt_apply_subtree_mask
            returns['tgt_add_subtree_operator_mask'] = tgt_apply_subtree_mask
            returns['log_probs'] += returns['tgt_add_subtree_log_probs']

        return returns

    def get_surrounding_node_index(self, batch_edits, batch_max_node_num):
        batch_size = len(batch_edits)
        parent_mask = torch.zeros(batch_size, batch_max_node_num, dtype=torch.long)
        parent_field_idx = torch.zeros(batch_size, dtype=torch.long)
        left_sibling_mask = torch.zeros(batch_size, batch_max_node_num, dtype=torch.long)
        right_sibling_mask = torch.zeros(batch_size, batch_max_node_num, dtype=torch.long)

        for example_id, edit in enumerate(batch_edits):
            if isinstance(edit, Stop):
                continue

            parent_node = edit.field.parent_node
            parent_mask[example_id, parent_node.id] = 1

            parent_field = edit.field
            parent_field_idx[example_id] = self.grammar.prod_field2id[(parent_node.production, parent_field.field)]

            left_sibling_ids = edit.meta['left_sibling_ids']
            left_sibling_mask[example_id, left_sibling_ids] = 1

            right_sibling_ids = edit.meta['right_sibling_ids']
            right_sibling_mask[example_id, right_sibling_ids] = 1

        parent_mask = parent_mask.to(self.device)
        parent_field_idx = parent_field_idx.to(self.device)
        left_sibling_mask = left_sibling_mask.to(self.device)
        right_sibling_mask = right_sibling_mask.to(self.device)

        return parent_mask, parent_field_idx, left_sibling_mask, right_sibling_mask

    def get_surrounding_node_index_over_time(self, batch_edits_list, batch_max_node_num_over_time):
        max_iteration_step = len(batch_edits_list)

        parent_mask_over_time, parent_field_idx_over_time, \
        left_sibling_mask_over_time, right_sibling_mask_over_time = [], [], [], []

        for t in range(max_iteration_step):
            parent_mask, parent_field_idx, left_sibling_mask, right_sibling_mask = self.get_surrounding_node_index(
                batch_edits_list[t], batch_max_node_num_over_time)
            parent_mask_over_time.append(parent_mask)
            parent_field_idx_over_time.append(parent_field_idx)
            left_sibling_mask_over_time.append(left_sibling_mask)
            right_sibling_mask_over_time.append(right_sibling_mask)

        return torch.stack(parent_mask_over_time, dim=0), torch.stack(parent_field_idx_over_time, dim=0), \
               torch.stack(left_sibling_mask_over_time, dim=0), torch.stack(right_sibling_mask_over_time, dim=0)

    def one_step_beam_search_with_source_encodings(self, cur_tree_hyp_list, init_code_ast, context,
                                                   global_hidden_states, context_encodings,
                                                   init_input_encodings, memory_encodings, cur_input_encodings,
                                                   substitution_system, beam_size, time_step=-1,
                                                   relax_beam_search=False):
        """
        Performing beam search for one-step decoding.
        :param cur_tree_hyp_list: a list of trees.Hypothesis instance, the candidate code snippets at this step.
        :param init_code_ast: AbstractSyntaxTree instance, the initial code snippet.
        :param context: a list of tokens, the context data.
        :param global_hidden_states: (len(cur_tree_hyp_list), global_hidden_size).
        :param context_encodings: encoding result of context.
            Encoding of size (1, num_words_in_context, source_element_encoding_size).
        :param init_input_encodings: encoding result of initial code snippets.
            Encoding of size (1, batch_max_node_num_in_init_code, source_element_encoding_size).
        :param memory_encodings: (len(cur_tree_hyp_list), num_of_subtrees_in_memory, source_element_encoding_size) or None,
            the encoding representation of each subtree in memory for the current hypothesis.
        :param cur_input_encodings: encoding result of each candidate code snippet (tree hypothesis).
            Encoding of size (len(cur_tree_hyp_list), batch_max_node_num_in_cur_code, source_element_encoding_size).
        :param substitution_system: tree.substitution_system.SubstitutionSystem instance.
        :param beam_size: beam size.
        :param time_step: the current time step.
        :param relax_beam_search: relax beams in intermediate decisions (i.e., op/node/add/subtree selection) and apply
            beam constraint at the end of the whole step.
        :return: a list of top ranked trees.Hypothesis instances.
        """
        # data structure for using self.get_surrounding_node_index
        pseudo_edit_tuple = namedtuple('PseudoEdit', ['field', 'meta'])

        # temporarily completed hypotheses to save tree copy time
        pseudo_del_edit = namedtuple('PseudoDelete', ['cur_hyp_idx', 'field', 'value_idx', 'node', 'score', 'score_by_step'])
        pseudo_add_apply_rule_edit = namedtuple('PseudoAdd', ['cur_hyp_idx', 'field', 'value_idx', 'rule_idx', 'score', 'score_by_step'])
        pseudo_add_gen_token_edit = namedtuple('PseudoAdd', ['cur_hyp_idx', 'field', 'value_idx', 'token', 'score', 'score_by_step'])
        pseudo_add_subtree_edit = namedtuple('PseudoAddSubtree', ['cur_hyp_idx', 'field', 'value_idx', 'subtree', 'score', 'score_by_step'])
        pseudo_stop_edit = namedtuple('PseudoStop', ['cur_hyp_idx', 'score', 'score_by_step'])

        def _convert_to_hyp(pseudo_edit_instances):
            tree_hypotheses = []
            for pseudo_edit_instance in pseudo_edit_instances:
                if isinstance(pseudo_edit_instance, pseudo_stop_edit):
                    edit = Stop()
                elif isinstance(pseudo_edit_instance, pseudo_del_edit):
                    edit = Delete(pseudo_edit_instance.field, pseudo_edit_instance.value_idx,
                                  pseudo_edit_instance.node)
                elif isinstance(pseudo_edit_instance, pseudo_add_apply_rule_edit):
                    rule_idx = pseudo_edit_instance.rule_idx
                    action = ApplyRuleAction(self.grammar.id2prod[rule_idx])
                    edit = Add(pseudo_edit_instance.field, pseudo_edit_instance.value_idx, action)
                elif isinstance(pseudo_edit_instance, pseudo_add_gen_token_edit):
                    parent_field = pseudo_edit_instance.field
                    token = pseudo_edit_instance.token
                    action = GenTokenAction(SyntaxToken(parent_field.type, token))
                    edit = Add(parent_field, pseudo_edit_instance.value_idx, action)
                else:
                    assert isinstance(pseudo_edit_instance, pseudo_add_subtree_edit)
                    edit = AddSubtree(pseudo_edit_instance.field, pseudo_edit_instance.value_idx,
                                      pseudo_edit_instance.subtree)

                if beam_size == 1: # just to save time from copying trees
                    assert len(pseudo_edit_instances) <= 1
                    cur_tree_hyp_list[pseudo_edit_instance.cur_hyp_idx].apply_edit(
                        edit, score=pseudo_edit_instance.score)
                    _new_hyp = cur_tree_hyp_list[pseudo_edit_instance.cur_hyp_idx]
                else:
                    _new_hyp = cur_tree_hyp_list[pseudo_edit_instance.cur_hyp_idx].copy_and_apply_edit(
                        edit, score=pseudo_edit_instance.score)
                tree_hypotheses.append(_new_hyp)
                tree_hypotheses[-1].meta.update({'score_by_step_at_%d' % time_step: pseudo_edit_instance.score_by_step})

            return tree_hypotheses

        cur_tree_list = [cur_tree_hyp.tree for cur_tree_hyp in cur_tree_hyp_list]
        completed_hypotheses = []  # pseudo edit instances

        # Round 1: operator selection
        _global_hidden_states = global_hidden_states # (len(cur_tree_hyp_list), global_hidden_size)
        op_logits = self.operator_readout(_global_hidden_states)
        op_log_probs = F.log_softmax(op_logits, dim=-1)  # (len(cur_tree_hyp_list), len(self.operators))
        # (len(cur_tree_hyp_list), K)
        if relax_beam_search:
            top_op_values = op_log_probs
            top_op_indices = torch.arange(op_log_probs.size(1)).expand(op_log_probs.shape)
        else:
            top_op_values, top_op_indices = torch.topk(op_log_probs, dim=-1, k=min(beam_size, op_log_probs.size(1)))

        # initialize hypothesis = [(score, [op_idx], [op_log_prob])]
        hypotheses, hypotheses_idx_map = [], []
        op_selection_idx = []
        for cur_hyp_idx, cur_tree_hyp in enumerate(cur_tree_hyp_list):
            for top_op_val, top_op_idx in zip(top_op_values[cur_hyp_idx], top_op_indices[cur_hyp_idx]):
                op_idx = top_op_idx.item()
                if self.operators[op_idx] == 'stop':  # Stop
                    # completed_hypotheses.append(cur_tree_hyp.copy_and_apply_edit(Stop(), score=top_op_val.item()))
                    completed_hypotheses.append(pseudo_stop_edit(cur_hyp_idx=cur_hyp_idx, score=top_op_val.item(),
                                                                 score_by_step=[top_op_val.item()]))
                else:
                    hypotheses.append((top_op_val.item(), [top_op_idx.item()], [top_op_val.item()]))
                    hypotheses_idx_map.append(cur_hyp_idx)
                    op_selection_idx.append(op_idx)
        if len(hypotheses) == 0:
            top_completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x.score, reverse=True)[:beam_size]
            return _convert_to_hyp(top_completed_hypotheses)
        # going forward, "hypotheses" contains top operators for each cur_tree_hyp
        # len(hypotheses) roughly equals len(cur_tree_hyp_list) * beam_size (over operators)

        # Round 2: node selection
        batch_max_node_num_in_cur_code = cur_input_encodings.encoding.size(1)
        op_selection_idx = torch.tensor(op_selection_idx, dtype=torch.long).to(self.device)
        op_embeds = self.operator_embedding(op_selection_idx)
        _global_hidden_states = torch.stack([global_hidden_states[cur_hyp_idx]
                                            for cur_hyp_idx in hypotheses_idx_map], dim=0)
        _cur_input_encodings_encoding = torch.stack([cur_input_encodings.encoding[cur_hyp_idx]
                                                    for cur_hyp_idx in hypotheses_idx_map], dim=0)

        # (len(hypotheses), batch_max_node_num_in_cur_code)
        node_logits = self.node_readout(torch.cat([_global_hidden_states, op_embeds], dim=1),
                                        _cur_input_encodings_encoding)

        hyp_valid_continuating_node_ids = []
        node_cand_mask = torch.zeros(len(hypotheses), batch_max_node_num_in_cur_code, dtype=torch.long).to(self.device)
        for hyp_idx, (score, partial_decisions, _) in enumerate(hypotheses):
            op_idx = partial_decisions[0]
            operator = self.operators[op_idx]
            cur_hyp_idx = hypotheses_idx_map[hyp_idx]
            valid_continuating_node_ids = substitution_system.get_valid_continuating_node_ids(
                cur_tree_hyp_list[cur_hyp_idx], operator)
            node_cand_mask[hyp_idx, valid_continuating_node_ids] = 1
            hyp_valid_continuating_node_ids.append(valid_continuating_node_ids)

        if node_cand_mask.sum(dim=1).eq(0).all():
            top_completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x.score, reverse=True)[:beam_size]
            return _convert_to_hyp(top_completed_hypotheses)

        node_cand_mask[node_cand_mask.sum(dim=1).eq(0), :] = 1.  # safeguard
        # (len(hypotheses), batch_max_node_num_in_cur_code)
        node_log_probs = F.log_softmax(node_logits + (node_cand_mask + 1e-45).log(), dim=-1)
        # (len(hypotheses), K)
        if relax_beam_search:
            top_node_values, top_node_indices = torch.topk(node_log_probs, dim=-1,
                                                           k=node_cand_mask.sum(dim=1).max())
        else:
            top_node_values, top_node_indices = torch.topk(node_log_probs, dim=-1,
                                                           k=min(beam_size, batch_max_node_num_in_cur_code))

        # "new_hypotheses" roughly the size of len(cur_tree_hyp_list) * beam_size (over op) * beam_size (over node);
        # all of them are grammatically valid.
        new_hypotheses, new_hypotheses_idx_map = [], []
        for hyp_idx, (score, partial_decisions, partial_log_probs) in enumerate(hypotheses):
            if len(hyp_valid_continuating_node_ids[hyp_idx]) == 0:
                continue

            cur_hyp_idx = hypotheses_idx_map[hyp_idx]

            for top_node_val, top_node_idx in zip(top_node_values[hyp_idx][:len(hyp_valid_continuating_node_ids[hyp_idx])],
                                                  top_node_indices[hyp_idx][:len(hyp_valid_continuating_node_ids[hyp_idx])]):
                node_id = top_node_idx.item()
                assert node_id in hyp_valid_continuating_node_ids[hyp_idx]
                new_hyp = (score + top_node_val.item(), partial_decisions + [node_id],
                           partial_log_probs + [top_node_val.item()])
                new_hypotheses.append(new_hyp)
                new_hypotheses_idx_map.append(cur_hyp_idx)

        # aggregate; going forward, "top_hypotheses" contains only top combs of <cur_tree_hyp, operator, node>
        if relax_beam_search:
            top_hypotheses, top_hypotheses_idx_map = new_hypotheses, new_hypotheses_idx_map
        else:
            top_hypotheses_idx_map_pairs = sorted(zip(new_hypotheses, new_hypotheses_idx_map),
                                                  key=lambda x: x[0][0], reverse=True)
            top_hypotheses, top_hypotheses_idx_map = zip(
                *top_hypotheses_idx_map_pairs[:(len(cur_tree_hyp_list) * beam_size - len(completed_hypotheses))])

        # check partial completeness (Delete)
        add_hypotheses, add_subtree_hypotheses, add_hypotheses_idx_map, add_subtree_hypotheses_idx_map = [], [], [], []
        add_pseudo_batch_edits, add_subtree_pseudo_batch_edits = [], []
        hyp_apply_rule_cands, hyp_apply_subtree_cands = [], []
        for hyp_idx, hyp in enumerate(top_hypotheses):
            score, partial_decisions, partial_log_probs = hyp
            cur_hyp_idx = top_hypotheses_idx_map[hyp_idx]
            op_idx = partial_decisions[0]
            node_id = partial_decisions[1]
            node = cur_tree_list[cur_hyp_idx].id2node[node_id]
            parent_field = node.parent_field
            value_idx = find_by_id(parent_field.as_value_list, node)
            if self.operators[op_idx] == 'delete':
                # completed_hypotheses.append(cur_tree_hyp_list[cur_hyp_idx].copy_and_apply_edit(
                #     Delete(parent_field, value_idx, node), score=score))
                completed_hypotheses.append(pseudo_del_edit(cur_hyp_idx=cur_hyp_idx, field=parent_field,
                                                            value_idx=value_idx, node=node, score=score,
                                                            score_by_step=partial_log_probs))
                continue

            cur_field = parent_field
            valid_add_types = substitution_system.get_valid_continuating_add_types(cur_tree_hyp_list[cur_hyp_idx], cur_field)
            if self.operators[op_idx] == 'add':
                # grammar check
                if "add_apply_rule" in valid_add_types:
                    hyp_apply_rule_cands.append(substitution_system.get_valid_continuating_add_production_ids(
                        cur_tree_hyp_list[cur_hyp_idx], cur_field))
                else:
                    assert "add_gen_token" in valid_add_types
                    hyp_apply_rule_cands.append([])

                add_hypotheses.append(hyp)
                add_hypotheses_idx_map.append(cur_hyp_idx)
                left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, node)
                add_pseudo_batch_edits.append(pseudo_edit_tuple(cur_field, {
                    'left_sibling_ids': left_sibling_ids, 'right_sibling_ids': right_sibling_ids}))
            else:
                assert self.operators[op_idx] == 'add_subtree'
                assert "add_subtree" in valid_add_types
                valid_continuating_subtree_ids = substitution_system.get_valid_continuating_add_subtree(
                    cur_tree_hyp_list[cur_hyp_idx], cur_field)
                assert len(valid_continuating_subtree_ids)

                hyp_apply_subtree_cands.append(valid_continuating_subtree_ids)
                add_subtree_hypotheses.append(hyp)
                add_subtree_hypotheses_idx_map.append(cur_hyp_idx)
                left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, node)
                add_subtree_pseudo_batch_edits.append(pseudo_edit_tuple(cur_field, {
                    'left_sibling_ids': left_sibling_ids, 'right_sibling_ids': right_sibling_ids}))

        if len(add_hypotheses) == len(add_subtree_hypotheses) == 0:
            top_completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x.score, reverse=True)[:beam_size]
            return _convert_to_hyp(top_completed_hypotheses)

        # Round 3: Add/AddSubtree
        # encode local hidden states
        hypotheses = add_hypotheses + add_subtree_hypotheses
        hypotheses_idx_map = add_hypotheses_idx_map + add_subtree_hypotheses_idx_map
        pseudo_batch_edits = add_pseudo_batch_edits + add_subtree_pseudo_batch_edits
        node_selection_idx = torch.tensor([_hyp[1][-1] for _hyp in hypotheses], dtype=torch.long).to(self.device)
        parent_mask, parent_field_idx, left_sibling_mask, right_sibling_mask = self.get_surrounding_node_index(
            pseudo_batch_edits, batch_max_node_num_in_cur_code)
        # (len(hypotheses), field_emb_size)
        parent_field_encodings = self.field_embedding(parent_field_idx)

        _global_hidden_states = torch.stack([global_hidden_states[cur_hyp_idx]
                                            for cur_hyp_idx in hypotheses_idx_map], dim=0)
        _cur_input_encodings_encoding = torch.stack([cur_input_encodings.encoding[cur_hyp_idx]
                                                    for cur_hyp_idx in hypotheses_idx_map], dim=0)

        local_feed_inputs = [_global_hidden_states, parent_field_encodings]
        if self.local_feed_parent_node:
            parent_encodings = torch.zeros(_cur_input_encodings_encoding.shape, dtype=torch.float).to(self.device)
            parent_encodings[parent_mask.sum(dim=-1).ne(0)] = _cur_input_encodings_encoding[parent_mask.sum(dim=-1).ne(0)]
            parent_encodings = parent_encodings.sum(dim=1)  # (len(hypotheses), source_element_encoding_size)
            local_feed_inputs.append(parent_encodings)
        if self.local_feed_anchor_node:
            self_mask = torch.zeros(len(hypotheses), batch_max_node_num_in_cur_code, dtype=torch.long).to(self.device)
            self_mask[torch.arange(0, len(hypotheses), dtype=torch.long).to(self.device), node_selection_idx] = 1
            self_encodings = torch.sum(_cur_input_encodings_encoding * self_mask.unsqueeze(2), dim=1)
            local_feed_inputs.append(self_encodings)
        if self.local_feed_siblings:
            left_sibling_encodings = torch.zeros(_cur_input_encodings_encoding.shape, dtype=torch.float).to(self.device)
            left_sibling_encodings[left_sibling_mask.sum(dim=-1).ne(0)] = _cur_input_encodings_encoding[left_sibling_mask.sum(dim=-1).ne(0)]
            left_sibling_encodings = left_sibling_encodings.sum(dim=1) / (torch.sum(left_sibling_mask, dim=-1, keepdim=True) + 1e-45)
            right_sibling_encodings = torch.zeros(_cur_input_encodings_encoding.shape, dtype=torch.float).to(self.device)
            right_sibling_encodings[right_sibling_mask.sum(dim=-1).ne(0)] =  _cur_input_encodings_encoding[right_sibling_mask.sum(dim=-1).ne(0)]
            right_sibling_encodings = right_sibling_encodings.sum(dim=1) / (torch.sum(right_sibling_mask, dim=-1, keepdim=True) + 1e-45)
            local_feed_inputs.append(left_sibling_encodings)
            local_feed_inputs.append(right_sibling_encodings)
        # (len(hypotheses), hidden_size)
        local_hidden_states = self.local_encoder(torch.cat(local_feed_inputs, dim=1))

        # Add operator
        if len(add_hypotheses):
            # ApplyRule action
            apply_rule_mask = torch.tensor([len(_apply_rule_cands) > 0 for _apply_rule_cands in hyp_apply_rule_cands],
                                           dtype=torch.bool).to(self.device)
            if apply_rule_mask.any():
                apply_rule_cand_mask = torch.zeros(len(add_hypotheses), len(self.grammar) + 1, dtype=torch.float).to(self.device)
                for hyp_idx, _apply_rule_cands in enumerate(hyp_apply_rule_cands):
                    apply_rule_cand_mask[hyp_idx, _apply_rule_cands] = 1.

                # (len(add_hypotheses), len(grammar)+1)
                apply_rule_logits = self.production_readout(local_hidden_states[:len(add_hypotheses)])
                apply_rule_log_probs = F.log_softmax(apply_rule_logits + (apply_rule_cand_mask + 1e-45).log(), dim=-1)
                top_rule_values, top_rule_indices = torch.topk(apply_rule_log_probs, dim=-1,
                                                               k=min(len(self.grammar) + 1, beam_size))

            # GenToken
            gen_token_mask = ~apply_rule_mask
            if gen_token_mask.any():
                aggregated_prev_code_tokens = OrderedDict()
                for token_pos, token in init_code_ast.syntax_tokens_and_ids:
                    aggregated_prev_code_tokens.setdefault(token.value, []).append(token_pos)

                aggregated_context_tokens = OrderedDict()
                for token_pos, token in enumerate(context):
                    aggregated_context_tokens.setdefault(token, []).append(token_pos)

                # (len(add_hypotheses), terminal_vocab_size)
                gen_terminal_token_prob = F.softmax(self.terminal_token_readout(
                    local_hidden_states[:len(add_hypotheses)]), dim=-1)

                # (len(add_hypotheses), ctx_len)
                copy_ctx_token_prob = self.syntax_token_copy_ptr_net(
                    context_encodings.encoding.expand(len(add_hypotheses),
                                                      context_encodings.encoding.size(1),
                                                      context_encodings.encoding.size(2)),
                    context_encodings.mask.expand(len(add_hypotheses),
                                                  context_encodings.mask.size(1)),
                    local_hidden_states[:len(add_hypotheses)].unsqueeze(0)).squeeze(0)

                # (len(add_hypotheses), ctx_len)
                copy_init_token_prob = self.syntax_token_copy_ptr_net(
                    init_input_encodings.encoding.expand(len(add_hypotheses),
                                                         init_input_encodings.encoding.size(1),
                                                         init_input_encodings.encoding.size(2)),
                    init_input_encodings.syntax_token_mask.expand(len(add_hypotheses),
                                                                  init_input_encodings.syntax_token_mask.size(1)),
                    local_hidden_states[:len(add_hypotheses)].unsqueeze(0)).squeeze(0)

                # (len(add_hypotheses), [COPY_FROM_PREV, COPY_FROM_CONTEXT, GEN])
                token_copy_gen_switch = F.softmax(self.copy_gen_switch(local_hidden_states[:len(add_hypotheses)]),
                                                  dim=-1)
                terminal_token_prob = token_copy_gen_switch[:, 2].unsqueeze(1) * gen_terminal_token_prob

                hyp_unk_copy_score = OrderedDict()

                if self.copy_syntax_token:
                    for token, token_pos_list in aggregated_prev_code_tokens.items():
                        sum_copy_prob = copy_init_token_prob[:, token_pos_list].sum(dim=-1)
                        gated_copy_prob = token_copy_gen_switch[:, 0] * sum_copy_prob

                        if token in self.vocab:
                            token_id = self.vocab[token]
                            terminal_token_prob[:, token_id] = terminal_token_prob[:, token_id] + gated_copy_prob
                        else:
                            if token in hyp_unk_copy_score:
                                hyp_unk_copy_score[token] = hyp_unk_copy_score[token] + gated_copy_prob
                            else:
                                hyp_unk_copy_score[token] = gated_copy_prob

                    for token, token_pos_list in aggregated_context_tokens.items():
                        sum_copy_prob = copy_ctx_token_prob[:, token_pos_list].sum(dim=-1)
                        gated_copy_prob = token_copy_gen_switch[:, 1] * sum_copy_prob

                        if token in self.vocab:
                            token_id = self.vocab[token]
                            terminal_token_prob[:, token_id] = terminal_token_prob[:, token_id] + gated_copy_prob
                        else:
                            if token in hyp_unk_copy_score:
                                hyp_unk_copy_score[token] = hyp_unk_copy_score[token] + gated_copy_prob
                            else:
                                hyp_unk_copy_score[token] = gated_copy_prob

                if len(hyp_unk_copy_score):
                    unk_tokens = []
                    for token, copy_prob in hyp_unk_copy_score.items():
                        unk_tokens.append(token)
                        terminal_token_prob = torch.cat([terminal_token_prob, copy_prob.unsqueeze(-1)], dim=-1)

                # (len(add_hypotheses), len(self.vocab) + len(hyp_unk_copy_score))
                terminal_token_log_probs = terminal_token_prob.log()
                top_token_values, top_token_indices = torch.topk(terminal_token_log_probs, dim=-1, k=beam_size)

            for hyp_idx, hyp in enumerate(add_hypotheses):
                score, partial_decisions, partial_log_probs = hyp
                cur_hyp_idx = add_hypotheses_idx_map[hyp_idx]
                node_id = partial_decisions[1]
                node = cur_tree_list[cur_hyp_idx].id2node[node_id]
                parent_field = node.parent_field
                value_idx = find_by_id(parent_field.as_value_list, node)
                if len(hyp_apply_rule_cands[hyp_idx]):
                    for top_rule_val, top_rule_idx in zip(top_rule_values[hyp_idx][:len(hyp_apply_rule_cands[hyp_idx])],
                                                          top_rule_indices[hyp_idx][:len(hyp_apply_rule_cands[hyp_idx])]):
                        rule_idx = top_rule_idx.item()
                        assert rule_idx in hyp_apply_rule_cands[hyp_idx]
                        # action = ApplyRuleAction(self.grammar.id2prod[rule_idx])
                        # edit = Add(parent_field, value_idx, action)
                        # completed_hypotheses.append(
                        #     cur_tree_hyp_list[cur_hyp_idx].copy_and_apply_edit(edit, score=score + top_rule_val.item()))
                        completed_hypotheses.append(pseudo_add_apply_rule_edit(
                            cur_hyp_idx=cur_hyp_idx, field=parent_field,
                            value_idx=value_idx, rule_idx=rule_idx, score=score+top_rule_val.item(),
                            score_by_step=partial_log_probs+[top_rule_val.item()]))
                else:
                    for top_token_val, top_token_idx in zip(top_token_values[hyp_idx], top_token_indices[hyp_idx]):
                        token_idx = top_token_idx.item()
                        if token_idx < len(self.vocab):
                            token = self.vocab.id2word[token_idx]
                        else:
                            token = unk_tokens[token_idx - len(self.vocab)]
                        # action = GenTokenAction(SyntaxToken(parent_field.type, token))
                        # edit = Add(parent_field, value_idx, action)
                        # completed_hypotheses.append(
                        #     cur_tree_hyp_list[cur_hyp_idx].copy_and_apply_edit(edit, score=score + top_token_val.item()))
                        completed_hypotheses.append(pseudo_add_gen_token_edit(
                            cur_hyp_idx=cur_hyp_idx, field=parent_field, value_idx=value_idx,
                            token=token, score=score+top_token_val.item(),
                            score_by_step=partial_log_probs+[top_token_val.item()]))

        # AddSubtree operator
        if self.copy_subtree and len(add_subtree_hypotheses):
            assert memory_encodings is not None

            # (len(add_subtree_hypotheses), max_memory_size, source_element_encoding_size)
            add_subtree_hypotheses_memory_encodings = torch.stack([memory_encodings[add_subtree_hypotheses_idx_map[hyp_idx]]
                                                                   for hyp_idx, hyp in enumerate(add_subtree_hypotheses)], dim=0)
            # add_subtree_hypotheses_memory_encodings = init_input_encodings.encoding.expand(
            #   len(add_subtree_hypotheses), -1, -1)

            # (len(add_subtree_hypotheses), node_num_in_init_code)
            apply_subtree_logits = self.sub_tree_copy_ptr_net(
                add_subtree_hypotheses_memory_encodings, None,
                local_hidden_states[-len(add_subtree_hypotheses):].unsqueeze(0),
                return_logits=True).squeeze(0)
            apply_subtree_cand_mask = torch.zeros(len(add_subtree_hypotheses),
                                                  add_subtree_hypotheses_memory_encodings.size(1),
                                                  dtype=torch.float).to(self.device)
            for hyp_idx, _apply_subtree_cands in enumerate(hyp_apply_subtree_cands):
                apply_subtree_cand_mask[hyp_idx, _apply_subtree_cands] = 1.
            # apply_subtree_cand_mask[apply_subtree_cand_mask.sum(dim=-1).eq(0), :] = 1.  # safeguard mask
            apply_subtree_log_probs = F.log_softmax(apply_subtree_logits + (apply_subtree_cand_mask + 1e-45).log(), dim=-1)

            # aggregating same subtrees in memory
            aggregated_subtree_reprs = OrderedDict()
            aggregated_subtrees, aggregated_subtrees_idx = [], []
            for node_idx_in_memory, node in enumerate(cur_tree_hyp_list[0].memory): # same memory for all hyps
                node_repr = node.to_string()  # to save time
                if node_repr not in aggregated_subtree_reprs:
                    # aggregated_subtree_reprs[node_repr] = [node.id]
                    aggregated_subtree_reprs[node_repr] = [node_idx_in_memory]
                    aggregated_subtrees.append(node)
                    aggregated_subtrees_idx.append(node_idx_in_memory)
                else:
                    # aggregated_subtree_reprs[node_repr].append(node.id)
                    aggregated_subtree_reprs[node_repr].append(node_idx_in_memory)

            aggregated_apply_subtree_log_probs = torch.zeros(len(add_subtree_hypotheses), len(aggregated_subtrees),
                                                             dtype=torch.float).to(self.device)
            for node_set_idx, (node_repr, node_ids) in enumerate(aggregated_subtree_reprs.items()):
                aggregated_apply_subtree_log_probs[:, node_set_idx] = nn_utils.log_sum_exp(
                    apply_subtree_log_probs[:, node_ids]) # log(p_tree1 + p_tree2 + ...)

            top_subtree_values, top_subtree_indices = torch.topk(aggregated_apply_subtree_log_probs, dim=-1,
                                                                 k=min(aggregated_apply_subtree_log_probs.size(-1),
                                                                       beam_size))

            for hyp_idx, hyp in enumerate(add_subtree_hypotheses):
                score, partial_decisions, partial_log_probs = hyp
                cur_hyp_idx = add_subtree_hypotheses_idx_map[hyp_idx]
                node_id = partial_decisions[1]
                node = cur_tree_list[cur_hyp_idx].id2node[node_id]
                parent_field = node.parent_field
                value_idx = find_by_id(parent_field.as_value_list, node)

                for top_subtree_val, top_subtree_idx in zip(top_subtree_values[hyp_idx], top_subtree_indices[hyp_idx]):
                    subtree_idx = top_subtree_idx.item()
                    subtree_idx_in_memory = aggregated_subtrees_idx[subtree_idx]

                    if subtree_idx_in_memory in hyp_apply_subtree_cands[hyp_idx]:  # out of scope when some cands are aggregated
                        subtree = aggregated_subtrees[subtree_idx]

                        # edit = AddSubtree(parent_field, value_idx, subtree)
                        # completed_hypotheses.append(
                        #     cur_tree_hyp_list[cur_hyp_idx].copy_and_apply_edit(edit, score=score + top_subtree_val.item()))
                        completed_hypotheses.append(pseudo_add_subtree_edit(cur_hyp_idx=cur_hyp_idx,
                                                                            field=parent_field, value_idx=value_idx,
                                                                            subtree=subtree,
                                                                            score=score+top_subtree_val.item(),
                                                                            score_by_step=partial_log_probs+[top_subtree_val.item()]))

        top_completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x.score, reverse=True)[:beam_size]
        return _convert_to_hyp(top_completed_hypotheses)

    def beam_search_with_source_encodings(self, controller, encoder, init_code_ast, context,
                                          init_global_states, init_token_encodings, context_encodings,
                                          edit_encodings, input_aggregation_fn, substitution_system,
                                          beam_size, max_iteration_step, length_norm=False,
                                          preset_hyp=None, return_ast=False):
        """
        Beam search for one hypothesis.
        :param controller: controller to encode the edit history.
        :param encoder: encoder to encode code snippets in each time step.
        :param init_code_ast: AbstractSyntaxTree, the initial code snippet.
        :param context: a list of tokens, the context data.
        :param init_global_states: a tuple of the initial (global_hidden_states, global_cell_states).
            Each of size (1, global_hidden_size).
        :param init_token_encodings: encoding result of initial code snippet tokens.
        :param context_encodings: encoding result of context.
            Encoding of size (1, num_words_in_context, source_element_encoding_size).
        :param edit_encodings: (1, edit_encoding_size).
        :param input_aggregation_fn: a function for aggregating the current input encodings.
        :param substitution_system: tree.substitution_system.SubstitutionSystem instance.
        :param beam_size: beam size.
        :param max_iteration_step: maximum number of edit steps.
        :param length_norm: whether to apply length normalization in beam search.
        :param preset_hyp: given hypothesis to continue decoding.
        :param return_ast: whether to return hyp.tree as an AST object or a ASNode object. Default False.
        :return: a list of top ranked trees.Hypothesis instances.
        """

        # initial setup
        # init_code_ast = init_code_ast.copy_and_reindex_w_dummy_reduce()
        # note that the passed ast should have been reindexed
        assert hasattr(init_code_ast, 'dummy_node_ids') and init_code_ast.dummy_node_ids is not None
        init_input_encodings = encoder([init_code_ast], init_token_encodings.encoding)

        if preset_hyp is None:
            hypotheses = [Hypothesis(init_code_ast, bool_copy_subtree=self.copy_subtree,
                                     init_code_tokens=init_token_encodings.data[0],
                                     length_norm=length_norm)]
            hypotheses[0].meta = {'global_states': (init_global_states[0][0], init_global_states[1][0])}

            # memory setup: initial_memory_encodings of (1, memory_size, hidden_size)
            initial_full_memory_encodings = init_input_encodings.encoding
            valid_subtree_idx = [subtree.id for subtree in hypotheses[0].memory]
            initial_memory_encodings = initial_full_memory_encodings[:1, valid_subtree_idx, :]
            hypotheses[0].meta['memory_encodings'] = initial_memory_encodings
        else:
            hypotheses = [preset_hyp]

        completed_hypotheses = []

        for t in range(max_iteration_step):
            new_hypotheses = []

            # encode current input encodings
            # (len(hypotheses), batch_max_code_num, source_element_encoding_size)
            cur_input_encodings = encoder([hyp.tree for hyp in hypotheses],
                                          init_token_encodings.encoding.expand(len(hypotheses), -1, -1))

            # memory encodings
            memory_sizes = [len(hyp.memory) for hyp in hypotheses]
            max_memory_size = max(memory_sizes)
            if max_memory_size == 0:
                memory_encodings = None
            else:
                _valid_hyp_idx = memory_sizes.index(max_memory_size)
                memory_encoding_size = hypotheses[_valid_hyp_idx].meta['memory_encodings'].size(-1)
                memory_encodings = []
                for hyp_idx, hyp in enumerate(hypotheses):
                    if memory_sizes[hyp_idx] == 0:
                        memory_encodings.append(torch.zeros(1, max_memory_size, memory_encoding_size).to(self.device))
                    elif memory_sizes[hyp_idx] < max_memory_size:
                        memory_encodings.append(torch.cat(
                            [hyp.meta['memory_encodings'],
                             torch.zeros(1, max_memory_size - memory_sizes[hyp_idx], memory_encoding_size).to(self.device)],
                            dim=1))
                    else:
                        memory_encodings.append(hyp.meta['memory_encodings'])
                memory_encodings = torch.cat(memory_encodings, dim=0) # (len(hypotheses), max_memory_size, -1)

            # update global hidden states
            # (len(hypotheses), global_hidden_size)
            last_global_hidden_states = torch.stack([hyp.meta['global_states'][0] for hyp in hypotheses], dim=0)
            last_global_cell_states = torch.stack([hyp.meta['global_states'][1] for hyp in hypotheses], dim=0)

            global_hidden_states, global_cell_states = controller(
                torch.cat([edit_encodings.expand(len(hypotheses), edit_encodings.size(1)),
                           input_aggregation_fn(cur_input_encodings.encoding, cur_input_encodings.mask)], dim=1),
                (last_global_hidden_states, last_global_cell_states))

            for hyp_idx, hyp in enumerate(hypotheses): # update, will be copied to new hypotheses
                hyp.meta['global_states'] = (global_hidden_states[hyp_idx], global_cell_states[hyp_idx])

            top_partial_hypotheses = self.one_step_beam_search_with_source_encodings(
                hypotheses, init_code_ast, context, global_hidden_states, context_encodings,
                init_input_encodings, memory_encodings, cur_input_encodings, substitution_system, beam_size, t)

            for partial_hyp in top_partial_hypotheses:
                if partial_hyp.stopped:
                    completed_hypotheses.append(partial_hyp)
                else:
                    new_hypotheses.append(partial_hyp)

            if len(new_hypotheses) == 0 or len(completed_hypotheses) >= beam_size:
                break

            hypotheses = sorted(new_hypotheses, key=lambda x: x.score, reverse=True)[:beam_size]

        completed_hypotheses.extend(hypotheses)

        sorted_completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x.score, reverse=True)[:beam_size]
        top_completed_hypotheses = []
        for hyp in sorted_completed_hypotheses:
            # hyp.tree.reindex_wo_dummy_reduce()
            if not return_ast:
                hyp.tree = hyp.tree.root_node
            top_completed_hypotheses.append(hyp)

        return top_completed_hypotheses
