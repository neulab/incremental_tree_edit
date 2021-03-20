import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TreeDiffEncoder(nn.Module):
    def __init__(self, graph_encoding_size, input_size, change_vector_size, operators,
                 operator_embedding, production_embedding, field_embedding, token_embedding,
                 **kwargs):
        super(TreeDiffEncoder, self).__init__()

        self.input_size = input_size
        self.change_vector_size = change_vector_size
        self.operators = operators
        self.copy_syntax_token = kwargs['copy_syntax_token']

        self.operator_embedding = operator_embedding
        self.production_embedding = production_embedding
        self.field_embedding = field_embedding
        self.token_embedding = token_embedding

        operator_embed_size = self.operator_embedding.embedding_dim
        action_embed_size = self.production_embedding.embedding_dim
        field_embed_size = self.field_embedding.embedding_dim

        self.change_seq_encoder_lstm = nn.LSTM(self.input_size, self.change_vector_size // 2, bidirectional=True)

        # project [op; field; node] for Delete
        self.delete_projection = nn.Linear(operator_embed_size + field_embed_size + graph_encoding_size,
                                           input_size, bias=True)
        # project [op; field; node; action] for Add
        self.add_projection = nn.Linear(operator_embed_size + field_embed_size + graph_encoding_size +
                                        action_embed_size, input_size, bias=True)
        if 'add_subtree' in self.operators:
            # project [op; field; node; subtree] for AddSubtree
            self.add_subtree_projection = nn.Linear(operator_embed_size + field_embed_size + graph_encoding_size * 2,
                                                    input_size, bias=True)
        # project [op] for Stop
        self.stop_projection = nn.Linear(operator_embed_size, input_size, bias=True)

    @property
    def device(self):
        return self.production_embedding.weight.device

    def forward(self, batch_edits_list, batch_actual_edits_length, masks_cache,
                context_encodings, init_input_encodings, cur_input_encodings_list,
                batch_memory_encodings=None):

        assert len(batch_edits_list) == len(cur_input_encodings_list)
        max_iteration_step = len(batch_edits_list)
        batch_size = len(batch_edits_list[0])
        batch_max_node_num_over_time = max(cur_input_encodings.size(1)
                                           for cur_input_encodings in cur_input_encodings_list)

        # (max_iteration_step, batch_size, batch_max_node_num_over_time, source_element_encoding_size)
        cur_input_encodings_encoding_over_time = torch.zeros(
            max_iteration_step, batch_size, batch_max_node_num_over_time,
            cur_input_encodings_list[0].size(2)).to(self.device)
        for t, cur_input_encodings in enumerate(cur_input_encodings_list):
            cur_input_encodings_encoding_over_time[t, :, :cur_input_encodings.size(1)] = cur_input_encodings

        operator_selection_idx, \
        node_selection_idx, node_selection_mask, node_cand_mask, parent_field_idx, \
        tgt_apply_rule_idx, tgt_apply_rule_mask, apply_rule_cand_mask, \
        tgt_apply_subtree_idx, tgt_apply_subtree_idx_mask, tgt_apply_subtree_mask, apply_subtree_cand_mask, \
        tgt_gen_token_idx, tgt_gen_token_mask, tgt_copy_ctx_token_idx_mask, tgt_copy_ctx_token_mask, \
        tgt_copy_init_token_idx_mask, tgt_copy_init_token_mask = masks_cache

        # (max_iteration_step, batch_size, operator_emb_size)
        tgt_operator_embeddings = self.operator_embedding(operator_selection_idx)

        # (max_iteration_step, batch_size, field_emb_size)
        tgt_field_embeddings = self.field_embedding(parent_field_idx)

        self_mask = torch.zeros(max_iteration_step * batch_size, batch_max_node_num_over_time,
                                dtype=torch.long).to(self.device)
        self_mask[torch.arange(0, max_iteration_step * batch_size, dtype=torch.long).to(self.device),
                  node_selection_idx.view(-1)] = 1
        self_mask = self_mask.reshape(max_iteration_step, batch_size, batch_max_node_num_over_time)
        # (max_iteration_step, batch_size, source_element_encoding_size)
        tgt_node_encodings = torch.sum(cur_input_encodings_encoding_over_time * self_mask.unsqueeze(-1), dim=2)

        tgt_production_embeddings = self.production_embedding(tgt_apply_rule_idx)
        tgt_gen_token_embeddings = self.token_embedding(tgt_gen_token_idx)

        if self.copy_syntax_token:
            tgt_copy_ctx_token_idx_mask[tgt_copy_ctx_token_idx_mask.sum(-1).eq(0), :] = 1
            tgt_copy_ctx_token_embeddings = torch.sum(
                context_encodings.expand(max_iteration_step, -1, -1, -1) * tgt_copy_ctx_token_idx_mask.unsqueeze(-1),
                dim=2) / tgt_copy_ctx_token_idx_mask.sum(dim=-1, keepdim=True)

            tgt_copy_init_token_idx_mask[tgt_copy_init_token_idx_mask.sum(-1).eq(0), :] = 1
            tgt_copy_init_token_embeddings = torch.sum(
                init_input_encodings.expand(max_iteration_step, -1, -1, -1) * tgt_copy_init_token_idx_mask.unsqueeze(-1),
                dim=2) / tgt_copy_init_token_idx_mask.sum(dim=-1, keepdim=True)

        # prepare inputs
        _cand_inputs = [] # a list of (max_iteration_step, batch_size, input_size)
        for operator in self.operators: # ['delete', 'add', 'add_subtree', 'stop']
            if operator == 'stop':
                stop_inputs = self.stop_projection(tgt_operator_embeddings)
                _cand_inputs.append(stop_inputs)
            elif operator == 'delete':
                delete_inputs = self.delete_projection(
                    torch.cat([tgt_operator_embeddings, tgt_field_embeddings, tgt_node_encodings], dim=-1)
                )
                _cand_inputs.append(delete_inputs)
            elif operator == 'add':
                if self.copy_syntax_token:
                    tgt_token_gates = tgt_gen_token_mask + tgt_copy_ctx_token_mask + tgt_copy_init_token_mask
                    tgt_token_gates[tgt_token_gates.eq(0)] = 1 # safeguard
                    tgt_token_embeddings = tgt_gen_token_embeddings * tgt_gen_token_mask.unsqueeze(-1) + \
                                           tgt_copy_ctx_token_embeddings * tgt_copy_ctx_token_mask.unsqueeze(-1) + \
                                           tgt_copy_init_token_embeddings * tgt_copy_init_token_mask.unsqueeze(-1)
                    # (max_iteration_step, batch_size, action_emb_size)
                    tgt_token_embeddings = tgt_token_embeddings / tgt_token_gates.unsqueeze(-1)
                else:
                    tgt_token_embeddings = tgt_gen_token_embeddings

                tgt_action_embeddings = tgt_production_embeddings * tgt_apply_rule_mask.unsqueeze(-1) + \
                                        tgt_token_embeddings * (1 - tgt_apply_rule_mask.unsqueeze(-1))

                add_inputs = self.add_projection(
                    torch.cat([tgt_operator_embeddings, tgt_field_embeddings, tgt_node_encodings,
                               tgt_action_embeddings], dim=-1)
                )
                _cand_inputs.append(add_inputs)
            elif operator == 'add_subtree': # and tgt_apply_subtree_mask.eq(1).any()
                if batch_memory_encodings is None:
                    assert tgt_apply_subtree_mask.eq(0).all()
                    add_subtree_inputs = torch.zeros(max_iteration_step, batch_size, self.input_size).to(self.device)
                    _cand_inputs.append(add_subtree_inputs)
                    continue

                # a list of (max_iteration_step, batch_size), length=max_num_copied_nodes
                _tgt_apply_subtree_idx_unbind = torch.unbind(tgt_apply_subtree_idx, dim=-1)
                # reshape to (max_iteration_step*batch_size, max_num_nodes, encoding_size)
                _expanded_batch_memory_encodings = batch_memory_encodings.expand(max_iteration_step, -1, -1, -1).\
                    reshape(max_iteration_step * batch_size, batch_memory_encodings.size(1), batch_memory_encodings.size(2))

                _gathered_tgt_apply_subtree_encodings = []
                _count = torch.arange(max_iteration_step * batch_size).to(self.device)
                for _tgt_apply_subtree_idx_col in _tgt_apply_subtree_idx_unbind:
                    _gathered_tgt_apply_subtree_encodings.append(
                        _expanded_batch_memory_encodings[_count, _tgt_apply_subtree_idx_col.reshape(-1)])
                gathered_tgt_apply_subtree_encodings = torch.stack(_gathered_tgt_apply_subtree_encodings, dim=1).\
                    reshape(max_iteration_step, batch_size, -1, batch_memory_encodings.size(2))

                # safeguard
                _reshaped_tgt_apply_subtree_idx_mask = tgt_apply_subtree_idx_mask.reshape(
                    max_iteration_step * batch_size, -1)
                _reshaped_tgt_apply_subtree_idx_mask[_reshaped_tgt_apply_subtree_idx_mask.sum(-1).eq(0), 0] = 1
                tgt_apply_subtree_idx_mask = _reshaped_tgt_apply_subtree_idx_mask.reshape(max_iteration_step, batch_size, -1)

                # (max_iteration_step, batch_size, source_element_encoding_size)
                tgt_apply_subtree_encodings = torch.sum(
                    gathered_tgt_apply_subtree_encodings * tgt_apply_subtree_idx_mask.unsqueeze(-1), dim=2) / \
                                              tgt_apply_subtree_idx_mask.sum(dim=-1, keepdim=True)

                add_subtree_inputs = self.add_subtree_projection(
                    torch.cat([tgt_operator_embeddings, tgt_field_embeddings, tgt_node_encodings,
                               tgt_apply_subtree_encodings], dim=-1)
                )
                _cand_inputs.append(add_subtree_inputs)

        # (max_iteration_step*batch_size, num of operators, input_size)
        _cand_inputs = torch.stack(_cand_inputs, dim=2).reshape(max_iteration_step * batch_size, -1, self.input_size)
        _count = torch.arange(max_iteration_step * batch_size).to(self.device)
        # (max_iteration_step, batch_size, input_size)
        tgt_inputs = _cand_inputs[_count, operator_selection_idx.view(-1)].reshape(max_iteration_step, batch_size, self.input_size)

        padded_inputs_in_steps = pack_padded_sequence(tgt_inputs, batch_actual_edits_length,
                                                      batch_first=False, enforce_sorted=False)

        change_seq_encodings, (last_state, last_cell) = self.change_seq_encoder_lstm(padded_inputs_in_steps)
        # change_seq_encodings, _ = pad_packed_sequence(change_seq_encodings)

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        return last_state
