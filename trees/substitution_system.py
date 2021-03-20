# coding=utf-8
import numpy as np
from asdl.asdl_ast import AbstractSyntaxTree, AbstractSyntaxNode, SyntaxToken, RealizedField, DummyReduce
from asdl.transition_system import ApplyRuleAction, GenTokenAction
from trees.utils import copy_tree_field, stack_subtrees, get_sibling_ids, get_field_repr, find_by_id, get_field_node_queue
from trees.hypothesis import Hypothesis
from trees.edits import Delete, Add, AddSubtree, Stop

from asdl.lang.csharp.csharp_transition import CSharpTransitionSystem

INF = 1000


class SubstitutionSystem(object):
    def __init__(self, transition_system):
        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

    def get_decoding_edits_fast(self, src_ast: AbstractSyntaxTree, tgt_ast: AbstractSyntaxTree,
                                bool_copy_subtree=False, memory_space='all_init', memory_encode='joint',
                                preset_memory=None, init_code_tokens=None, last_edit_field_node=None,
                                bool_debug=False):
        """
        Generate gold edit sequence for training model decoder. Fast implementation.

        edit_priority_field_node_queue (optional): a root-to-node trace of the last edit location; when it is given, the algorithm
            prioritizes gold edits at or after the location.
        """
        assert memory_space in ('all_init', 'deleted')
        assert memory_encode in ('joint', 'distinct')
        memory_type = '_'.join([memory_space, memory_encode])

        # src_ast = src_ast.copy_and_reindex_w_dummy_reduce()
        # tgt_ast = tgt_ast.copy_and_reindex_w_dummy_reduce()

        assert src_ast.root_node.production == tgt_ast.root_node.production, "WARNING: Different AST roots found!"
        memory = None
        if bool_copy_subtree:
            if preset_memory is not None:
                preset_memory_repr = []
                for _node in preset_memory:
                    if isinstance(_node, AbstractSyntaxNode):
                        preset_memory_repr.append(_node.to_string())
                    else:
                        preset_memory_repr.append(str(_node).replace(' ', '-SPACE-'))
                memory = set(preset_memory_repr)
            else:
                if memory_space == 'all_init':  # from the whole initial code snippet
                    memory = set(stack_subtrees(src_ast.root_node, bool_repr=True))
                else:
                    assert memory_space == 'deleted'  # from previously deleted subtrees
                    memory = set()
        edit_size, edit_mappings, end_memory = self.ast_tree_compare(
            src_ast.root_node, tgt_ast.root_node,
            memory=memory, memory_space=memory_space)

        def _generate_decoding_edits(src_node_in_hyp, tgt_node, edit_mappings, hyp, priority_field_node_queue=None):
            tgt_decoding_edits = []

            src_node_parent_field = src_node_in_hyp.parent_field
            if src_node_parent_field is not None:
                src_node_parent_field_repr = get_field_repr(src_node_parent_field)
                src_node_val_idx = find_by_id(src_node_parent_field.as_value_list, src_node_in_hyp)

            num_total_fields = len(src_node_in_hyp.fields)

            prioritized_field_indices = range(num_total_fields)
            if priority_field_node_queue is not None and len(priority_field_node_queue):
                prioritized_field_idx, prioritized_field = priority_field_node_queue.pop(0)
                assert isinstance(prioritized_field, RealizedField)
                assert len(src_node_in_hyp.fields) > prioritized_field_idx and get_field_repr(prioritized_field) == \
                       get_field_repr(src_node_in_hyp.fields[prioritized_field_idx])
                if prioritized_field_idx != 0:
                    prioritized_field_indices = list(range(prioritized_field_idx, num_total_fields)) + list(range(prioritized_field_idx))

            for level_edit_idx, field_idx in enumerate(prioritized_field_indices):
                if level_edit_idx == 0: # the very first edit
                    working_src_node = src_node_in_hyp
                else: # locate "src_node_in_hyp" in the current hyp
                    if src_node_parent_field is None: # root
                        working_src_node = hyp.tree.root_node
                    else:
                        working_src_node_parent_field = hyp.repr2field[src_node_parent_field_repr]
                        working_src_node = working_src_node_parent_field.as_value_list[src_node_val_idx]

                src_field = working_src_node.fields[field_idx]
                tgt_field = tgt_node.fields[field_idx]

                if (field_idx, field_idx) not in edit_mappings or \
                        len(edit_mappings[(field_idx, field_idx)]) == 0:  # empty field
                    assert src_field.cardinality in ('optional', 'multiple')
                    continue

                mappings = edit_mappings[(field_idx, field_idx)]

                src_value_list = src_field.as_value_list
                value_pointer = 0

                prioritized_mappings = mappings
                reset_mapping_idx = 0
                if priority_field_node_queue is not None and len(priority_field_node_queue):
                    prioritized_node_idx, prioritized_node = priority_field_node_queue.pop(0)
                    assert isinstance(prioritized_node, (AbstractSyntaxNode, SyntaxToken))
                    if src_field.cardinality == 'multiple':
                        src_val_idx_in_mappings = [_i for _i in range(len(mappings)) if mappings[_i][0] is not None]
                        assert mappings[src_val_idx_in_mappings[prioritized_node_idx]][0] == prioritized_node

                        start_mapping_idx = src_val_idx_in_mappings[prioritized_node_idx]
                        if start_mapping_idx != 0:
                            prioritized_mappings = mappings[start_mapping_idx:] + mappings[:start_mapping_idx]
                            reset_mapping_idx = len(mappings) - start_mapping_idx
                            value_pointer = prioritized_node_idx # revise from the prioritized_node_idx-th child

                for mapping_idx, (src_val, tgt_val, child_edit_mappings) in enumerate(prioritized_mappings):
                    if mapping_idx == reset_mapping_idx:
                        value_pointer = 0

                    if src_val is not None and tgt_val is None:  # del
                        assert child_edit_mappings is None

                        if bool_debug:
                            field_repr = get_field_repr(src_field)
                            assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                            cur_field = hyp.repr2field[field_repr]
                            assert id(cur_field) == id(src_field)
                        else:
                            cur_field = src_field

                        cur_code_ast = hyp.tree
                        anchor_node = cur_field.as_value_list[value_pointer]
                        assert anchor_node == src_val
                        valid_cont_node_and_ids = self.get_valid_continuating_node_ids(hyp, "delete", bool_return_node=True)
                        if bool_debug: assert (anchor_node, anchor_node.id) in valid_cont_node_and_ids
                        valid_cont_node_ids = [node_id for node, node_id in valid_cont_node_and_ids]
                        left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, anchor_node)

                        decoding_edit = Delete(cur_field, value_pointer, anchor_node,
                                               meta={'tree': cur_code_ast,
                                                     'left_sibling_ids': left_sibling_ids,
                                                     'right_sibling_ids': right_sibling_ids,
                                                     'valid_cont_node_ids': valid_cont_node_ids})
                        tgt_decoding_edits.append(decoding_edit)

                        hyp = hyp.copy_and_apply_edit(decoding_edit)
                        edited_field_hyp = hyp.last_edit_field_node[0]
                        if bool_debug and preset_memory is None:
                            pos_in_init_tree = set(hyp.init_tree_w_dummy_reduce.syntax_token_position2id.keys())
                            pos_in_cur_tree = set(hyp.tree.syntax_token_position2id.keys())
                            assert len(pos_in_cur_tree - pos_in_init_tree) == 0

                        src_field = edited_field_hyp
                        src_value_list = src_field.as_value_list

                        # clear priority_field_node_queue
                        if priority_field_node_queue is not None and len(priority_field_node_queue):
                            priority_field_node_queue.clear()

                    elif src_val is None and tgt_val is not None:  # add new node
                        if child_edit_mappings is None:
                            assert isinstance(tgt_val, SyntaxToken)
                            tgt_val.position = -1 # safeguard

                            if tgt_field.cardinality == 'multiple':
                                _tmp_tgt_field = RealizedField(tgt_field.field, value=[tgt_val])
                            else:
                                _tmp_tgt_field = RealizedField(tgt_field.field, value=tgt_val)
                            field_actions = self.transition_system.get_primitive_field_actions(_tmp_tgt_field)
                            # for action in field_actions:
                            #     if isinstance(action.token, SyntaxToken):
                            #         action.token = action.token.value

                            _value_buffer = []
                            for action in field_actions:
                                if bool_debug:
                                    field_repr = get_field_repr(src_field)
                                    assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                                    cur_field = hyp.repr2field[field_repr]
                                    assert id(cur_field) == id(src_field)
                                else:
                                    cur_field = src_field

                                cur_code_ast = hyp.tree
                                anchor_node = cur_field.as_value_list[value_pointer]
                                valid_cont_node_and_ids = self.get_valid_continuating_node_ids(hyp, "add", bool_return_node=True)
                                if bool_debug: assert (anchor_node, anchor_node.id) in valid_cont_node_and_ids
                                valid_cont_node_ids = [node_id for node, node_id in valid_cont_node_and_ids]
                                left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, anchor_node)

                                meta = {'tree': cur_code_ast,
                                        'left_sibling_ids': left_sibling_ids,
                                        'right_sibling_ids': right_sibling_ids,
                                        'valid_cont_node_ids': valid_cont_node_ids}

                                assert isinstance(action, GenTokenAction) and \
                                       "add_gen_token" in self.get_valid_continuating_add_types(hyp, cur_field)
                                assert isinstance(action.token, SyntaxToken) and action.token.position == -1

                                decoding_edit = Add(cur_field, value_pointer, action,
                                                    value_buffer=list(_value_buffer) if _value_buffer is not None else None,
                                                    meta=meta)
                                tgt_decoding_edits.append(decoding_edit)

                                # buffer update
                                if src_field.type.name == 'string':
                                    assert not isinstance(self.transition_system, CSharpTransitionSystem)  # FIXME
                                    if not action.is_stop_signal():
                                        _value_buffer.append(action.token.value if isinstance(action.token, SyntaxToken)
                                                             else action.token)

                                hyp = hyp.copy_and_apply_edit(decoding_edit)
                                edited_field_hyp = hyp.last_edit_field_node[0]
                                if bool_debug and preset_memory is None:
                                    pos_in_init_tree = set(hyp.init_tree_w_dummy_reduce.syntax_token_position2id.keys())
                                    pos_in_cur_tree = set(hyp.tree.syntax_token_position2id.keys())
                                    assert len(pos_in_cur_tree - pos_in_init_tree) == 0

                                src_field = edited_field_hyp
                                src_value_list = src_field.as_value_list

                            value_pointer += 1
                        elif isinstance(child_edit_mappings, str) and child_edit_mappings == "[COPY]":  # copy subtree
                            if bool_debug:
                                field_repr = get_field_repr(src_field)
                                assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                                cur_field = hyp.repr2field[field_repr]
                                assert id(cur_field) == id(src_field)
                            else:
                                cur_field = src_field

                            cur_code_ast = hyp.tree
                            anchor_node = cur_field.as_value_list[value_pointer]
                            valid_cont_node_and_ids = self.get_valid_continuating_node_ids(hyp, "add_subtree", bool_return_node=True)
                            if bool_debug: assert (anchor_node, anchor_node.id) in valid_cont_node_and_ids
                            valid_cont_node_ids = [node_id for node, node_id in valid_cont_node_and_ids]

                            left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, anchor_node)
                            # tree_node_ids_to_copy = [node.id for node in hyp.memory if node == edit.node]
                            tree_node_ids_to_copy = []
                            safe_edit_node, safe_edit_node_idx_in_memory = None, None
                            for node_idx_in_memory, node in enumerate(hyp.memory):
                                if node == tgt_val:
                                    # tree_node_ids_to_copy.append(node.id)
                                    tree_node_ids_to_copy.append(node_idx_in_memory)
                                    if safe_edit_node is None:
                                        safe_edit_node = node
                                        safe_edit_node_idx_in_memory = node_idx_in_memory

                            assert safe_edit_node is not None
                            if bool_debug: assert "add_subtree" in self.get_valid_continuating_add_types(hyp, cur_field)
                            valid_cont_subtree_node_and_ids = self.get_valid_continuating_add_subtree(hyp, cur_field, bool_return_subtree=True)
                            # if bool_debug: assert (edit.node, edit.node.id) in valid_cont_subtree_node_and_ids
                            if bool_debug: assert (safe_edit_node, safe_edit_node_idx_in_memory) in valid_cont_subtree_node_and_ids
                            valid_cont_subtree_ids = [node_id for node, node_id in valid_cont_subtree_node_and_ids]
                            if bool_debug: assert max(valid_cont_subtree_ids) < len(hyp.memory)

                            decoding_edit = AddSubtree(cur_field, value_pointer, safe_edit_node,
                                                       meta={'tree': cur_code_ast,
                                                             'left_sibling_ids': left_sibling_ids,
                                                             'right_sibling_ids': right_sibling_ids,
                                                             'valid_cont_node_ids': valid_cont_node_ids,
                                                             'tree_node_ids_to_copy': tree_node_ids_to_copy,
                                                             'valid_cont_subtree_ids': valid_cont_subtree_ids})
                            tgt_decoding_edits.append(decoding_edit)

                            value_pointer += 1

                            hyp = hyp.copy_and_apply_edit(decoding_edit)
                            edited_field_hyp = hyp.last_edit_field_node[0]
                            if bool_debug and preset_memory is None:
                                pos_in_init_tree = set(hyp.init_tree_w_dummy_reduce.syntax_token_position2id.keys())
                                pos_in_cur_tree = set(hyp.tree.syntax_token_position2id.keys())
                                assert len(pos_in_cur_tree - pos_in_init_tree) == 0

                            src_field = edited_field_hyp
                            src_value_list = src_field.as_value_list
                        else:
                            if bool_debug:
                                field_repr = get_field_repr(src_field)
                                assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                                cur_field = hyp.repr2field[field_repr]
                                assert id(cur_field) == id(src_field)
                            else:
                                cur_field = src_field

                            cur_code_ast = hyp.tree
                            anchor_node = cur_field.as_value_list[value_pointer]
                            valid_cont_node_and_ids = self.get_valid_continuating_node_ids(hyp, "add", bool_return_node=True)
                            if bool_debug: assert (anchor_node, anchor_node.id) in valid_cont_node_and_ids
                            valid_cont_node_ids = [node_id for node, node_id in valid_cont_node_and_ids]
                            left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, anchor_node)

                            meta = {'tree': cur_code_ast,
                                    'left_sibling_ids': left_sibling_ids,
                                    'right_sibling_ids': right_sibling_ids,
                                    'valid_cont_node_ids': valid_cont_node_ids}

                            parent_action = ApplyRuleAction(tgt_val.production)
                            assert isinstance(parent_action, ApplyRuleAction)
                            if bool_debug: assert "add_apply_rule" in self.get_valid_continuating_add_types(hyp, cur_field)
                            valid_cont_prod_ids = self.get_valid_continuating_add_production_ids(hyp, cur_field)
                            if bool_debug: assert self.grammar.prod2id[parent_action.production] in valid_cont_prod_ids
                            meta['valid_cont_prod_ids'] = valid_cont_prod_ids

                            decoding_edit = Add(cur_field, value_pointer, parent_action, value_buffer=None, meta=meta)
                            tgt_decoding_edits.append(decoding_edit)

                            hyp = hyp.copy_and_apply_edit(decoding_edit)
                            edited_field_hyp = hyp.last_edit_field_node[0]
                            if bool_debug and preset_memory is None:
                                pos_in_init_tree = set(hyp.init_tree_w_dummy_reduce.syntax_token_position2id.keys())
                                pos_in_cur_tree = set(hyp.tree.syntax_token_position2id.keys())
                                assert len(pos_in_cur_tree - pos_in_init_tree) == 0

                            src_field = edited_field_hyp
                            src_value_list = src_field.as_value_list

                            # check children
                            new_src_val = src_value_list[value_pointer]
                            child_decoding_edits, hyp = _generate_decoding_edits(
                                new_src_val, tgt_val, child_edit_mappings, hyp,
                                priority_field_node_queue=priority_field_node_queue)
                            tgt_decoding_edits.extend(child_decoding_edits)

                            field_repr = get_field_repr(src_field)
                            assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                            src_field = hyp.repr2field[field_repr] # the corresponding field in the updated hyp
                            src_value_list = src_field.as_value_list

                            value_pointer += 1

                    else:
                        assert src_val is not None and tgt_val is not None
                        # assert src_val == src_value_list[value_pointer]
                        if child_edit_mappings is not None:
                            child_decoding_edits, hyp = _generate_decoding_edits(
                                src_value_list[value_pointer], tgt_val, child_edit_mappings, hyp,
                                priority_field_node_queue=priority_field_node_queue)
                            tgt_decoding_edits.extend(child_decoding_edits)

                            field_repr = get_field_repr(src_field)
                            assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                            src_field = hyp.repr2field[field_repr]  # the corresponding field in the updated hyp
                            src_value_list = src_field.as_value_list

                            value_pointer += 1
                        else:
                            value_pointer += 1

            return tgt_decoding_edits, hyp

        hyp = Hypothesis(src_ast, bool_copy_subtree=bool_copy_subtree, memory_type=memory_type,
                         init_code_tokens=init_code_tokens, memory=preset_memory)

        root_to_edit_field_node_queue = None
        if last_edit_field_node is not None:
            last_edit_node = last_edit_field_node[1]
            if isinstance(last_edit_node, DummyReduce): # either the last or the only child
                last_edit_field = last_edit_field_node[0]
                if len(last_edit_field.as_value_list) == 1:
                    last_edit_node = last_edit_field.parent_node
                else:
                    last_edit_node = last_edit_field.as_value_list[-2]
            root_to_edit_field_node_queue = get_field_node_queue(last_edit_node)

        tgt_decoding_edits, hyp = _generate_decoding_edits(hyp.tree.root_node, tgt_ast.root_node,
                                                           edit_mappings, hyp,
                                                           priority_field_node_queue=root_to_edit_field_node_queue)

        # final Stop
        cur_code_ast = hyp.tree
        decoding_edit = Stop(meta={'tree': cur_code_ast})
        tgt_decoding_edits.append(decoding_edit)

        assert hyp.tree.root_node == tgt_ast.root_node # sanity check
        last_edit = tgt_decoding_edits[-1]
        last_edit.meta['memory'] = [AbstractSyntaxTree(subtree) for subtree in hyp.memory]

        return tgt_decoding_edits

    def ast_tree_compare(self, src_ast: AbstractSyntaxNode, tgt_ast: AbstractSyntaxNode,
                         memory=None, memory_space='all_init'):
        """
        Compare two AST trees.
        :param src_ast: source AST.
        :param tgt_ast: target AST.
        :param memory: subtree memory if applicable.
        :param memory_space: two options 1) 'all_init' means using a fixed memory space (i.e., "memory");
            2) 'deleted' means dynamically adding previously deleted subtrees into the memory space (so "memory"
            should be empty in the beginning).
        :return:
        """
        def _collect_new_nodes(asdl_ast, memory=None):
            edit_mappings = {} # field to mappings
            count_dist = 1 # count all its children (including root)

            for field_idx, field in enumerate(asdl_ast.fields):
                edit_mappings[(field_idx, field_idx)] = []
                for val in field.as_value_list:
                    if isinstance(val, DummyReduce):
                        continue

                    if memory is not None:
                        if isinstance(val, AbstractSyntaxNode):
                            val_repr = val.to_string()
                        else:
                            val_repr = str(val).replace(' ', '-SPACE-')
                        if val_repr in memory:
                            edit_mappings[(field_idx, field_idx)].append((None, val, "[COPY]"))
                            count_dist += 1
                            continue

                    if isinstance(val, AbstractSyntaxNode):
                        child_edit_mappings, child_count_dist = _collect_new_nodes(val, memory=memory)
                        edit_mappings[(field_idx, field_idx)].append((None, val, child_edit_mappings))
                        count_dist += child_count_dist
                    else:
                        edit_mappings[(field_idx, field_idx)].append((None, val, None))
                        count_dist += 1

            return edit_mappings, count_dist

        def _ast_tree_edit_distance(src_ast, tgt_ast, memory=None, memory_space="all_init"):
            if len(src_ast.fields) == len(tgt_ast.fields) == 0:
                return 0, {}, memory

            edit_distance_per_field = []
            edit_mappings = {} # field to mappings
            end_memory_per_field = []

            for field_idx, (src_field, tgt_field) in enumerate(zip(src_ast.fields, tgt_ast.fields)):
                src_value_list = ["dummy"] + src_field.as_value_list
                tgt_value_list = ["dummy"] + tgt_field.as_value_list

                edit_step = {}
                DP = np.zeros((len(src_value_list), len(tgt_value_list)), dtype=int)  # shortest distance
                # backtrace path: 0 border, 1 del, 2 add/copy, 3 match/substitute
                edit_backtrace_m = np.zeros((len(src_value_list), len(tgt_value_list)), dtype=int)
                accumulate_memory = {}

                # initialization
                DP[0, 0] = 0
                edit_step[(0, 0)] = []
                if memory is not None:
                    accumulate_memory[(0, 0)] = memory

                for src_value_idx in range(1, len(src_value_list)):
                    src_val = src_value_list[src_value_idx]
                    if isinstance(src_val, DummyReduce):
                        DP[src_value_idx, 0] = DP[src_value_idx - 1, 0]
                        edit_step[(src_value_idx, 0)] = [] # no need to delete a dummy node
                        edit_backtrace_m[src_value_idx, 0] = 1
                        if memory is not None:
                            accumulate_memory[(src_value_idx, 0)] = accumulate_memory[(src_value_idx - 1, 0)]
                    else:
                        DP[src_value_idx, 0] = DP[src_value_idx - 1, 0] + 1
                        edit_step[(src_value_idx, 0)] = [(src_val, None, None)] # delete src val
                        edit_backtrace_m[src_value_idx, 0] = 1
                        if memory is not None:
                            if memory_space == 'all_init':
                                accumulate_memory[(src_value_idx, 0)] = accumulate_memory[(src_value_idx - 1, 0)]
                            else:
                                accumulate_memory[(src_value_idx, 0)] = accumulate_memory[(src_value_idx-1, 0)].union(
                                    stack_subtrees(src_value_list[src_value_idx], bool_repr=True))

                for tgt_value_idx in range(1, len(tgt_value_list)):
                    if DP[0, tgt_value_idx - 1] == INF or \
                            (src_field.cardinality in ('single', 'optional') and src_field.value is not None):
                        # adding value to a finished single/optional field is not allowed
                        DP[0, tgt_value_idx] = INF
                        edit_step[(0, tgt_value_idx)] = None
                        if memory is not None:
                            accumulate_memory[(0, tgt_value_idx)] = None
                    else:
                        tgt_val = tgt_value_list[tgt_value_idx]
                        if isinstance(tgt_val, DummyReduce):
                            DP[0, tgt_value_idx] = DP[0, tgt_value_idx - 1]
                            edit_step[(0, tgt_value_idx)] = []
                            edit_backtrace_m[0, tgt_value_idx] = 2
                            if memory is not None:
                                accumulate_memory[(0, tgt_value_idx)] = accumulate_memory[(0, tgt_value_idx - 1)]
                            continue

                        if isinstance(tgt_val, AbstractSyntaxNode):
                            tgt_val_repr = tgt_val.to_string()
                        else:
                            tgt_val_repr = str(tgt_val).replace(' ', '-SPACE-')

                        if memory is not None:
                            if tgt_val_repr in accumulate_memory[(0, tgt_value_idx-1)]:
                                DP[0, tgt_value_idx] = DP[0, tgt_value_idx - 1] + 1
                                edit_step[(0, tgt_value_idx)] = [(None, tgt_val, "[COPY]")]
                                edit_backtrace_m[0, tgt_value_idx] = 2
                            else:
                                child_edit_mappings, step_dist = _collect_new_nodes(
                                    tgt_val, memory=accumulate_memory[(0, tgt_value_idx - 1)]) \
                                    if isinstance(tgt_val, AbstractSyntaxNode) else (None, 1)
                                DP[0, tgt_value_idx] = DP[0, tgt_value_idx - 1] + step_dist
                                edit_step[(0, tgt_value_idx)] = [(None, tgt_val, child_edit_mappings)]
                                edit_backtrace_m[0, tgt_value_idx] = 2

                            accumulate_memory[(0, tgt_value_idx)] = accumulate_memory[(0, tgt_value_idx - 1)]
                        else:
                            # add one subtree costs subtree.size
                            child_edit_mappings, step_dist = _collect_new_nodes(tgt_val) \
                                if isinstance(tgt_val, AbstractSyntaxNode) else (None, 1)
                            assert step_dist == tgt_val.size
                            DP[0, tgt_value_idx] = DP[0, tgt_value_idx - 1] + step_dist
                            edit_step[(0, tgt_value_idx)] = [(None, tgt_val, child_edit_mappings)]
                            edit_backtrace_m[0, tgt_value_idx] = 2

                for src_value_idx in range(1, len(src_value_list)):
                    for tgt_value_idx in range(1, len(tgt_value_list)):
                        src_val = src_value_list[src_value_idx]
                        tgt_val = tgt_value_list[tgt_value_idx]
                        if isinstance(tgt_val, AbstractSyntaxNode):
                            tgt_val_repr = tgt_val.to_string()
                        else:
                            tgt_val_repr = str(tgt_val).replace(' ', '-SPACE-')

                        possible_edits = []

                        if DP[src_value_idx - 1, tgt_value_idx] < INF:
                            if isinstance(src_val, DummyReduce): # delete a dummy tree node is meaningless
                                possible_edits.append((DP[src_value_idx-1, tgt_value_idx],
                                                       [], 1, accumulate_memory[(src_value_idx-1, tgt_value_idx)]
                                                       if memory is not None else None))
                            else:
                                end_memory = None
                                if memory is not None:
                                    if memory_space == 'all_init':
                                        end_memory = accumulate_memory[(src_value_idx - 1, tgt_value_idx)]
                                    else:
                                        end_memory = accumulate_memory[(src_value_idx-1, tgt_value_idx)].union(
                                            stack_subtrees(src_val, bool_repr=True))
                                possible_edits.append((DP[src_value_idx - 1, tgt_value_idx] + 1,
                                                       [(src_val, None, None)], 1,
                                                       end_memory))  # delete src_val

                        if DP[src_value_idx, tgt_value_idx - 1] < INF:
                            if isinstance(tgt_val, DummyReduce):
                                possible_edits.append((DP[src_value_idx, tgt_value_idx-1],
                                                       [], 2, accumulate_memory[(src_value_idx, tgt_value_idx-1)]
                                                       if memory is not None else None))
                            else:
                                if memory is not None:
                                    if tgt_val_repr in accumulate_memory[(src_value_idx, tgt_value_idx-1)]:
                                        # copy subtree
                                        possible_edits.append((DP[src_value_idx, tgt_value_idx - 1] + 1,
                                                               [(None, tgt_val, "[COPY]")], 2,
                                                               accumulate_memory[(src_value_idx, tgt_value_idx-1)]))
                                    else:
                                        child_edit_mappings, step_dist = _collect_new_nodes(
                                            tgt_val, memory=accumulate_memory[(src_value_idx, tgt_value_idx-1)]) \
                                            if isinstance(tgt_val, AbstractSyntaxNode) else (None, 1)
                                        possible_edits.append((DP[src_value_idx, tgt_value_idx - 1] + step_dist,
                                                               [(None, tgt_val, child_edit_mappings)], 2,
                                                               accumulate_memory[(src_value_idx, tgt_value_idx-1)]))
                                else:
                                    child_edit_mappings, step_dist = _collect_new_nodes(tgt_val) \
                                        if isinstance(tgt_val, AbstractSyntaxNode) else (None, 1)
                                    assert step_dist == tgt_val.size
                                    possible_edits.append((DP[src_value_idx, tgt_value_idx - 1] + step_dist,
                                                           [(None, tgt_val, child_edit_mappings)], 2,
                                                           None))  # add tgt_val

                        if DP[src_value_idx - 1, tgt_value_idx - 1] < INF:
                            if isinstance(src_val, AbstractSyntaxNode) and isinstance(tgt_val, AbstractSyntaxNode) and \
                                    src_val.production == tgt_val.production:
                                # edit src_val to tgt_val
                                end_memory = None
                                if memory is not None:
                                    child_edit_distance, child_edit_mappings, child_memory = _ast_tree_edit_distance(
                                        src_val, tgt_val, memory=accumulate_memory[(src_value_idx-1, tgt_value_idx-1)])
                                    if memory_space == 'all_init':
                                        end_memory = accumulate_memory[(src_value_idx-1, tgt_value_idx-1)]
                                    else:
                                        end_memory = accumulate_memory[(src_value_idx-1, tgt_value_idx-1)].union(child_memory)
                                else:
                                    child_edit_distance, child_edit_mappings, child_memory = _ast_tree_edit_distance(
                                        src_val, tgt_val)
                                possible_edits.append((DP[src_value_idx - 1, tgt_value_idx - 1] + child_edit_distance,
                                                       [(src_val, tgt_val, child_edit_mappings)], 3,
                                                       end_memory))

                            elif isinstance(src_val, SyntaxToken) and isinstance(tgt_val, SyntaxToken) and \
                                    src_val == tgt_val:
                                if isinstance(src_val, DummyReduce):
                                    possible_edits.append((DP[src_value_idx - 1, tgt_value_idx - 1],
                                                           [], 3,
                                                           accumulate_memory[(src_value_idx - 1, tgt_value_idx - 1)]
                                                           if memory is not None else None))
                                else:
                                    possible_edits.append((DP[src_value_idx - 1, tgt_value_idx - 1],
                                                           [(src_val, tgt_val, None)], 3,
                                                           accumulate_memory[(src_value_idx-1, tgt_value_idx-1)]
                                                           if memory is not None else None))

                        assert len(possible_edits), "Found no valid trace!"

                        shortest_edit_idx = int(np.argmin([edit[0] for edit in possible_edits]))
                        DP[src_value_idx, tgt_value_idx] = possible_edits[shortest_edit_idx][0]
                        edit_step[(src_value_idx, tgt_value_idx)] = possible_edits[shortest_edit_idx][1]
                        edit_backtrace_m[src_value_idx, tgt_value_idx] = possible_edits[shortest_edit_idx][2]
                        if memory is not None:
                            accumulate_memory[(src_value_idx, tgt_value_idx)] = possible_edits[shortest_edit_idx][3]

                edit_distance_per_field.append(DP[len(src_value_list) - 1, len(tgt_value_list) - 1])
                edit_mappings[(field_idx, field_idx)] = []

                # backtrace edit process
                src_value_idx = len(src_value_list) - 1
                tgt_value_idx = len(tgt_value_list) - 1
                while src_value_idx >= 0 and tgt_value_idx >= 0:
                    assert edit_step[(src_value_idx, tgt_value_idx)] is not None
                    edit_mappings[(field_idx, field_idx)] = edit_step[(src_value_idx, tgt_value_idx)] + \
                                                            edit_mappings[(field_idx, field_idx)]
                    backtrace_act = edit_backtrace_m[src_value_idx, tgt_value_idx]
                    if backtrace_act == 1:
                        src_value_idx = src_value_idx - 1
                        tgt_value_idx = tgt_value_idx
                    elif backtrace_act == 2:
                        src_value_idx = src_value_idx
                        tgt_value_idx = tgt_value_idx - 1
                    elif backtrace_act == 3:
                        src_value_idx = src_value_idx - 1
                        tgt_value_idx = tgt_value_idx - 1
                    else:  # border
                        break

                if memory is not None and memory_space != 'all_init':
                    end_memory_per_field.append(accumulate_memory[(len(src_value_list) - 1, len(tgt_value_list) - 1)])
                    memory = end_memory_per_field[-1] # broadcast memory to the next field

            total_edit_distance = sum(edit_distance_per_field)
            total_memory = None
            if memory is not None and memory_space != 'all_init':
                total_memory = end_memory_per_field[-1]

            return total_edit_distance, edit_mappings, total_memory

        assert memory_space in ('all_init', 'deleted')

        if src_ast.production != tgt_ast.production:
            child_edit_mappings, step_dist = _collect_new_nodes(tgt_ast, memory=memory)
            edit_mappings = {(None, None): [(src_ast, None, None), (None, tgt_ast, child_edit_mappings)]}
            assert step_dist == tgt_ast.size
            edit_size = step_dist + 1 # del src_ast

            stack_memory = None
            if memory is not None and memory_space != 'all_init':
                stack_memory = memory.union(stack_subtrees(src_ast, bool_repr=True))
        else:
            edit_size, edit_mappings, stack_memory = _ast_tree_edit_distance(
                src_ast, tgt_ast, memory=memory, memory_space=memory_space)

        return edit_size, edit_mappings, stack_memory

    def get_decoding_edits(self, src_ast: AbstractSyntaxTree, tgt_ast: AbstractSyntaxTree,
                           bool_copy_subtree=False, memory_space='all_init', memory_encode='joint',
                           preset_memory=None, init_code_tokens=None, bool_debug=False):
        """
        Generate gold edit sequence for training model decoder. Compared with self.get_edits, aux decoding
        information is included in edit.meta.
        UPDATE 09/03: please use self.get_decoding_edits_fast for a faster implementation.
        """
        assert memory_space in ('all_init', 'deleted')
        assert memory_encode in ('joint', 'distinct')
        memory_type = '_'.join([memory_space, memory_encode])

        tgt_edits = self.get_edits(src_ast, tgt_ast, bool_copy_subtree=bool_copy_subtree, memory_space=memory_space,
                                   preset_memory=preset_memory)

        tgt_decoding_edits = []
        src_ast = src_ast.copy_and_reindex_w_dummy_reduce()
        tgt_ast = tgt_ast.copy_and_reindex_w_dummy_reduce()

        hyp = Hypothesis(src_ast, bool_copy_subtree=bool_copy_subtree, memory_type=memory_type,
                         init_code_tokens=init_code_tokens, memory=preset_memory)

        for edit in tgt_edits:
            if isinstance(edit, Delete):
                field_repr = get_field_repr(edit.field)
                assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                cur_code_ast = hyp.tree
                cur_field = hyp.repr2field[field_repr]
                anchor_node = cur_field.as_value_list[edit.value_idx]
                if bool_debug: assert anchor_node == edit.node
                valid_cont_node_and_ids = self.get_valid_continuating_node_ids(hyp, "delete", bool_return_node=True)
                if bool_debug: assert (anchor_node, anchor_node.id) in valid_cont_node_and_ids
                valid_cont_node_ids = [node_id for node, node_id in valid_cont_node_and_ids]
                left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, anchor_node)

                decoding_edit = Delete(cur_field, edit.value_idx, anchor_node,
                                       meta={'tree': cur_code_ast,
                                             'left_sibling_ids': left_sibling_ids,
                                             'right_sibling_ids': right_sibling_ids,
                                             'valid_cont_node_ids': valid_cont_node_ids})
                tgt_decoding_edits.append(decoding_edit)

            elif isinstance(edit, Add):
                field_repr = get_field_repr(edit.field)
                assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                cur_code_ast = hyp.tree
                cur_field = hyp.repr2field[field_repr]
                anchor_node = cur_field.as_value_list[edit.value_idx]
                valid_cont_node_and_ids = self.get_valid_continuating_node_ids(hyp, "add", bool_return_node=True)
                if bool_debug: assert (anchor_node, anchor_node.id) in valid_cont_node_and_ids
                valid_cont_node_ids = [node_id for node, node_id in valid_cont_node_and_ids]
                left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, anchor_node)

                meta = {'tree': cur_code_ast,
                        'left_sibling_ids': left_sibling_ids,
                        'right_sibling_ids': right_sibling_ids,
                        'valid_cont_node_ids': valid_cont_node_ids}

                if isinstance(edit.action, ApplyRuleAction):
                    if bool_debug: assert "add_apply_rule" in self.get_valid_continuating_add_types(hyp, cur_field)
                    valid_cont_prod_ids = self.get_valid_continuating_add_production_ids(hyp, cur_field)
                    if bool_debug: assert self.grammar.prod2id[edit.action.production] in valid_cont_prod_ids
                    meta['valid_cont_prod_ids'] = valid_cont_prod_ids
                else:
                    if bool_debug:
                        assert isinstance(edit.action, GenTokenAction) and \
                           "add_gen_token" in self.get_valid_continuating_add_types(hyp, cur_field)
                        assert isinstance(edit.action.token, SyntaxToken) and edit.action.token.position == -1

                decoding_edit = Add(cur_field, edit.value_idx, edit.action,
                                    value_buffer=list(edit._value_buffer) if edit._value_buffer is not None else None,
                                    meta=meta)
                tgt_decoding_edits.append(decoding_edit)

            elif isinstance(edit, AddSubtree):
                field_repr = get_field_repr(edit.field)
                assert field_repr in hyp.repr2field, "Apply edit: Field not found in state!"
                cur_code_ast = hyp.tree
                cur_field = hyp.repr2field[field_repr]
                anchor_node = cur_field.as_value_list[edit.value_idx]
                valid_cont_node_and_ids = self.get_valid_continuating_node_ids(hyp, "add_subtree", bool_return_node=True)
                if bool_debug: assert (anchor_node, anchor_node.id) in valid_cont_node_and_ids
                valid_cont_node_ids = [node_id for node, node_id in valid_cont_node_and_ids]

                left_sibling_ids, right_sibling_ids = get_sibling_ids(cur_field, anchor_node)
                # tree_node_ids_to_copy = [node.id for node in hyp.memory if node == edit.node]
                tree_node_ids_to_copy = []
                safe_edit_node, safe_edit_node_idx_in_memory = None, None
                for node_idx_in_memory, node in enumerate(hyp.memory):
                    if node == edit.node:
                        # tree_node_ids_to_copy.append(node.id)
                        tree_node_ids_to_copy.append(node_idx_in_memory)
                        if safe_edit_node is None:
                            safe_edit_node = node
                            safe_edit_node_idx_in_memory = node_idx_in_memory
                edit.node = safe_edit_node # replace, ensure the node can be found in memory

                if bool_debug: assert "add_subtree" in self.get_valid_continuating_add_types(hyp, cur_field)
                valid_cont_subtree_node_and_ids = self.get_valid_continuating_add_subtree(hyp, cur_field, bool_return_subtree=True)
                # if bool_debug: assert (edit.node, edit.node.id) in valid_cont_subtree_node_and_ids
                if bool_debug: assert (edit.node, safe_edit_node_idx_in_memory) in valid_cont_subtree_node_and_ids
                valid_cont_subtree_ids = [node_id for node, node_id in valid_cont_subtree_node_and_ids]
                if bool_debug: assert max(valid_cont_subtree_ids) < len(hyp.memory)

                decoding_edit = AddSubtree(cur_field, edit.value_idx, edit.node,
                                           meta={'tree': cur_code_ast,
                                                 'left_sibling_ids': left_sibling_ids,
                                                 'right_sibling_ids': right_sibling_ids,
                                                 'valid_cont_node_ids': valid_cont_node_ids,
                                                 'tree_node_ids_to_copy': tree_node_ids_to_copy,
                                                 'valid_cont_subtree_ids': valid_cont_subtree_ids})
                tgt_decoding_edits.append(decoding_edit)

            else:
                assert isinstance(edit, Stop)
                cur_code_ast = hyp.tree
                decoding_edit = Stop(meta={'tree': cur_code_ast})
                tgt_decoding_edits.append(decoding_edit)

            hyp = hyp.copy_and_apply_edit(decoding_edit)
            if bool_debug and preset_memory is None:
                pos_in_init_tree = set(hyp.init_tree_w_dummy_reduce.syntax_token_position2id.keys())
                pos_in_cur_tree = set(hyp.tree.syntax_token_position2id.keys())
                assert len(pos_in_cur_tree - pos_in_init_tree) == 0

        if bool_debug: assert hyp.tree.root_node == tgt_ast.root_node # sanity check
        last_edit = tgt_decoding_edits[-1]
        last_edit.meta['memory'] = [AbstractSyntaxTree(subtree) for subtree in hyp.memory]

        return tgt_decoding_edits

    def get_edits(self, src_ast: AbstractSyntaxTree, tgt_ast: AbstractSyntaxTree,
                  bool_copy_subtree=False, memory_space='all_init',
                  preset_memory=None):
        """
        Generate edit sequence to turn source AST into target AST.
        """
        assert memory_space in ('all_init', 'deleted')

        src_ast = src_ast.copy_and_reindex_w_dummy_reduce()
        tgt_ast = tgt_ast.copy_and_reindex_w_dummy_reduce()

        assert src_ast.root_node.production == tgt_ast.root_node.production, "WARNING: Different AST roots found!"
        memory = None
        if bool_copy_subtree:
            if preset_memory is not None:
                preset_memory_repr = []
                for _node in preset_memory:
                    if isinstance(_node, AbstractSyntaxNode):
                        preset_memory_repr.append(_node.to_string())
                    else:
                        preset_memory_repr.append(str(_node).replace(' ', '-SPACE-'))
                memory = set(preset_memory_repr)
            else:
                if memory_space == 'all_init': # from the whole initial code snippet
                    memory = set(stack_subtrees(src_ast.root_node, bool_repr=True))
                else:
                    assert memory_space == 'deleted' # from previously deleted subtrees
                    memory = set()
        edit_size, edit_mappings, end_memory = self.ast_tree_compare(src_ast.root_node, tgt_ast.root_node,
                                                                     memory=memory,
                                                                     memory_space=memory_space)
        # print(edit_mappings)

        def _restore_field_state_edits(src_node, tgt_node, edit_mappings, memory=None):
            edit_trace = []
            for field_idx, (src_field, tgt_field) in enumerate(zip(src_node.fields, tgt_node.fields)):
                if (field_idx, field_idx) not in edit_mappings or \
                        len(edit_mappings[(field_idx, field_idx)]) == 0:  # empty field
                    # empty field
                    assert src_field.cardinality in ('optional', 'multiple')
                    # edit_trace.append(Add(src_field, 0, ReduceAction()))
                    continue

                mappings = edit_mappings[(field_idx, field_idx)]

                src_value_list = src_field.as_value_list
                value_pointer = 0

                for src_val, tgt_val, child_edit_mappings in mappings:
                    if src_val is not None and tgt_val is None:  # del
                        assert child_edit_mappings is None
                        assert src_val == src_value_list[value_pointer]
                        src_ast_copy, src_field_copy = copy_tree_field(src_ast, src_field, bool_w_dummy_reduce=True)
                        edit_trace.append(Delete(src_field_copy, value_pointer,
                                                 src_field_copy.as_value_list[value_pointer]))

                        # actual perform to revise field state
                        if memory is not None and memory_space == 'deleted': # otherwise no need to modify
                            memory.extend(stack_subtrees(src_value_list[value_pointer]))
                        src_field.remove_w_idx(value_pointer)

                    elif src_val is None and tgt_val is not None:  # add new node
                        if child_edit_mappings is None:
                            assert isinstance(tgt_val, SyntaxToken)
                            tgt_val.position = -1 # safeguard

                            if tgt_field.cardinality == 'multiple':
                                _tmp_tgt_field = RealizedField(tgt_field.field, value=[tgt_val])
                            else:
                                _tmp_tgt_field = RealizedField(tgt_field.field, value=tgt_val)
                            field_actions = self.transition_system.get_primitive_field_actions(_tmp_tgt_field)
                            # for action in field_actions:
                            #     if isinstance(action.token, SyntaxToken):
                            #         action.token = action.token.value

                            _value_buffer = []
                            for action in field_actions:
                                src_ast_copy, src_field_copy = copy_tree_field(src_ast, src_field,
                                                                               bool_w_dummy_reduce=True)
                                edit = Add(src_field_copy, value_pointer, action,
                                           value_buffer=list(_value_buffer))
                                edit_trace.append(edit)

                                # actual perform to revise field state
                                if src_field.type.name == 'string':
                                    assert not isinstance(self.transition_system, CSharpTransitionSystem)  # FIXME
                                    if action.is_stop_signal():
                                        assert _value_buffer is not None and len(_value_buffer)
                                        src_field.add_value_w_idx(
                                            SyntaxToken(src_field.type, ' '.join(_value_buffer)),
                                            value_pointer)
                                        _value_buffer = []
                                    else:
                                        _value_buffer.append(action.token.value if isinstance(action.token, SyntaxToken)
                                                             else action.token)
                                else:
                                    src_field.add_value_w_idx(
                                        SyntaxToken(src_field.type, action.token.value if isinstance(action.token, SyntaxToken)
                                                    else action.token), value_pointer)

                            value_pointer += 1
                        elif isinstance(child_edit_mappings, str) and child_edit_mappings == "[COPY]":  # copy subtree
                            assert tgt_val in memory
                            src_ast_copy, src_field_copy = copy_tree_field(src_ast, src_field, bool_w_dummy_reduce=True)
                            edit_trace.append(AddSubtree(src_field_copy, value_pointer, tgt_val.copy())) # memory[memory.index(tgt_val)]

                            # actual perform to revise field state
                            src_field.add_value_w_idx(tgt_val.copy(), value_pointer)
                            value_pointer += 1
                        else:
                            parent_action = ApplyRuleAction(tgt_val.production)
                            src_ast_copy, src_field_copy = copy_tree_field(src_ast, src_field, bool_w_dummy_reduce=True)
                            edit_trace.append(Add(src_field_copy, value_pointer, parent_action))

                            # actual perform to revise field state
                            new_src_val = AbstractSyntaxNode(tgt_val.production)
                            src_field.add_value_w_idx(new_src_val, value_pointer)

                            # check children
                            child_edit_trace = _restore_field_state_edits(new_src_val, tgt_val, child_edit_mappings,
                                                                          memory=memory)
                            edit_trace.extend(child_edit_trace)

                            value_pointer += 1

                    else:
                        assert src_val is not None and tgt_val is not None
                        # assert src_val == src_value_list[value_pointer]
                        if child_edit_mappings is not None:
                            child_edit_trace = _restore_field_state_edits(
                                src_value_list[value_pointer], tgt_val, child_edit_mappings, memory=memory)
                            edit_trace.extend(child_edit_trace)
                            value_pointer += 1
                        else:
                            value_pointer += 1

                    src_value_list = src_field.as_value_list  # src_field might have changed
                    src_ast.reindex_w_dummy_reduce()

            return edit_trace

        src_ast = src_ast.copy()  # src_ast_copy will be edited in order to generate gold edit state and seq
        memory = None
        if bool_copy_subtree:
            if preset_memory is not None:
                memory = preset_memory
            else:
                if memory_space == 'all_init': # from the whole initial code snippet
                    memory = stack_subtrees(src_ast.root_node.copy())
                else:
                    assert memory_space == 'deleted' # from previously deleted subtrees
                    memory = []
        edit_trace = _restore_field_state_edits(src_ast.root_node, tgt_ast.root_node, edit_mappings, memory=memory)
        edit_trace += [Stop()]

        return edit_trace

    def get_valid_continuating_node_ids(self, hyp, operator, bool_return_node=False):
        if operator == "delete":
            if bool_return_node:
                return hyp.open_del_node_and_ids
            else:
                return [node_id for node, node_id in hyp.open_del_node_and_ids]
        elif operator == "add":
            if bool_return_node:
                return [(node, node.id) for field in hyp.open_add_fields for node in field.as_value_list]
            else:
                return [node.id for field in hyp.open_add_fields for node in field.as_value_list]
        elif operator == "add_subtree":
            # check if valid subtrees exist for each open field
            valid_fields = []
            for field in hyp.open_add_fields:
                if self.grammar.is_composite_type(field.type):
                    valid_productions = self.grammar[field.type]
                    for subtree in hyp.memory:
                        if isinstance(subtree, AbstractSyntaxNode) and subtree.production in valid_productions:
                            valid_fields.append(field)
                            break
            # return valid node and/or node ids
            if bool_return_node:
                return [(node, node.id) for field in valid_fields for node in field.as_value_list]
            else:
                return [node.id for field in valid_fields for node in field.as_value_list]
        elif operator == "stop":
            return []
        else:
            raise Exception("Operator %s not found!" % operator)

    def get_valid_continuating_add_types(self, hyp, field):
        if self.grammar.is_composite_type(field.type):
            return ("add_apply_rule", "add_subtree")
        else:
            return ("add_gen_token",)

    def get_valid_continuating_add_production_ids(self, hyp, field): # for Add(ApplyRule)
        return [self.grammar.prod2id[prod] for prod in self.grammar[field.type]]

    def get_valid_continuating_add_subtree(self, hyp, field, bool_return_subtree=False): # for AddSubtree
        valid_productions = self.grammar[field.type]

        valid_subtrees = []
        for subtree_idx_in_memory, subtree in enumerate(hyp.memory):
            if isinstance(subtree, AbstractSyntaxNode) and subtree.production in valid_productions:
                # valid_subtrees.append((subtree, subtree.id) if bool_return_subtree else subtree.id)
                valid_subtrees.append((subtree, subtree_idx_in_memory) if bool_return_subtree else subtree_idx_in_memory)
            elif isinstance(subtree, SyntaxToken):
                pass  # currently not allowed

        return valid_subtrees

