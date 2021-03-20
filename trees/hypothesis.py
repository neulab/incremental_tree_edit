# coding=utf-8
from copy import deepcopy
import numpy as np

from .substitution_system import *
from asdl.asdl import ASDLCompositeType, ASDLPrimitiveType
from asdl.asdl_ast import AbstractSyntaxTree, DummyReduce, AbstractSyntaxNode
from trees.utils import get_field_repr, find_by_id, stack_subtrees
from trees.edits import Edit, Delete, Add, AddSubtree, Stop


class Hypothesis(object):
    def __init__(self, init_tree_w_dummy_reduce: AbstractSyntaxTree, bool_copy_subtree=False, tree=None,
                 memory=None, memory_type='all_init_joint', init_code_tokens=None, length_norm=False):
        self.init_tree_w_dummy_reduce = init_tree_w_dummy_reduce
        self.bool_copy_subtree = bool_copy_subtree
        assert memory_type in ('all_init_joint', 'all_init_distinct', 'deleted_distinct')
        self.memory_type = memory_type
        self.init_code_tokens = init_code_tokens
        self.length_norm = length_norm

        if tree is not None:
            self.tree = tree
        else:
            self.tree = init_tree_w_dummy_reduce.copy()

        if bool_copy_subtree and memory is None:
            if self.memory_type == 'all_init_joint':
                self.memory = stack_subtrees(self.init_tree_w_dummy_reduce.root_node)
            elif self.memory_type == 'all_init_distinct':
                self.memory = []
                for node in stack_subtrees(self.init_tree_w_dummy_reduce.root_node):
                    if node not in self.memory:
                        self.memory.append(node)
            else:
                self.memory = []
        else:
            self.memory = memory
        # self.set_tree_all_finish() # redundant?

        self.edits = []
        self.score_per_edit = []
        self.score = 0.

        self.repr2field = {}
        self.open_del_node_and_ids = [] # nodes available to delete
        self.open_add_fields = [] # fields open to add nodes
        self.restricted_frontier_fields = [] # fields (esp. with single cardinality) grammatically need to fill
        self.update_frontier_info()

        # record the current time step
        self.last_edit_field_node = None # trace the last edit
        self.t = 0
        self.stop_t = None

    def apply_edit(self, edit: Edit, score=0.0):
        if isinstance(edit, Stop):
            self.stop_t = self.t
            self.last_edit_field_node = None
        elif isinstance(edit, (Add, Delete, AddSubtree)):
            if isinstance(edit, AddSubtree):
                assert self.bool_copy_subtree and edit.node in self.memory

            field_repr = get_field_repr(edit.field)
            assert field_repr in self.repr2field, "Apply edit: Field not found in state!"
            old_field = self.repr2field[field_repr]

            field_idx = find_by_id(old_field.parent_node.fields, old_field) #old_field.parent_node.fields.index(old_field)
            assert field_idx != -1
            edited_field = edit.output
            old_field.parent_node.replace_child_w_idx(edited_field, field_idx)

            if self.bool_copy_subtree and self.memory_type == 'deleted_distinct' and isinstance(edit, Delete):
                for new_subtree in stack_subtrees(edit.node):
                    if new_subtree not in self.memory:
                        self.memory.append(new_subtree) # note this does not map to the original init tree

            self.tree.reindex_w_dummy_reduce()
            self.update_frontier_info()

            if isinstance(edit, (Add, AddSubtree)):
                self.last_edit_field_node = (edited_field, edited_field.as_value_list[edit.value_idx])
            elif isinstance(edit, Delete):
                # self.last_edit_field_node = (edited_field, None)
                valid_edit_value_idx = edit.value_idx
                if len(edited_field.as_value_list) <= valid_edit_value_idx:
                    valid_edit_value_idx = len(edited_field.as_value_list) - 1
                self.last_edit_field_node = (edited_field, edited_field.as_value_list[valid_edit_value_idx])

        else:
            raise ValueError('Invalid edit!')

        self.t += 1
        self.edits.append(edit)

        self.score_per_edit.append(score)
        if self.length_norm:
            assert len(self.edits) == len(self.score_per_edit)
            self.score = np.average(self.score_per_edit)
        else:
            self.score = sum(self.score_per_edit)

    def copy_and_apply_edit(self, edit: Edit, score=0.0):
        new_hyp = self.copy()
        new_hyp.apply_edit(edit, score=score)

        return new_hyp

    def copy(self):
        new_hyp = Hypothesis(self.init_tree_w_dummy_reduce,
                             bool_copy_subtree=self.bool_copy_subtree,
                             tree=self.tree.copy(),
                             memory=list(self.memory), # deepcopy(self.memory) # usually existing memory will not be modified
                             memory_type=self.memory_type,
                             init_code_tokens=self.init_code_tokens,
                             length_norm=self.length_norm)

        new_hyp.edits = list(self.edits)
        new_hyp.score_per_edit = list(self.score_per_edit)
        new_hyp.score = self.score
        new_hyp.t = self.t
        new_hyp.last_edit_field_node = None

        if hasattr(self, 'meta'):
            new_hyp.meta = deepcopy(self.meta)

        new_hyp.update_frontier_info()

        return new_hyp

    def update_frontier_info(self):
        open_del_node_and_ids = []
        open_add_fields = []
        restricted_frontier_fields = []
        repr2field = {}

        def _find_frontier_node_and_field(tree_node, field_repr_prefix):
            if tree_node:
                for field_idx, field in enumerate(tree_node.fields):
                    tmp_field_repr_prefix = field_repr_prefix + "%s-%d" % (str(tree_node), field_idx) + "-SEP-"
                    cur_field_repr = tmp_field_repr_prefix + str(field)
                    repr2field[cur_field_repr] = field

                    # if it's an intermediate node, check its children
                    if (isinstance(field.type, ASDLCompositeType) or
                        (not isinstance(field.type, ASDLPrimitiveType) and field.type.is_composite)) and \
                            field.value:
                        if field.cardinality in ('single', 'optional'): iter_values = [field.value]
                        else: iter_values = field.value

                        for child_node_idx, child_node in enumerate(iter_values):
                            if isinstance(child_node, DummyReduce):
                                continue
                            _find_frontier_node_and_field(
                                child_node, tmp_field_repr_prefix + "%s-%d" % (str(field), child_node_idx) + '-SEP-')

                    # now all its possible children are checked
                    # fields must add node
                    if field.cardinality == 'single' and (field.value is None or isinstance(field.value, DummyReduce)):
                        restricted_frontier_fields.append(field) # break grammar

                    # fields okay to delete node
                    if (field.cardinality in ('single', 'optional') and field.value is not None and
                        not isinstance(field.value, DummyReduce)) or \
                            (field.cardinality == 'multiple' and len(field.as_value_list)):
                        open_del_node_and_ids.extend([(node, node.id) for node in field.as_value_list
                                                      if not isinstance(node, DummyReduce)])

                    # fields okay to add node
                    if (field.cardinality in ('single', 'optional') and
                        (field.value is None or isinstance(field.value, DummyReduce))) or \
                            (field.cardinality == 'multiple'):
                        open_add_fields.append(field)

        _find_frontier_node_and_field(self.tree.root_node, '')
        self.open_del_node_and_ids = open_del_node_and_ids
        self.open_add_fields = open_add_fields
        self.restricted_frontier_fields = restricted_frontier_fields
        self.repr2field = repr2field

    def set_tree_all_finish(self):
        def _set_field_finish(tree_node):
            for field in tree_node.fields:
                field.set_finish()

                if (isinstance(field.type, ASDLCompositeType) or
                    (not isinstance(field.type, ASDLPrimitiveType) and field.type.is_composite)) and \
                        field.value:
                    if field.cardinality in ('single', 'optional'): iter_values = [field.value]
                    else: iter_values = field.value

                    for child_node in iter_values:
                        _set_field_finish(child_node)

        _set_field_finish(self.tree.root_node)

    @property
    def syntax_valid(self):
        return len(self.restricted_frontier_fields) == 0

    @property
    def stopped(self):
        return self.stop_t is not None
