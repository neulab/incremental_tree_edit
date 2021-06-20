# coding=utf-8

try:
    from cStringIO import StringIO
except:
    from io import StringIO

from .asdl import *
from collections import OrderedDict
from typing import List, Tuple, Dict, Union


class AbstractSyntaxNode(object):
    def __init__(self, production, realized_fields=None, id=-1):
        self.id = id
        self.production = production

        # a child is essentially a *realized_field*
        self.fields = []

        # record its parent field to which it's attached
        self.parent_field = None

        # used in decoding, record the time step when this node was created
        self.created_time = 0

        if realized_fields:
            assert len(realized_fields) == len(self.production.fields)

            for field in realized_fields:
                self.add_child(field)
        else:
            for field in self.production.fields:
                self.add_child(RealizedField(field))

        self._to_string = None

    def add_child(self, realized_field):
        # if isinstance(realized_field.value, AbstractSyntaxNode):
        #     realized_field.value.parent = self
        self.fields.append(realized_field)
        realized_field.parent_node = self

    def replace_child_w_idx(self, realized_field, field_idx):
        self.fields[field_idx] = realized_field
        realized_field.parent_node = self
        realized_field.remove_ancestor_to_string_stamp()

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field_name: return field
        raise KeyError

    def sanity_check(self):
        if len(self.production.fields) != len(self.fields):
            raise ValueError('filed number must match')
        for field, realized_field in zip(self.production.fields, self.fields):
            assert field == realized_field.field
        for child in self.fields:
            for child_val in child.as_value_list:
                if isinstance(child_val, AbstractSyntaxNode):
                    child_val.sanity_check()

    def copy(self):
        new_tree = AbstractSyntaxNode(self.production, id=self.id)
        new_tree.created_time = self.created_time
        for i, old_field in enumerate(self.fields):
            new_field = new_tree.fields[i]
            new_field._not_single_cardinality_finished = old_field._not_single_cardinality_finished
            if isinstance(old_field.type, ASDLCompositeType) or \
                    (not isinstance(old_field.type, ASDLPrimitiveType) and old_field.type.is_composite):
                for value in old_field.as_value_list:
                    new_field.add_value(value.copy())
            else:
                for value in old_field.as_value_list:
                    new_field.add_value(value.copy())

        return new_tree

    def to_string(self):
        if self._to_string is not None:
            return self._to_string

        self._to_string = '('
        self._to_string += self.production.constructor.name

        for field in self.fields:
            self._to_string += ' '
            self._to_string += '('
            self._to_string += field.type.name
            self._to_string += (Field.get_cardinality_repr(field.cardinality))
            self._to_string += '-'
            self._to_string += field.name

            if field.value is not None:
                for val_node in field.as_value_list:
                    self._to_string += ' '
                    if isinstance(field.type, ASDLCompositeType) or \
                            (not isinstance(field.type, ASDLPrimitiveType) and field.type.is_composite):
                        val_to_string = val_node.to_string()
                        self._to_string += val_to_string
                    else:
                        self._to_string += (str(val_node).replace(' ', '-SPACE-'))

            self._to_string += ')'  # of field

        self._to_string += ')'  # of node
        return self._to_string

    def __hash__(self):
        code = hash(self.production)
        for field in self.fields:
            code = code + 37 * hash(field)

        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        # if self.created_time != other.created_time:
        #     return False

        if self.production != other.production:
            return False

        if len(self.fields) != len(other.fields):
            return False

        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]: return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.production)

    @property
    def descendant_nodes(self):
        def _visit(node):
            if isinstance(node, AbstractSyntaxNode):
                yield node

                for field in node.fields:
                    for field_val in field.as_value_list:
                        yield from _visit(field_val)

        yield from _visit(self)

    @property
    def descendant_nodes_and_tokens(self):
        def _visit(node):
            if isinstance(node, AbstractSyntaxNode):
                yield node

                for field in node.fields:
                    for field_val in field.as_value_list:
                        yield from _visit(field_val)
            else:
                yield node

        yield from _visit(self)

    @property
    def descendant_tokens(self):
        def _visit(node):
            if isinstance(node, AbstractSyntaxNode):
                for field in node.fields:
                    for field_val in field.as_value_list:
                        yield from _visit(field_val)
            else:
                yield node

        yield from _visit(self)

    @property
    def size(self):
        node_num = 1
        for field in self.fields:
            for val in field.as_value_list:
                # if isinstance(val, AbstractSyntaxNode):
                #     node_num += val.size
                # else:
                #     node_num += 1
                node_num += val.size

        return node_num

    @property
    def depth(self):
        return 1 + max(max(val.depth) for field in self.fields for val in field.as_value_list)

    @property
    def height(self):
        node_height = 1

        max_child_height = 0
        for field in self.fields:
            for val in field.as_value_list:
                if isinstance(val, AbstractSyntaxNode):
                    child_height = val.height
                else:
                    child_height = 1
                if child_height > max_child_height:
                    max_child_height = child_height

        node_height += max_child_height

        return node_height


class RealizedField(Field):
    """wrapper of field realized with values"""
    def __init__(self, field, value=None, parent=None):
        super(RealizedField, self).__init__(field.name, field.type, field.cardinality)

        # record its parent AST node
        self.parent_node = None

        # FIXME: hack, return the field as a property
        self.field = field

        # initialize value to correct type
        if self.cardinality == 'multiple':
            self.value = []
            if value is not None:
                for child_node in value:
                    self.add_value(child_node)
        else:
            self.value = None
            # note the value could be 0!
            if value is not None: self.add_value(value)

        # properties only used in decoding, record if the field is finished generating
        # when card in [optional, multiple]
        self._not_single_cardinality_finished = False

    def copy(self):
        if self.cardinality == 'multiple':
            value_copy = [child_node.copy() if child_node is not None else None for child_node in self.value]
        else:
            value_copy = self.value.copy() if self.value is not None else None

        new_field = RealizedField(self.field, value=value_copy)
        new_field.parent_node = self.parent_node
        new_field._not_single_cardinality_finished = self._not_single_cardinality_finished

        return new_field

    def remove_ancestor_to_string_stamp(self):
        cur_field = self
        while cur_field:
            cur_parent_node = cur_field.parent_node
            if cur_parent_node is not None:
                cur_parent_node._to_string = None
                cur_field = cur_parent_node.parent_field
            else:
                break

    def add_value(self, value):
        # if isinstance(value, AbstractSyntaxNode):
        value.parent_field = self

        if self.cardinality == 'multiple':
            self.value.append(value)
        else:
            self.value = value
        self.remove_ancestor_to_string_stamp()

    def add_value_w_idx(self, value, value_idx):
        value.parent_field = self

        if self.cardinality == 'multiple':
            self.value.insert(value_idx, value)
        else:
            self.value = value
        self.remove_ancestor_to_string_stamp()

    def remove(self, value):
        """remove a value from the field"""
        if self.cardinality in ('single', 'optional'):
            if self.value == value:
                self.value = None
            else:
                raise ValueError(f'{value} is not a value of the field {self}')
        else:
            tgt_idx = self.value.index(value)
            self.value.pop(tgt_idx)
        self.remove_ancestor_to_string_stamp()

    def replace(self, value, new_value):
        """replace an old field value with a new one"""
        if self.cardinality == 'multiple':
            tgt_idx = self.value.index(value)

            new_value.parent_field = self
            self.value[tgt_idx] = new_value
        else:
            assert self.value == value

            new_value.parent_field = self
            self.value = new_value
        self.remove_ancestor_to_string_stamp()

    def replace_w_idx(self, new_value, value_idx):
        new_value.parent_field = self
        if self.cardinality == 'multiple':
            self.value[value_idx] = new_value
        else:
            self.value = new_value
        self.remove_ancestor_to_string_stamp()

    def remove_w_idx(self, value_idx):
        if self.cardinality == 'multiple':
            self.value.pop(value_idx)
        else:
            self.value = None
        self.remove_ancestor_to_string_stamp()

    def find(self, value):
        if self.cardinality == 'multiple':
            value_list = self.value
        else:
            value_list = [self.value]

        for value_idx in range(len(value_list)):
            if value_list[value_idx] == value:
                return value_idx

        return -1 # not found

    @property
    def as_value_list(self):
        """get value as an iterable"""
        if self.cardinality == 'multiple': return self.value
        elif self.value is not None: return [self.value]
        else: return []

    @property
    def finished(self):
        if self.cardinality == 'single':
            if self.value is None: return False
            else: return True
        elif self.cardinality == 'optional' and self.value is not None:
            return True
        else:
            if self._not_single_cardinality_finished: return True
            else: return False

    def set_finish(self):
        # assert self.cardinality in ('optional', 'multiple')
        self._not_single_cardinality_finished = True

    def set_open(self):
        self._not_single_cardinality_finished = False

    def __eq__(self, other):
        if super(RealizedField, self).__eq__(other):
            if type(other) == Field: return True  # FIXME: hack, Field and RealizedField can compare!
            if self.value == other.value: return True
            else: return False
        else: return False


class SyntaxToken(object):
    """represent a terminal token on an AST"""
    def __init__(self, type, value, position=-1, id=-1):
        self.id = id
        self.type = type
        self.value = value
        self.position = position # index of the token in the original (code tok seq) input

        # record its parent field to which it's attached
        self.parent_field = None

    @property
    def size(self):
        return 1

    @property
    def depth(self):
        return 0

    def copy(self):
        return SyntaxToken(self.type, self.value, position=self.position, id=self.id)

    def __hash__(self):
        code = hash(self.type) + 37 * hash(self.value)

        return code

    def __repr__(self):
        return repr(self.value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.type == other.type and self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)


class DummyReduce(SyntaxToken):
    def __init__(self, type, position=-1, id=-1):
        super().__init__(type, value="[DUMMY-REDUCE]", position=position, id=id)

    def to_string(self):
        return str(self.value)

    def copy(self):
        return DummyReduce(self.type, position=self.position, id=self.id)

    @property
    def size(self):
        return 0


class AbstractSyntaxTree(object):
    def __init__(self, root_node: AbstractSyntaxNode):
        self.root_node = root_node

        self.adjacency_list: List[Tuple[int, int]] = None
        self.id2node: Dict[int, Union[AbstractSyntaxNode, SyntaxToken]] = None
        self.syntax_tokens_and_ids: List[Tuple[int, SyntaxToken]] = None

        self._get_properties()

    def _get_properties(self):
        """assign numerical indices to index each node"""
        id2nodes = OrderedDict()
        syntax_token_position2id = OrderedDict()
        terminal_tokens_list = []
        adj_list = []

        def _index_sub_tree(root_node, parent_node):
            if parent_node:
                adj_list.append((parent_node.id, root_node.id))

            id2nodes[root_node.id] = root_node
            if isinstance(root_node, AbstractSyntaxNode):
                for field in root_node.fields:
                    for field_val in field.as_value_list:
                        _index_sub_tree(field_val, root_node)
            else:
                # it's a syntax token
                terminal_tokens_list.append((root_node.id, root_node))
                syntax_token_position2id[root_node.position] = root_node.id

        _index_sub_tree(self.root_node, None)

        self.adjacency_list = adj_list
        self.id2node = id2nodes
        self.syntax_tokens_and_ids = terminal_tokens_list
        self.syntax_token_position2id = syntax_token_position2id
        self.syntax_tokens_set = {token: id for id, token in terminal_tokens_list}
        self.node_num = len(id2nodes)

        # this property are used for training and beam search, to get ids of syntax tokens
        # given their surface values
        syntax_token_value2ids = dict()
        for id, token in self.syntax_tokens_and_ids:
            syntax_token_value2ids.setdefault(token.value, []).append(id)
        self.syntax_token_value2ids = syntax_token_value2ids

        self._init_sibling_adjacency_list()

    def _init_sibling_adjacency_list(self):
        next_siblings = []

        def _travel(node):
            if isinstance(node, AbstractSyntaxNode):
                child_nodes = []
                for field in node.fields:
                    for val in field.as_value_list:
                        child_nodes.append(val)
                for i in range(len(child_nodes) - 1):
                    left_node = child_nodes[i]
                    right_node = child_nodes[i + 1]
                    next_siblings.append((left_node.id, right_node.id))

                for child_node in child_nodes:
                    _travel(child_node)

        _travel(self.root_node)
        setattr(self, 'next_siblings_adjacency_list', next_siblings)

    @property
    def syntax_tokens(self) -> List[SyntaxToken]:
        return [token for id, token in self.syntax_tokens_and_ids]

    @property
    def descendant_nodes(self) -> List[AbstractSyntaxNode]:
        for node_id, node in self.id2node.items():
            if isinstance(node, AbstractSyntaxNode):
                yield node_id, node

    def is_syntax_token(self, token):
        if isinstance(token, int):
            return isinstance(self.id2node[token], SyntaxToken)
        else:
            return token in self.syntax_tokens_set

    def find_node(self, query_node: AbstractSyntaxNode, return_id=True):
        search_results = []
        for node_id, node in self.descendant_nodes:
            if node.production == query_node.production:
                if node == query_node:
                    if return_id:
                        search_results.append((node_id, node))
                    else:
                        search_results.append(node)

        return search_results

    def copy(self):
        ast_copy = AbstractSyntaxTree(root_node=self.root_node.copy())

        return ast_copy

    def reindex_w_dummy_reduce(self):
        dummy_node_ids = []

        def _reassign_node_id(root_node, root_node_id):
            next_id = root_node_id
            root_node.id = next_id
            next_id += 1

            if isinstance(root_node, AbstractSyntaxNode):
                for field in root_node.fields:
                    bool_has_dummy_reduce = False
                    for value in field.as_value_list:
                        next_id = _reassign_node_id(value, next_id)
                        if isinstance(value, DummyReduce):
                            bool_has_dummy_reduce = True
                            dummy_node_ids.append(value.id)

                    field.set_open()
                    if not field.finished and not bool_has_dummy_reduce:
                        # add a dummy reduce child
                        dummy_child = DummyReduce(field.type, id=next_id)
                        field.add_value(dummy_child)
                        dummy_node_ids.append(dummy_child.id)
                        next_id += 1

            return next_id

        _reassign_node_id(self.root_node, 0)
        setattr(self, 'dummy_node_ids', dummy_node_ids)

        self._get_properties()

    def reindex_wo_dummy_reduce(self):
        def _reassign_node_id(root_node, root_node_id):
            next_id = root_node_id
            root_node.id = next_id
            next_id += 1

            if isinstance(root_node, AbstractSyntaxNode):
                for field in root_node.fields:
                    dummy_value_indices = [idx for idx in range(len(field.as_value_list))
                                           if isinstance(field.as_value_list[idx], DummyReduce)]
                    for idx in dummy_value_indices:
                        field.remove_w_idx(idx)
                    for value in field.as_value_list:
                        next_id = _reassign_node_id(value, next_id)

            return next_id

        _reassign_node_id(self.root_node, 0)
        setattr(self, 'dummy_node_ids', None)

        self._get_properties()

    def copy_and_reindex_w_dummy_reduce(self):
        ast_copy = self.copy()
        ast_copy.reindex_w_dummy_reduce()

        return ast_copy

    def copy_and_reindex_wo_dummy_reduce(self):
        ast_copy = self.copy()
        ast_copy.reindex_wo_dummy_reduce()

        return ast_copy
