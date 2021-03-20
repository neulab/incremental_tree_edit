# coding=utf-8
from asdl.asdl import ASDLCompositeType, ASDLPrimitiveType
from asdl.transition_system import ApplyRuleAction, GenTokenAction
from asdl.asdl_ast import AbstractSyntaxTree, AbstractSyntaxNode, SyntaxToken, RealizedField, DummyReduce


class Edit(object):
    pass


class Delete(Edit):
    def __init__(self, field, value_idx, node, meta=None):
        self.field = field
        self.value_idx = value_idx
        self.node = node
        self.meta = meta

    def _apply_edit(self):
        assert self.field.as_value_list[self.value_idx] == self.node, \
            "Delete: Node not found in Field (value idx %d)!" % self.value_idx

        edited_field = self.field.copy()
        edited_field.remove_w_idx(self.value_idx)
        return edited_field

    @property
    def output(self):
        return self._apply_edit()

    def __repr__(self):
        return 'Delete[%s, field %s, value idx %d]' % (
            self.node.__repr__(), self.field.__repr__(), self.value_idx)


class Add(Edit):
    def __init__(self, field, value_idx, action, value_buffer=None, meta=None):
        self.field = field
        self.value_idx = value_idx
        self.action = action
        self._value_buffer = value_buffer
        self.meta = meta

    def _apply_action(self):
        edited_field = self.field.copy()
        action = self.action

        if isinstance(edited_field.type, ASDLCompositeType) or \
                (not isinstance(edited_field.type, ASDLPrimitiveType) and edited_field.type.is_composite):
            if isinstance(action, ApplyRuleAction):
                field_value = AbstractSyntaxNode(action.production)
                edited_field.add_value_w_idx(field_value, self.value_idx)
                # edited_field.set_open()  # open the field once Add

            else:
                raise ValueError('Invalid action [%s] on field [%s]' % (action, edited_field))
        else:  # fill in a primitive field
            if isinstance(action, GenTokenAction):
                # only field of type string requires termination signal </primitive>
                end_primitive = False
                if edited_field.type.name == 'string':
                    if action.is_stop_signal():
                        assert self._value_buffer is not None and len(self._value_buffer)
                        edited_field.add_value_w_idx(
                            SyntaxToken(edited_field.type, ' '.join(self._value_buffer)), self.value_idx)
                        end_primitive = True
                else:
                    edited_field.add_value_w_idx(
                        SyntaxToken(edited_field.type, action.token.value if isinstance(action.token, SyntaxToken)
                        else action.token), self.value_idx)
                    end_primitive = True

                # if not end_primitive:
                #     edited_field.set_open()

                # if end_primitive and edited_field.cardinality in ('single', 'optional'):
                #     edited_field.set_finish()

            else:
                raise ValueError('Can only invoke GenToken or Reduce actions on primitive fields')

        return edited_field

    @property
    def output(self):
        return self._apply_action()

    def __repr__(self):
        return 'Add[%s, %s, value idx %d]' % (
            self.action.__repr__(), self.field.__repr__(), self.value_idx)


class AddSubtree(Edit):
    def __init__(self, field, value_idx, node, meta=None):
        self.field = field
        self.value_idx = value_idx
        self.node = node
        self.meta = meta

    def _apply_edit(self):
        edited_field = self.field.copy()
        edited_field.add_value_w_idx(self.node.copy(), self.value_idx)
        return edited_field

    @property
    def output(self):
        return self._apply_edit()

    def __repr__(self):
        return 'AddSubtree[%s, field %s, value idx %d]' % (
            self.node.__repr__(), self.field.__repr__(), self.value_idx)


class Stop(Edit):
    def __init__(self, meta=None):
        self.meta = meta

    def __repr__(self):
        return 'StopEdit'
