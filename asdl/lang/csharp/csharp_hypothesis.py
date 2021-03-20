from asdl.asdl_ast import AbstractSyntaxNode
from asdl.hypothesis import Hypothesis
from asdl.lang.csharp.csharp_transition import CSharpTransitionSystem
from asdl.transition_system import ApplySubTreeAction, ApplyRuleAction, ReduceAction, GenTokenAction


class CSharpHypothesis(Hypothesis):
    def __init__(self):
        super().__init__()

    def apply_action(self, action):
        if self.tree is None:
            assert isinstance(action, (ApplyRuleAction, ApplySubTreeAction)), \
                'Invalid action [%s], only ApplyRule and ApplySutTree actions are valid at the beginning of decoding'

            if isinstance(action, ApplyRuleAction):
                self.tree = AbstractSyntaxNode(action.production)
            elif isinstance(action, ApplySubTreeAction):
                self.tree = action.tree.copy()

            self.tree.created_time = self.t
            self.update_frontier_info()
        elif self.frontier_node:
            if self.frontier_field.type.is_composite:
                if isinstance(action, ApplyRuleAction):
                    field_value = AbstractSyntaxNode(action.production)
                    field_value.created_time = self.t
                    self.frontier_field.add_value(field_value)
                    self.update_frontier_info()
                elif isinstance(action, ApplySubTreeAction):
                    field_value = action.tree.copy()
                    self.frontier_field.add_value(field_value)
                    self.update_frontier_info()
                elif isinstance(action, ReduceAction):
                    assert self.frontier_field.cardinality in ('optional', 'multiple'), 'Reduce action can only be ' \
                                                                                        'applied on field with multiple ' \
                                                                                        'cardinality'
                    self.frontier_field.set_finish()
                    self.update_frontier_info()
                else:
                    raise ValueError('Invalid action [%s] on field [%s]' % (action, self.frontier_field))
            else:  # fill in a primitive field
                if isinstance(action, GenTokenAction):
                    # only field of type string requires termination signal </primitive>
                    if action.token.value == CSharpTransitionSystem.END_OF_SYNTAX_TOKEN_LIST_SYMBOL:
                        assert self.frontier_field.cardinality in ('optional', 'multiple'), 'Reduce action can only be ' \
                                                                                            'applied on field with multiple ' \
                                                                                            'cardinality'
                        self.frontier_field.set_finish()
                        self.update_frontier_info()
                    else:
                        self.frontier_field.add_value(action.token)

                        if self.frontier_field.cardinality in ('single', 'optional'):
                            self.frontier_field.set_finish()
                            self.update_frontier_info()
                else:
                    raise ValueError('Can only invoke GenToken actions on primitive fields')

        self.t += 1
        self.actions.append(action)

    def update_frontier_info(self):
        def _find_frontier_node_and_field(tree_node):
            if tree_node:
                for field in tree_node.fields:
                    # if it's an intermediate node, check its children
                    if field.type.is_composite and field.value:
                        if field.cardinality in ('single', 'optional'): iter_values = [field.value]
                        else: iter_values = field.value

                        for child_node in iter_values:
                            result = _find_frontier_node_and_field(child_node)
                            if result: return result

                    # now all its possible children are checked
                    if not field.finished:
                        return tree_node, field

                return None
            else: return None

        frontier_info = _find_frontier_node_and_field(self.tree)
        if frontier_info:
            self.frontier_node, self.frontier_field = frontier_info
        else:
            self.frontier_node, self.frontier_field = None, None

    def copy(self):
        new_hyp = CSharpHypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.score = self.score
        new_hyp._value_buffer = list(self._value_buffer)
        new_hyp.t = self.t

        new_hyp.update_frontier_info()

        return new_hyp
