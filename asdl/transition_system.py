# coding=utf-8


class Action(object):
    pass


class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.production.__repr__()


class GenTokenAction(Action):
    def __init__(self, token):
        self.token = token

    def is_stop_signal(self):
        return self.token == '</primitive>'

    def __repr__(self):
        return 'GenToken[%s]' % self.token

    def __eq__(self, other):
        return isinstance(other, GenTokenAction) and self.token == other.token


class ReduceAction(Action):
    def __repr__(self):
        return 'Reduce'

    def __eq__(self, other):
        return isinstance(other, ReduceAction)


class ApplySubTreeAction(Action):
    def __init__(self, tree, tree_node_ids=-1):
        self.tree = tree
        self.tree_node_ids = tree_node_ids

    def __repr__(self):
        return 'ApplySubTree[%s], Node[%d]' % (repr(self.tree), self.tree.id)


class DecodingAction:
    def __init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index):
        self.t = t
        self.parent_t = parent_t
        self.frontier_prod = frontier_prod
        self.frontier_field = frontier_field
        self.preceding_syntax_token_index = preceding_syntax_token_index


class ApplyRuleDecodingAction(ApplyRuleAction, DecodingAction):
    def __init__(self, t, parent_t, frontier_prod, frontier_field, production, preceding_syntax_token_index=None):
        ApplyRuleAction.__init__(self, production)
        DecodingAction.__init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)


class ApplySubTreeDecodingAction(ApplySubTreeAction, DecodingAction):
    def __init__(self, t, parent_t, frontier_prod, frontier_field, tree, tree_node_ids,
                 preceding_syntax_token_index=None):
        ApplySubTreeAction.__init__(self, tree, tree_node_ids)
        DecodingAction.__init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)


class ReduceDecodingAction(ReduceAction, DecodingAction):
    def __init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index=None):
        DecodingAction.__init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)


class GenTokenDecodingAction(GenTokenAction, DecodingAction):
    def __init__(self, t, parent_t, frontier_prod, frontier_field, token, preceding_syntax_token_index=None):
        GenTokenAction.__init__(self, token)
        DecodingAction.__init__(self, t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)


class TransitionSystem(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def get_actions(self, asdl_ast):
        """
        generate action sequence given the ASDL Syntax Tree
        """

        actions = []

        parent_action = ApplyRuleAction(asdl_ast.production)
        actions.append(parent_action)

        for field in asdl_ast.fields:
            # is a composite field
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    field_actions = self.get_actions(field.value)
                else:
                    field_actions = []

                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            for val in field.value:
                                cur_child_actions = self.get_actions(val)
                                field_actions.extend(cur_child_actions)
                        elif field.cardinality == 'optional':
                            field_actions = self.get_actions(field.value)

                    # if an optional field is filled, then do not need Reduce action
                    if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                        field_actions.append(ReduceAction())
            else:  # is a primitive field
                field_actions = self.get_primitive_field_actions(field)

                # if an optional field is filled, then do not need Reduce action
                if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                    # reduce action
                    field_actions.append(ReduceAction())

            actions.extend(field_actions)

        return actions

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field):
        raise NotImplementedError

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return GenTokenAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    if hyp._value_buffer:
                        return GenTokenAction,
                    else:
                        return GenTokenAction, ReduceAction
                else:
                    return GenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_valid_continuating_productions(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                return self.grammar[hyp.frontier_field.type]
            else:
                raise ValueError
        else:
            return self.grammar[self.grammar.root_type]

    @staticmethod
    def get_class_by_lang(lang):
        if lang == 'python':
            from .lang.py.py_transition_system import PythonTransitionSystem
            return PythonTransitionSystem
        elif lang == 'python3':
            from .lang.py3.py3_transition_system import Python3TransitionSystem
            return Python3TransitionSystem
        elif lang == 'lambda_dcs':
            from .lang.lambda_dcs.lambda_dcs_transition_system import LambdaCalculusTransitionSystem
            return LambdaCalculusTransitionSystem
        elif lang == 'prolog':
            from .lang.prolog.prolog_transition_system import PrologTransitionSystem
            return PrologTransitionSystem

        raise ValueError('unknown language %s' % lang)
