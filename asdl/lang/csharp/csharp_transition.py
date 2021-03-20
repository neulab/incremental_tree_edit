from typing import List

from asdl.asdl_ast import AbstractSyntaxNode, AbstractSyntaxTree, SyntaxToken
from asdl.transition_system import TransitionSystem, ApplySubTreeAction, ApplyRuleAction, ReduceAction, GenTokenAction, \
    ApplyRuleDecodingAction, ApplySubTreeDecodingAction, ReduceDecodingAction, GenTokenDecodingAction


class CSharpTransitionSystem(TransitionSystem):
    END_OF_SYNTAX_TOKEN_LIST_SYMBOL = '</s>'

    def __init__(self, grammar):
        super().__init__(grammar)
        self.starting_production = self.grammar.get_prod_by_ctr_name('BlockSyntax')

    def get_actions(self, target_ast_node: AbstractSyntaxNode, prev_ast: AbstractSyntaxTree=None, copy_identifier=True):
        """
        generate action sequence given the ASDL Syntax Tree
        """

        actions = []

        found_sub_tree = False
        if prev_ast:
            search_results = list(filter(lambda x: x[1].size > 1, prev_ast.find_node(target_ast_node)))  # list[(id, node)]
            if search_results:
                src_node_ids = [id for id, node in search_results]
                src_node = search_results[0][1]
                if not (src_node.production.type.name == 'IdentifierNameSyntax' and not copy_identifier):
                    action = ApplySubTreeAction(src_node, src_node_ids)
                    found_sub_tree = True

                    actions.append(action)

        if not found_sub_tree:
            action = ApplyRuleAction(target_ast_node.production)
            actions.append(action)

            for field in target_ast_node.fields:
                if self.grammar.is_composite_type(field.type):
                    if field.cardinality == 'single':
                        field_actions = self.get_actions(prev_ast=prev_ast, target_ast_node=field.value, copy_identifier=copy_identifier)
                    else:
                        field_actions = []

                        if field.value is not None:
                            if field.cardinality == 'multiple':
                                for val in field.value:
                                    cur_child_actions = self.get_actions(prev_ast=prev_ast, target_ast_node=val, copy_identifier=copy_identifier)
                                    field_actions.extend(cur_child_actions)
                            elif field.cardinality == 'optional':
                                field_actions = self.get_actions(prev_ast=prev_ast, target_ast_node=field.value, copy_identifier=copy_identifier)

                        # if an optional field is filled, then do not need Reduce action
                        if field.cardinality == 'multiple' or (field.cardinality == 'optional' and not field_actions):
                            field_actions.append(ReduceAction())
                else:  # is a primitive field
                    field_actions = self.get_primitive_field_actions(field)

                    # if an optional field is filled, then do not need Reduce action
                    if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                        # a special action to terminate populating a syntax token list
                        field_actions.append(GenTokenAction(SyntaxToken(field.type, CSharpTransitionSystem.END_OF_SYNTAX_TOKEN_LIST_SYMBOL)))

                actions.extend(field_actions)

        return actions

    def get_all_action_paths(self, target_ast_node: AbstractSyntaxNode, prev_ast: AbstractSyntaxTree, sample_size=None):
        def _extend_path(_current_paths, _new_paths):
            _combined_paths = []
            if len(_current_paths) == 0:
                return _new_paths
            if len(_new_paths) == 0:
                return _current_paths

            for old_action_path in _current_paths:
                for new_action_path in _new_paths[:sample_size]:
                    _combined_paths.append(old_action_path + new_action_path)

            return _combined_paths

        action_paths = []

        if prev_ast:
            search_result = prev_ast.find_node(target_ast_node)
            if search_result and search_result[1].size > 1:
                src_node_id, src_node = search_result
                copy_subtree_action = ApplySubTreeAction(target_ast_node, src_node_id)

                action_paths.append([copy_subtree_action])

        apply_rule_action = ApplyRuleAction(target_ast_node.production)
        apply_rule_action_paths = [[apply_rule_action]]

        for field in target_ast_node.fields:
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    field_action_paths = self.get_all_action_paths(prev_ast=prev_ast, target_ast_node=field.value)
                else:
                    field_action_paths = []

                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            for val in field.value:
                                cur_child_action_paths = self.get_all_action_paths(prev_ast=prev_ast, target_ast_node=val)
                                field_action_paths = _extend_path(field_action_paths, cur_child_action_paths)
                        elif field.cardinality == 'optional':
                            field_action_paths = self.get_all_action_paths(prev_ast=prev_ast, target_ast_node=field.value)

                    # if an optional field is filled, then do not need Reduce action
                    if field.cardinality == 'multiple' or (field.cardinality == 'optional' and field.value is None):
                        field_action_paths = _extend_path(field_action_paths, [[ReduceAction()]])
            else:  # is a primitive field
                field_actions = self.get_primitive_field_actions(field)

                # if an optional field is filled, then do not need Reduce action
                if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                    # reduce action
                    field_actions.append(ReduceAction())

                field_action_paths = [field_actions]

            apply_rule_action_paths = _extend_path(apply_rule_action_paths, field_action_paths)

        action_paths.extend(apply_rule_action_paths)

        return action_paths

    def get_decoding_actions(self, target_ast: AbstractSyntaxTree=None, prev_ast: AbstractSyntaxTree=None, actions=None, copy_identifier=True):
        from .csharp_hypothesis import CSharpHypothesis

        if actions is None:
            assert target_ast is not None
            actions = self.get_actions(target_ast_node=target_ast.root_node, prev_ast=prev_ast, copy_identifier=copy_identifier)

        hyp = CSharpHypothesis()
        decode_actions = []
        for t, action in enumerate(actions):
            if hyp.frontier_node:
                parent_t = hyp.frontier_node.created_time
                frontier_prod = hyp.frontier_node.production
                frontier_field = hyp.frontier_field.field
                prev_syntax_tokens = list(hyp.tree.descendant_tokens)
                preceding_syntax_token_index = len(prev_syntax_tokens) - 1
            else:
                parent_t = -1
                frontier_prod = frontier_field = None
                preceding_syntax_token_index = -1

            if isinstance(action, ApplyRuleAction):
                decode_action = ApplyRuleDecodingAction(t, parent_t, frontier_prod, frontier_field, action.production, preceding_syntax_token_index)
            elif isinstance(action, ApplySubTreeAction):
                decode_action = ApplySubTreeDecodingAction(t, parent_t, frontier_prod, frontier_field, action.tree, action.tree_node_ids, preceding_syntax_token_index)
            elif isinstance(action, ReduceAction):
                decode_action = ReduceDecodingAction(t, parent_t, frontier_prod, frontier_field, preceding_syntax_token_index)
            else:
                decode_action = GenTokenDecodingAction(t, parent_t, frontier_prod, frontier_field, action.token, preceding_syntax_token_index)

            if isinstance(action, (ApplyRuleAction, ReduceAction, ApplySubTreeAction)):
                valid_cont_action_types = self.get_valid_continuation_types(hyp)
                valid_cont_prod_ids = []

                for action_type in valid_cont_action_types:
                    if action_type == ApplyRuleAction:
                        # get valid continuating rules
                        if frontier_field:
                            valid_cont_prods = self.grammar[frontier_field.type]
                        else:
                            valid_cont_prods = [self.starting_production]

                        valid_cont_prod_ids.extend([self.grammar.prod2id[prod] for prod in valid_cont_prods])
                    elif action_type == ReduceAction:
                        valid_cont_prod_ids.append(len(self.grammar))
                    elif action_type == ApplySubTreeAction:
                        pass

                if frontier_field:
                    valid_subtree_types = self.grammar.descendant_types[frontier_field.type]
                else:
                    valid_subtree_types = [self.starting_production.type]

                if prev_ast:
                    valid_cont_subtree_ids = [node_id for node_id, node in prev_ast.descendant_nodes
                                              if node.production.type in valid_subtree_types]
                else:
                    valid_cont_subtree_ids = []

                if isinstance(action, ApplyRuleAction):
                    assert self.grammar.prod2id[action.production] in valid_cont_prod_ids
                elif isinstance(action, ReduceAction):
                    assert len(self.grammar) in valid_cont_prod_ids
                elif isinstance(action, ApplySubTreeAction):
                    assert all(tree_id in valid_cont_subtree_ids for tree_id in action.tree_node_ids)

                decode_action.valid_continuating_production_ids = valid_cont_prod_ids
                decode_action.valid_continuating_subtree_ids = valid_cont_subtree_ids

            hyp.apply_action(action)
            decode_actions.append(decode_action)

        return decode_actions

    def get_all_decoding_action_paths(self, target_ast: AbstractSyntaxTree, prev_ast: AbstractSyntaxTree, sample_size=None):
        decode_action_paths = []

        action_paths = self.get_all_action_paths(target_ast_node=target_ast.root_node, prev_ast=prev_ast, sample_size=sample_size)
        for actions in action_paths:
            decode_actions = self.get_decoding_actions(actions=actions)
            decode_action_paths.append(decode_actions)

        return decode_action_paths

    def get_transformation_actions_for_ast_node(self, prev_ast: AbstractSyntaxTree, target_ast_node: AbstractSyntaxNode):
        actions = []

        result = prev_ast.find_node(target_ast_node)
        if result and result[1].size > 1:
            src_node_id, src_node = result
            action = ApplySubTreeAction(target_ast_node, src_node_id)

            actions.append(action)
        else:
            action = ApplyRuleAction(target_ast_node.production)
            actions.append(action)

            for field in target_ast_node.fields:
                if self.grammar.is_composite_type(field.type):
                    if field.cardinality == 'single':
                        field_actions = self.get_transformation_actions_for_ast_node(prev_ast, field.value)
                    else:
                        field_actions = []

                        if field.value is not None:
                            if field.cardinality == 'multiple':
                                for val in field.value:
                                    cur_child_actions = self.get_transformation_actions_for_ast_node(prev_ast, val)
                                    field_actions.extend(cur_child_actions)
                            elif field.cardinality == 'optional':
                                field_actions = self.get_transformation_actions_for_ast_node(prev_ast, field.value)

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

    def get_transformation_decoding_actions(self, prev_ast: AbstractSyntaxTree, updated_ast: AbstractSyntaxTree):
        actions = self.get_transformation_actions_for_ast_node(prev_ast, updated_ast.root_node)
        decoding_actions = self.get_decoding_actions(actions=actions)

        return decoding_actions

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def hyp_correct(self, hyp, example):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field):
        actions = []
        if realized_field.value is not None:
            if realized_field.cardinality == 'multiple':
                field_values = realized_field.value
            else:
                field_values = [realized_field.value]

            for tok in field_values:
                actions.append(GenTokenAction(tok))

        return actions

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction, ApplySubTreeAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ApplySubTreeAction, ReduceAction
            else:
                return GenTokenAction,
        else:
            return ApplyRuleAction, ApplySubTreeAction,

    def get_valid_continuating_productions(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                return self.grammar[hyp.frontier_field.type]
            else:
                raise ValueError
        else:
            # return self.grammar[self.grammar.root_type]
            return [self.grammar.get_prod_by_ctr_name('BlockSyntax')]
