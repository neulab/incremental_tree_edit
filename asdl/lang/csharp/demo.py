import json

from asdl.hypothesis import ApplyRuleAction, GenTokenAction
from asdl.lang.csharp.csharp_hypothesis import CSharpHypothesis
from asdl.lang.csharp.csharp_transition import CSharpTransitionSystem
from asdl.lang.csharp.csharp_grammar import CSharpASDLGrammar


if __name__ == '__main__':
    # csharp_grammar_text = request.urlopen('https://raw.githubusercontent.com/dotnet/roslyn/master/src/Compilers'
    #                                       '/CSharp/Portable/Syntax/Syntax.xml').read()
    csharp_grammar_text = open('Syntax.xml').read()
    # fields_to_ignore = ['SemicolonToken', 'OpenBraceToken', 'CloseBraceToken', 'CommaToken', 'ColonToken',
    #                     'StartQuoteToken', 'EndQuoteToken', 'OpenBracketToken', 'CloseBracketToken', 'NewKeyword']

    grammar = CSharpASDLGrammar.from_roslyn_xml(csharp_grammar_text, pruning=True)

    open('grammar.json', 'w').write(grammar.to_json())

    ast_json_list = open('../../../source_data/githubedits/githubedits.train_20p.jsonl').readlines()
    ast_json = ast_json_list[0]

    ast_json_obj = json.loads(ast_json)['PrevCodeAST']
    syntax_tree = grammar.get_ast_from_json_obj(ast_json_obj)
    ast_root = syntax_tree.root_node
    print(ast_root.to_string())
    print(ast_root.size)

    ast_json_obj_updt = json.loads(ast_json)['UpdatedCodeAST']
    syntax_tree_updt = grammar.get_ast_from_json_obj(ast_json_obj_updt)
    ast_root_updt = syntax_tree_updt.root_node
    print(ast_root_updt.to_string())
    print(ast_root_updt.size)

    transition = CSharpTransitionSystem(grammar)
    actions = transition.get_actions(ast_root)
    decode_actions = transition.get_decoding_actions(syntax_tree)

    print('Len actions:', len(decode_actions))

    with open('actions.txt', 'w') as f:
        for action in actions:
            f.write(str(action) + '\n')

    hyp = CSharpHypothesis()
    for action, decode_action in zip(actions, decode_actions):
        assert action.__class__ in transition.get_valid_continuation_types(hyp)
        if isinstance(action, ApplyRuleAction):
            assert action.production in transition.get_valid_continuating_productions(hyp)
            assert action.production == decode_action.production
        elif isinstance(action, GenTokenAction):
            assert action.token == decode_action.token

        if hyp.frontier_node:
            assert hyp.frontier_field == decode_action.frontier_field
            assert hyp.frontier_node.production == decode_action.frontier_prod

        hyp.apply_action(action)
    print(hyp.tree.to_string() == ast_root.to_string())


