import json
from tqdm import tqdm

from asdl.lang.csharp.csharp_transition import CSharpTransitionSystem
from asdl.lang.csharp.csharp_grammar import CSharpASDLGrammar
from trees.substitution_system import SubstitutionSystem


def _encode(word_list):
    return [w.replace('\n', '-NEWLINE-') for w in word_list]


if __name__ == '__main__':
    # csharp_grammar_text = request.urlopen('https://raw.githubusercontent.com/dotnet/roslyn/master/src/Compilers'
    #                                       '/CSharp/Portable/Syntax/Syntax.xml').read()
    csharp_grammar_text = open('Syntax.xml').read()
    # fields_to_ignore = ['SemicolonToken', 'OpenBraceToken', 'CloseBraceToken', 'CommaToken', 'ColonToken',
    #                     'StartQuoteToken', 'EndQuoteToken', 'OpenBracketToken', 'CloseBracketToken', 'NewKeyword']

    grammar = CSharpASDLGrammar.from_roslyn_xml(csharp_grammar_text, pruning=True)

    open('grammar.json', 'w').write(grammar.to_json())

    ast_json_lines = open('../../../source_data/githubedits/githubedits.train_20p.jsonl').readlines()
    for ast_json_idx in tqdm(range(len(ast_json_lines))):
        ast_json = ast_json_lines[ast_json_idx]
        loaded_ast_json = json.loads(ast_json)

        ast_json_obj_prev = loaded_ast_json['PrevCodeAST']
        syntax_tree_prev = grammar.get_ast_from_json_obj(ast_json_obj_prev)
        # ast_root_prev = syntax_tree_prev.root_node
        # print(loaded_ast_json['PrevCodeChunk'])
        # print(ast_root_prev.to_string(), "\n")
        # print(ast_root_prev.size)

        ast_json_obj_updated = loaded_ast_json['UpdatedCodeAST']
        syntax_tree_updated = grammar.get_ast_from_json_obj(ast_json_obj_updated)
        # ast_root_updated = syntax_tree_updated.root_node
        # print(loaded_ast_json['UpdatedCodeChunk'])
        # print(ast_root_updated.to_string(), "\n")
        # print(ast_root_updated.size)

        transition = CSharpTransitionSystem(grammar)
        # actions = transition.get_actions(ast_root_updated)
        # decode_actions = transition.get_decoding_actions(
        #     target_ast=syntax_tree_updated,
        #     prev_ast=syntax_tree_prev,
        #     copy_identifier=True)
        # # print('Len actions:', len(decode_actions))
        # decode_action_lens.append(len(decode_actions))
        # non_reduce_decode_action_lens.append(len(list(filter(
        #     lambda x: not isinstance(x, ReduceAction), decode_actions))))

        syntax_tree_prev.reindex_w_dummy_reduce()
        syntax_tree_updated.reindex_w_dummy_reduce()
        previous_code_chunk = _encode(loaded_ast_json['PrevCodeChunkTokens'])
        substitution_system = SubstitutionSystem(transition)
        tgt_edits = substitution_system.get_decoding_edits_fast(syntax_tree_prev, syntax_tree_updated,
                                                                bool_copy_subtree=True,
                                                                init_code_tokens=previous_code_chunk,
                                                                bool_debug=True)

