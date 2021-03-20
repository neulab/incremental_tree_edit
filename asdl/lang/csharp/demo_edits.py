import json
import pickle
import os
import numpy as np
from tqdm import tqdm
import sys

from asdl.hypothesis import ApplyRuleAction, GenTokenAction, ReduceAction
from asdl.lang.csharp.csharp_hypothesis import CSharpHypothesis
from asdl.lang.csharp.csharp_transition import CSharpTransitionSystem
from asdl.lang.csharp.csharp_grammar import CSharpASDLGrammar
from edit_components.dataset import ChangeExample, DataSet

from trees.substitution_system import SubstitutionSystem, AddSubtree

BOOL_COPY_SUBTREE=True
MEMORY_SPACE='all_init'
MEMORY_ENCODE='joint'


def _encode(word_list):
    return [w.replace('\n', '-NEWLINE-') for w in word_list]


if __name__ == '__main__':
    # csharp_grammar_text = request.urlopen('https://raw.githubusercontent.com/dotnet/roslyn/master/src/Compilers'
    #                                       '/CSharp/Portable/Syntax/Syntax.xml').read()
    csharp_grammar_text = open('Syntax.xml').read()
    fields_to_ignore = ['SemicolonToken', 'OpenBraceToken', 'CloseBraceToken', 'CommaToken', 'ColonToken',
                        'StartQuoteToken', 'EndQuoteToken', 'OpenBracketToken', 'CloseBracketToken', 'NewKeyword']

    grammar = CSharpASDLGrammar.from_roslyn_xml(csharp_grammar_text, pruning=True)

    open('grammar.json', 'w').write(grammar.to_json())

    decode_action_lens, non_reduce_decode_action_lens = [], []
    edit_lens = []
    edit_w_cp_lens, edit_wo_cp_lens = [], []
    examples = []

    ast_json_lines = open('../../../githubedits_data/release/data/githubedits.dev.jsonl').readlines()
    # ast_json_lines = open('../../../githubedits_data/release/data/csharp_fixers.jsonl').readlines()
    for ast_json_idx in tqdm(range(len(ast_json_lines))):
        ast_json = ast_json_lines[ast_json_idx]
        loaded_ast_json = json.loads(ast_json)

        ast_json_obj_prev = loaded_ast_json['PrevCodeAST']
        syntax_tree_prev = grammar.get_ast_from_json_obj(ast_json_obj_prev)
        ast_root_prev = syntax_tree_prev.root_node
        # print(loaded_ast_json['PrevCodeChunk'])
        # print(ast_root_prev.to_string(), "\n")
        # print(ast_root_prev.size)

        ast_json_obj_updated = loaded_ast_json['UpdatedCodeAST']
        syntax_tree_updated = grammar.get_ast_from_json_obj(ast_json_obj_updated)
        ast_root_updated = syntax_tree_updated.root_node
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

        substitution_system = SubstitutionSystem(transition)
        tgt_edits = substitution_system.get_decoding_edits_fast(syntax_tree_prev, syntax_tree_updated,
                                                                bool_copy_subtree=BOOL_COPY_SUBTREE,
                                                                memory_space=MEMORY_SPACE,
                                                                memory_encode=MEMORY_ENCODE,
                                                                bool_debug=True)

        # entry = loaded_ast_json
        # previous_code_chunk = _encode(entry['PrevCodeChunkTokens'])
        # updated_code_chunk = _encode(entry['UpdatedCodeChunkTokens'])
        # context = _encode(entry['PrecedingContext'] + ['|||'] + entry['SucceedingContext'])
        # example = ChangeExample(id=entry['Id'],
        #                         prev_data=previous_code_chunk,
        #                         updated_data=updated_code_chunk,
        #                         raw_prev_data=entry['PrevCodeChunk'],
        #                         raw_updated_data=entry['UpdatedCodeChunk'],
        #                         context=context,
        #                         prev_code_ast=syntax_tree_prev,
        #                         updated_code_ast=syntax_tree_updated,
        #                         tgt_actions=tgt_edits)
        # examples.append(example)

        # print('Len edits:', len(tgt_edits))
        edit_lens.append(len(tgt_edits))

        # bool_cp = any(isinstance(edit, AddSubtree) for edit in tgt_edits)
        # if bool_cp:
        #     edit_w_cp_lens.append(len(tgt_edits))
        #     tgt_edits_wo_cp = substitution_system.get_edits(syntax_tree_prev, syntax_tree_updated)
        #     edit_wo_cp_lens.append(len(tgt_edits_wo_cp))

    print("#of examples:", len(edit_lens))
    # print("Avg action len: %.3f" % np.average(decode_action_lens))
    # print("Avg non-reduce action len: %.3f" % np.average(non_reduce_decode_action_lens))
    print("Avg edit len: %.3f" % np.average(edit_lens))

    # if BOOL_COPY_SUBTREE:
    #     print("#Seq with subtree copy: %d (for this subset: Avg edit len with copy = %.3f, without copy = %.3f)" % (
    #         len(edit_w_cp_lens), np.average(edit_w_cp_lens), np.average(edit_wo_cp_lens)))

    # data_set = DataSet([e for e in examples if e])
    # pickle.dump(data_set, open("../../../githubedits_data/release/data/csharp_fixers.pkl", "wb"))
