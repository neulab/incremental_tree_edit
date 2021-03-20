# coding=utf-8
import json

import ast
from asdl.lang.py3.py3_transition_system import Python3TransitionSystem, python_ast_to_asdl_ast
from asdl.hypothesis import *
import astor
from asdl.asdl_ast import AbstractSyntaxTree

from trees.substitution_system import SubstitutionSystem
from trees.utils import stack_subtrees

BOOL_COPY_SUBTREE=True

if __name__ == '__main__':
    # read in the grammar specification of Python 2.7, defined in ASDL
    asdl_text = open('py3_asdl.simplified.txt').read()
    grammar = ASDLGrammar.from_text(asdl_text)

    transition_system = Python3TransitionSystem(grammar)
    substitution_system = SubstitutionSystem(transition_system)

    train_data = open('../../../conala_retrieval_data/data/train_ret_neural.jsonl', 'r').readlines()
    for line in train_data:
        json_item = json.loads(line.strip())
        tgt_code = json_item['snippet']
        tgt_code_ast = ast.parse(tgt_code)
        tgt_asdl_ast = AbstractSyntaxTree(python_ast_to_asdl_ast(tgt_code_ast.body[0], grammar))
        tgt_asdl_ast = tgt_asdl_ast.copy_and_reindex_w_dummy_reduce()

        memory = None
        if BOOL_COPY_SUBTREE:
            memory = set(stack_subtrees(tgt_asdl_ast.root_node, bool_repr=True))

        retrieval_results = json_item['retrieval_results']
        for ret_item in retrieval_results:
            ret_code = ret_item['snippet']
            ret_code_ast = ast.parse(ret_code)
            ret_asdl_ast = AbstractSyntaxTree(python_ast_to_asdl_ast(ret_code_ast.body[0], grammar))
            ret_asdl_ast = ret_asdl_ast.copy_and_reindex_w_dummy_reduce()

            edit_length, _, _ = substitution_system.ast_tree_compare(
                ret_asdl_ast.root_node, tgt_asdl_ast.root_node, memory=memory)
