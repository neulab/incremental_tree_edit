import sys
from urllib import request

from asdl.asdl_ast import AbstractSyntaxTree, AbstractSyntaxNode
from edit_components.dataset import DataSet


class ChangeGraph(object):
    def __init__(self):
        pass

    @staticmethod
    def build_change_graph(old_ast: AbstractSyntaxTree, new_ast: AbstractSyntaxTree):
        equality_links = []

        def _modify_id(node, prefix=''):
            node.id = f'{prefix}-{node.id}'
            if isinstance(node, AbstractSyntaxNode):
                for field in node.fields:
                    for field_val in field.as_value_list:
                        _modify_id(field_val, prefix)

        old_ast_root_copy = old_ast.root_node.copy()
        _modify_id(old_ast_root_copy, 'old')

        new_ast_root_copy = new_ast.root_node.copy()
        _modify_id(new_ast_root_copy, 'new')

        old_ast = AbstractSyntaxTree(old_ast_root_copy)
        new_ast = AbstractSyntaxTree(new_ast_root_copy)

        def _search_common_sub_tree(tgt_ast_node):
            node_query_result = old_ast.find_node(tgt_ast_node)
            if node_query_result:
                src_node_id, src_node = node_query_result
                tgt_ast_node.parent_field.replace(tgt_ast_node, src_node)

                # register this link
                equality_links.append((tgt_ast_node.id, src_node.id))
            else:
                for field in tgt_ast_node.fields:
                    if field.type.is_composite:
                        for field_val in field.as_value_list:
                            _search_common_sub_tree(field_val)

        _search_common_sub_tree(new_ast.root_node)

        visited = set()
        adjacency_list = []

        def _visit(node, parent_node):
            if parent_node:
                adjacency_list.append((parent_node.id, node.id))

            if node.id in visited:
                return

            if isinstance(node, AbstractSyntaxNode):
                for field in node.fields:
                    for field_val in field.as_value_list:
                        _visit(field_val, node)

            visited.add(node.id)

        _visit(old_ast.root_node, None)
        _visit(new_ast.root_node, None)
        pass


if __name__ == '__main__':
    dataset_path = 'data/commit_files.from_repo.processed.071009.jsonl.dev.top100'
    grammar_text = request.urlopen('https://raw.githubusercontent.com/dotnet/roslyn/master/src/Compilers'
                                   '/CSharp/Portable/Syntax/Syntax.xml').read()

    from asdl.lang.csharp.csharp_grammar import CSharpASDLGrammar
    from asdl.lang.csharp.csharp_transition import CSharpTransitionSystem

    grammar = CSharpASDLGrammar.from_roslyn_xml(grammar_text, pruning=True)
    transition_system = CSharpTransitionSystem(grammar)

    print('Loading datasets...', file=sys.stderr)
    dataset = DataSet.load_from_jsonl(dataset_path, type='tree2tree_subtree_copy',
                                      transition_system=transition_system,
                                      parallel=False,
                                      debug=True)
    for example in dataset:
        list1 = list(example.prev_code_ast.id2node.values())
        list2 = list(example.prev_code_ast.root_node.descendant_nodes_and_tokens)

        assert list1 == list2

        ChangeGraph.build_change_graph(example.prev_code_ast, example.updated_code_ast)
