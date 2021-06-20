# coding=utf-8
import re
from collections import OrderedDict
from itertools import chain
from typing import Dict, Union

from bs4 import BeautifulSoup
import json

from asdl.asdl import ASDLGrammar, ASDLType, ASDLProduction, ASDLConstructor, Field


class CSharpASDLGrammar(ASDLGrammar):
    """
    Collection of types, constructors and productions
    """
    def __init__(self, productions, root_type):
        super().__init__(productions, root_type, language='csharp')

        self._primitive_types = [type for type in self.types if type not in self.type2productions and type.is_leaf]
        for type in self._primitive_types:
            type.is_composite = False

        self._composite_types = [type for type in self.types if type not in self.primitive_types]
        for type in self._composite_types:
            type.is_composite = True

    @property
    def primitive_types(self):
        return self._primitive_types

    @property
    def composite_types(self):
        return self._composite_types

    def to_json(self):
        grammar_rules = []
        for prod in self.productions:
            entry = dict(constructor=prod.constructor.name,
                         fields=[dict(name=f.name, type=f.type.name) for f in prod.constructor.fields])
            grammar_rules.append(entry)

        return json.dumps(grammar_rules, indent=2)

    @classmethod
    def from_roslyn_xml(cls, xml_text, pruning=False):
        bs = BeautifulSoup(xml_text, 'xml')
        token_kinds_to_keep = {'NumericLiteralToken', 'StringLiteralToken', 'CharacterLiteralToken'}

        from bs4 import Tag

        all_types = dict()
        productions = []
        generic_list_productions = set()

        # add base type
        grammar_root_type = ASDLType('SyntaxNode')
        all_types[grammar_root_type.name] = grammar_root_type

        for node in bs.Tree.find_all(lambda x: isinstance(x, Tag), recursive=False):
            # process type information
            base_type_name = node['Base']
            if base_type_name not in all_types:
                all_types[base_type_name] = ASDLType(base_type_name)
            base_type = all_types[base_type_name]

            node_name = node['Name']
            if node_name in all_types:
                node_type = all_types[node_name]
                if node_type not in base_type.child_types:
                    base_type.add_child(node_type)
            else:
                node_type = ASDLType(node_name, parent_type=base_type)
                all_types[node_type.name] = node_type

            if node.name == 'Node':
                fields = []
                for field_node in node.find_all('Field', recursive=False):
                    field_name = field_node['Name']
                    field_type_str = field_node['Type']

                    field_kinds = set(kind['Name'] for kind in field_node.find_all('Kind'))

                    if pruning:
                        # if field_type_str == 'SyntaxToken' and (field_name not in {'Identifier', 'OperatorToken'} and #!= 'Identifier'
                        #                                         len(field_kinds.intersection(token_kinds_to_keep)) == 0):
                        #     continue

                        if field_type_str == 'SyntaxToken' and \
                                field_name != 'Identifier' and \
                                not (field_name == 'OperatorToken' and node_name in {'BinaryExpressionSyntax', 'AssignmentExpressionSyntax',
                                                                                     'PostfixUnaryExpressionSyntax', 'PrefixUnaryExpressionSyntax'}) and \
                                not (field_name == 'Keyword' and node_name == 'PredefinedTypeSyntax') and \
                                len(field_kinds.intersection(token_kinds_to_keep)) == 0:
                            continue

                    if field_type_str not in all_types:
                        all_types[field_type_str] = ASDLType(field_type_str)
                    field_type = all_types[field_type_str]

                    if 'SyntaxList' in field_type_str:
                        base_type_name = re.match('\w+<(.*?)>', field_type_str).group(1)
                        if base_type_name not in all_types:
                            all_types[base_type_name] = ASDLType(base_type_name)
                        base_type = all_types[base_type_name]

                        production = ASDLProduction(field_type,
                                                    ASDLConstructor(field_type.name, fields=[
                                                        Field('Element', base_type, 'multiple')]))
                        generic_list_productions.add(production)

                    field_cardinality = 'optional' if field_node.get('Optional', None) == 'true' else 'single'
                    field = Field(field_name, field_type, field_cardinality)
                    fields.append(field)

                constructor = ASDLConstructor(node['Name'], fields)
                production = ASDLProduction(node_type, constructor)
                productions.append(production)

        productions.extend(generic_list_productions)
        grammar = CSharpASDLGrammar(productions, root_type=all_types['CSharpSyntaxNode'])

        return grammar

    def get_ast_from_json_str(self, json_str):
        json_obj = json.loads(json_str)

        return self.get_ast_from_json_obj(json_obj)

    def convert_ast_into_json_obj(self, ast_node):
        from asdl.asdl_ast import AbstractSyntaxNode, RealizedField, SyntaxToken, AbstractSyntaxTree

        if isinstance(ast_node, SyntaxToken):
            entry = OrderedDict(Constructor='SyntaxToken',
                                Value=ast_node.value,
                                Position=-1)
        else:
            entry_fields = dict()
            for realized_field in ast_node.fields:
                field = realized_field.field

                if 'SyntaxList' in field.type.name:
                    child_entry = []
                    # SyntaxList<T> -> (T* Element)
                    field_elements = realized_field.value.fields[0].as_value_list

                    for field_element_ast in field_elements:
                        element_ast = self.convert_ast_into_json_obj(field_element_ast)
                        child_entry.append(element_ast)
                elif realized_field.value is not None:
                    child_entry = self.convert_ast_into_json_obj(realized_field.value)
                else:
                    child_entry = None

                entry_fields[field.name] = child_entry

            constructor_name = ast_node.production.constructor.name
            entry = OrderedDict(Constructor=constructor_name,
                                Fields=entry_fields)

        return entry

    def get_ast_from_json_obj(self, json_obj: Dict):
        """read an AST from serialized JSON string"""
        # FIXME: cyclic import
        from asdl.asdl_ast import AbstractSyntaxNode, RealizedField, SyntaxToken, AbstractSyntaxTree

        def get_subtree(entry, parent_field, next_available_id):
            if entry is None:
                return None, next_available_id

            constructor_name = entry['Constructor']

            # terminal case
            if constructor_name == 'SyntaxToken':
                if entry['Value'] is None:
                    return None, next_available_id  # return None for optional field whose value is null

                token = SyntaxToken(parent_field.type, entry['Value'], position=entry['Position'], id=next_available_id)
                next_available_id += 1

                return token, next_available_id

            field_entries = entry['Fields']
            node_id = next_available_id
            next_available_id += 1
            prod = self.get_prod_by_ctr_name(constructor_name)
            realized_fields = []
            for field in prod.constructor.fields:
                field_value = field_entries[field.name]

                if isinstance(field_value, list):
                    assert 'SyntaxList' in field.type.name

                    sub_ast_id = next_available_id
                    next_available_id += 1

                    sub_ast_prod = self.get_prod_by_ctr_name(field.type.name)
                    sub_ast_constr_field = sub_ast_prod.constructor.fields[0]
                    sub_ast_field_values = []
                    for field_child_entry in field_value:
                        child_sub_ast, next_available_id = get_subtree(field_child_entry, sub_ast_constr_field, next_available_id=next_available_id)
                        sub_ast_field_values.append(child_sub_ast)

                    sub_ast = AbstractSyntaxNode(sub_ast_prod,
                                                 [RealizedField(sub_ast_constr_field,
                                                                sub_ast_field_values)],
                                                 id=sub_ast_id)

                    # FIXME: have a global mark_finished method!
                    for sub_ast_field in sub_ast.fields:
                        if sub_ast_field.cardinality in ('multiple', 'optional'):
                            sub_ast_field._not_single_cardinality_finished = True

                    realized_field = RealizedField(field, sub_ast)
                else:
                    # if the child is an AST or terminal SyntaxNode
                    sub_ast, next_available_id = get_subtree(field_value, field, next_available_id)
                    realized_field = RealizedField(field, sub_ast)

                realized_fields.append(realized_field)

            ast_node = AbstractSyntaxNode(prod, realized_fields, id=node_id)
            for field in ast_node.fields:
                if field.cardinality in ('multiple', 'optional'):
                    field._not_single_cardinality_finished = True

            return ast_node, next_available_id

        ast_root, _ = get_subtree(json_obj, parent_field=None, next_available_id=0)
        ast = AbstractSyntaxTree(ast_root)

        return ast
