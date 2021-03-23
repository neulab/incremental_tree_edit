# coding=utf-8
from collections import OrderedDict, Counter
from itertools import chain

from .utils import remove_comment


class ASDLGrammar(object):
    """
    Collection of types, constructors and productions
    """
    def __init__(self, productions, root_type, language):
        self.language = language

        # productions are indexed by their head types
        self.productions = sorted(productions, key=lambda x: repr(x))

        self.type2productions = dict()
        self._constructor_production_map = dict()
        for prod in productions:
            cur_type = prod.type
            while cur_type:
                self.type2productions.setdefault(cur_type, []).append(prod)
                cur_type = cur_type.parent_type

            self._constructor_production_map[prod.constructor.name] = prod

        # number of productions
        self.size = len(productions)

        # get entities to their ids map
        self.prod2id = {prod: i for i, prod in enumerate(self.productions)}
        self.type2id = {type: i for i, type in enumerate(self.types)}
        self.field2id = {field: i for i, field in enumerate(self.fields)}
        self.id2prod = {i: prod for i, prod in enumerate(self.productions)}
        self.id2type = {i: type for i, type in enumerate(self.types)}
        self.id2field = {i: field for i, field in enumerate(self.fields)}

        # field is indexed by its production and its field name
        self.prod_field2id = {(prod, field): i for i, (prod, field) in enumerate(self.production_and_fields)}
        self.id2prod_field = {i: (prod, field) for (prod, field), i in self.prod_field2id.items()}

        # get the root type
        self.root_type = root_type

        # get primitive types
        self.primitive_types = [type for type in self.types if type not in self.type2productions and type.is_leaf]
        for type in self.primitive_types:
            type.is_composite = False
        self.composite_types = [type for type in self.types if type not in self.primitive_types]

    def __len__(self):
        return self.size

    def __getitem__(self, datum):
        # get all descendant productions given a type (string)
        if isinstance(datum, str):
            return self.type2productions[ASDLType(datum)]
        elif isinstance(datum, ASDLType):
            return self.type2productions[datum]

    def get_prod_by_ctr_name(self, name):
        return self._constructor_production_map[name]

    def get_constructor_by_name(self, name):
        return self._constructor_production_map[name].constructor

    @property
    def types(self):
        if not hasattr(self, '_types'):
            all_types = set()
            for prod in self.productions:
                all_types.add(prod.type)
                all_types.update(map(lambda x: x.type, prod.constructor.fields))

            self._types = sorted(all_types, key=lambda x: x.name)

        return self._types

    @property
    def descendant_types(self):
        if not hasattr(self, '_descendant_types'):
            self._descendant_types = dict()
            for parent_type, prods in self.type2productions.items():
                self._descendant_types.setdefault(parent_type, set()).update(map(lambda prod: prod.type, prods))

        return self._descendant_types

    @property
    def fields(self):
        if not hasattr(self, '_fields'):
            all_fields = set()
            for prod in self.productions:
                all_fields.update(prod.constructor.fields)

            self._fields = sorted(all_fields, key=lambda x: x.name)

        return self._fields

    @property
    def production_and_fields(self):
        if not hasattr(self, '_prod_and_fields'):
            all_fields = set()
            for prod in self.productions:
                for field in prod.constructor.fields:
                    all_fields.add((prod, field))

            self._prod_and_fields = sorted(all_fields, key=lambda x: (x[0].type.name, x[0].constructor.name, x[1].name))

        return self._prod_and_fields

    def is_composite_type(self, asdl_type):
        return asdl_type in self.composite_types and asdl_type.is_composite

    def is_primitive_type(self, asdl_type):
        return asdl_type in self.primitive_types

    @staticmethod
    def from_text(text, language):
        def _parse_field_from_text(_text):
            d = _text.strip().split(' ')
            name = d[1].strip()
            type_str = d[0].strip()
            cardinality = 'single'
            if type_str[-1] == '*':
                type_str = type_str[:-1]
                cardinality = 'multiple'
            elif type_str[-1] == '?':
                type_str = type_str[:-1]
                cardinality = 'optional'

            if type_str in primitive_type_names:
                return Field(name, ASDLPrimitiveType(type_str), cardinality=cardinality)
            else:
                return Field(name, ASDLCompositeType(type_str), cardinality=cardinality)

        def _parse_constructor_from_text(_text):
            _text = _text.strip()
            fields = None
            if '(' in _text:
                name = _text[:_text.find('(')]
                field_blocks = _text[_text.find('(') + 1:_text.find(')')].split(',')
                fields = map(_parse_field_from_text, field_blocks)
            else:
                name = _text

            if name == '': name = None

            return ASDLConstructor(name, fields)

        lines = remove_comment(text).split('\n')
        lines = list(map(lambda l: l.strip(), lines))
        lines = list(filter(lambda l: l, lines))
        line_no = 0

        # first line is always the primitive types
        primitive_type_names = list(map(lambda x: x.strip(), lines[line_no].split(',')))
        line_no += 1

        all_productions = list()

        while True:
            type_block = lines[line_no]
            type_name = type_block[:type_block.find('=')].strip()
            constructors_blocks = type_block[type_block.find('=') + 1:].split('|')
            i = line_no + 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                t = lines[i].strip()
                cont_constructors_blocks = t[1:].split('|')
                constructors_blocks.extend(cont_constructors_blocks)

                i += 1

            constructors_blocks = filter(lambda x: x and x.strip(), constructors_blocks)

            # parse type name
            new_type = ASDLPrimitiveType(type_name) if type_name in primitive_type_names else ASDLCompositeType(
                type_name)
            constructors = map(_parse_constructor_from_text, constructors_blocks)

            productions = list(map(lambda c: ASDLProduction(new_type, c), constructors))
            all_productions.extend(productions)

            line_no = i
            if line_no == len(lines):
                break

        root_type = all_productions[0].type
        grammar = ASDLGrammar(all_productions, root_type, language)

        return grammar


class ASDLProduction(object):
    def __init__(self, type, constructor):
        self.type = type
        self.constructor = constructor

    @property
    def fields(self):
        return self.constructor.fields

    def __getitem__(self, field_name):
        return self.constructor[field_name]

    def __hash__(self):
        h = hash(self.type) ^ hash(self.constructor)

        return h

    def __eq__(self, other):
        return isinstance(other, ASDLProduction) and \
               self.type == other.type and \
               self.constructor == other.constructor

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '%s -> %s' % (self.type.__repr__(plain=True), self.constructor.__repr__(plain=True))


class ASDLConstructor(object):
    def __init__(self, name, fields=None):
        self.name = name
        self.fields = []
        if fields:
            self.fields = list(fields)

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field_name: return field

        raise KeyError

    def __hash__(self):
        h = hash(self.name)
        for field in self.fields:
            h ^= hash(field)

        return h

    def __eq__(self, other):
        return isinstance(other, ASDLConstructor) and \
               self.name == other.name and \
               self.fields == other.fields

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = '%s(%s)' % (self.name,
                                 ', '.join(f.__repr__(plain=True) for f in self.fields))
        if plain:
            return plain_repr
        else:
            return 'Constructor(%s)' % plain_repr


class Field(object):
    def __init__(self, name, type, cardinality):
        self.name = name
        self.type = type

        assert cardinality in ['single', 'optional', 'multiple']
        self.cardinality = cardinality

    def __hash__(self):
        h = hash(self.name) ^ hash(self.type)
        h ^= hash(self.cardinality)

        return h

    def __eq__(self, other):
        return isinstance(other, Field) and \
               self.name == other.name and \
               self.type == other.type and \
               self.cardinality == other.cardinality

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = '%s%s %s' % (self.type.__repr__(plain=True),
                                  Field.get_cardinality_repr(self.cardinality),
                                  self.name)
        if plain:
            return plain_repr
        else:
            return 'Field(%s)' % plain_repr

    @staticmethod
    def get_cardinality_repr(cardinality):
        return '' if cardinality == 'single' else '?' if cardinality == 'optional' else '*'


class ASDLType(object):
    def __init__(self, type_name, parent_type=None, is_composite=True):
        self.name = type_name
        self.is_composite = is_composite
        self.child_types = []
        self.parent_type = None
        if parent_type:
            parent_type.add_child(self)

    @property
    def is_leaf(self):
        return len(self.child_types) == 0

    def add_child(self, child_type):
        child_type.parent_type = self
        self.child_types.append(child_type)

    # FIXME: for efficiency consideration, we do not use
    # the child information for the following methods
    # we assume type names are unique!
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, ASDLType) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = self.name
        if plain:
            return plain_repr
        else:
            return '%s(%s)' % (self.__class__.__name__, plain_repr)


class ASDLCompositeType(ASDLType):
    pass


class ASDLPrimitiveType(ASDLType):
    pass


if __name__ == '__main__':
    asdl_desc = """
var, ent, num, var_type

expr = Variable(var variable)
| Entity(ent entity)
| Number(num number)
| Apply(pred predicate, expr* arguments)
| Argmax(var variable, expr domain, expr body)
| Argmin(var variable, expr domain, expr body)
| Count(var variable, expr body)
| Exists(var variable, expr body)
| Lambda(var variable, var_type type, expr body)
| Max(var variable, expr body)
| Min(var variable, expr body)
| Sum(var variable, expr domain, expr body)
| The(var variable, expr body)
| Not(expr argument)
| And(expr* arguments)
| Or(expr* arguments)
| Compare(cmp_op op, expr left, expr right)

cmp_op = GreaterThan | Equal | LessThan
"""

    grammar = ASDLGrammar.from_text(asdl_desc, 'py')
    print(ASDLCompositeType('1') == ASDLPrimitiveType('1'))
