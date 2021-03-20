from collections import namedtuple

EncodingResult = namedtuple('EncodingResult', ['data', 'encoding', 'last_state', 'last_cell', 'mask'])
