import json
import gzip
import glob
import re
import io
import os
import codecs
from collections import defaultdict, OrderedDict, Counter, namedtuple
from itertools import chain
from typing import Any

import difflib


class TokenLevelDiffer:
    '''
    Re-render diffs as word-level diffs with the appropriate alignment.
    This class is NOT thread-safe.
    '''

    def __init__(self):
        self.__matcher = difflib.SequenceMatcher()

    tokenization_regex = re.compile('((?!\W)\w+)|(\w+(?=\W))')

    TokenDiff = namedtuple('TokenDiff', ['added', 'removed', 'same', 'has_comment', 'diff_lines'])

    def __tokenize(self, code: str):
        '''Tokenize code in a naive way: split on spaces and at the intersection of alphanumeric characters with other
        characters. This is necessarily noisy, but is universal.'''
        for token in TokenLevelDiffer.tokenization_regex.split(code):
            if token is not None and len(token) > 0:
                yield token

    def __assert_correct_size(self):
        assert len(self.add_list) == len(self.remove_list) == len(self.same_list) == len(self.diff_line_numbers), \
            (len(self.add_list), len(self.remove_list), len(self.same_list), len(self.diff_line_numbers))

    def __consolidate_changed_region_buffers(self):
        self.__matcher.set_seqs(self.removed_buffer, self.added_buffer)
        for tag, i1, i2, j1, j2 in self.__matcher.get_opcodes():
            if tag == 'equal':
                assert i2 - i1 == j2 - j1
                padding = [None] * (i2 - i1)
                self.add_list.extend(padding)
                self.remove_list.extend(padding)
                self.same_list.extend(self.added_buffer[j1:j2])
                self.has_comment.extend(a_has_comment or b_had_comment for a_has_comment, b_had_comment in
                                        zip(self.removed_has_comment_buffer[i1: i2],
                                            self.added_has_comment_buffer[j1: j2]))
                self.diff_line_numbers.extend((k1, k2) for k1, k2 in zip(self.removed_diff_line_number_buffer[i1:i2],
                                                                         self.added_diff_line_number_buffer[j1:j2]))
                self.__assert_correct_size()
            else:
                self.__assert_correct_size()
                max_change_size = max(i2 - i1, j2 - j1)
                self.same_list.extend([None] * max_change_size)
                self.add_list.extend(self.added_buffer[j1:j2])
                self.add_list.extend([None] * (max_change_size - (j2 - j1)))
                self.remove_list.extend(self.removed_buffer[i1:i2])
                self.remove_list.extend([None] * (max_change_size - (i2 - i1)))
                comment_data = [False] * max_change_size
                line_data = [list() for k in range(max_change_size)]
                for k, (com_data, line_num) in enumerate(
                        zip(self.added_has_comment_buffer[j1:j2], self.added_diff_line_number_buffer[j1:j2])):
                    comment_data[k] |= com_data
                    line_data[k].append(line_num)
                for k, (com_data, line_num) in enumerate(
                        zip(self.removed_has_comment_buffer[i1:i2], self.removed_diff_line_number_buffer[i1:i2])):
                    comment_data[k] |= com_data
                    line_data[k].append(line_num)

                self.has_comment.extend(comment_data)
                self.diff_line_numbers.extend(tuple(k) for k in line_data)
                self.__assert_correct_size()

        # Clean-up buffers
        self.added_buffer = []
        self.added_diff_line_number_buffer = []
        self.removed_buffer = []
        self.has_comment_buffer = []
        self.removed_diff_line_number_buffer = []

    def unified_format(self, diff_file):
        self.add_list = []
        self.remove_list = []
        self.same_list = []
        self.has_comment = []
        self.diff_line_numbers = []  # type: List[Tuple]

        self.added_buffer = []
        self.added_has_comment_buffer = []
        self.added_diff_line_number_buffer = []
        self.removed_buffer = []
        self.removed_has_comment_buffer = []
        self.removed_diff_line_number_buffer = []

        for i, line in enumerate(diff_file['diff'].split('\n')):
            self.__assert_correct_size()
            line_has_comment = False  # str(i) in diff_file['comments']

            if line.startswith('@'):  # Ignore diff header
                continue
            elif line.startswith('+'):
                change_type = 0
                line = line[1:]
            elif line.startswith('-'):
                change_type = 1
                line = line[1:]
            else:
                assert line[0] == ' '
                line = line[1:]
                change_type = 2

            # tokenized_line = list(self.__tokenize(line))
            # tokenized_line = ([line] if len(line.strip()) > 0 else []) + ['\n']
            tokenized_line = [line]   # FIXME: revise this hack!
            if change_type == 2 and len(self.added_buffer) + len(self.removed_buffer) > 0:
                self.__consolidate_changed_region_buffers()

                # Now add current change!
            if change_type == 0:
                self.added_buffer.extend(tokenized_line)
                self.added_has_comment_buffer.extend([line_has_comment] * len(tokenized_line))
                self.added_diff_line_number_buffer.extend([i] * len(tokenized_line))
            elif change_type == 1:
                self.removed_buffer.extend(tokenized_line)
                self.removed_has_comment_buffer.extend([line_has_comment] * len(tokenized_line))
                self.removed_diff_line_number_buffer.extend([i] * len(tokenized_line))
            else:
                self.__assert_correct_size()
                self.same_list.extend(tokenized_line)
                padding = [None] * len(tokenized_line)
                self.remove_list.extend(padding)
                self.add_list.extend(padding)
                self.has_comment.extend([line_has_comment] * len(tokenized_line))
                self.diff_line_numbers.extend([(i,)] * len(tokenized_line))
                self.__assert_correct_size()

        if len(self.added_buffer) + len(self.removed_buffer) > 0:
            self.__consolidate_changed_region_buffers()

        self.__assert_correct_size()
        return self.TokenDiff(self.add_list, self.remove_list, self.same_list, self.has_comment, self.diff_line_numbers)


if __name__ == '__main__':
    differ = TokenLevelDiffer()
    diff_dict = {"diff": """@@ -1,4 +1,3 @@
 01
-02
+05
-03
 04
"""}

    diff_result = differ.unified_format(diff_dict)
    pass