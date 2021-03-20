from typing import List
import difflib

from edit_components.diff_utils import TokenLevelDiffer


class ChangeExample:
    def __init__(self, prev_data: List[str], updated_data: List[str], context: List[str],
                 raw_prev_data: str=None, raw_updated_data: str=None,
                 id: str='default_id', **kwargs):
        self.id = id

        self.prev_data = prev_data
        self.updated_data = updated_data

        self.raw_prev_data = raw_prev_data
        self.raw_updated_data = raw_updated_data

        self.context = context

        diff_hunk = '\n'.join(list(x.strip('\n') if x.startswith('@') else x
                                   for x in difflib.unified_diff(a=prev_data, b=updated_data,
                                                                 n=len(self.prev_data) + len(self.updated_data),
                                                                 lineterm=''))[2:])
        self.diff_hunk = diff_hunk

        self._init_change_seq()

        self.__dict__.update(kwargs)

    def _init_change_seq(self):
        differ = TokenLevelDiffer()
        diff_result = differ.unified_format(dict(diff=self.diff_hunk))
        change_seq = []

        prev_token_ptr = updated_token_ptr = 0
        for i, (added, removed, same) in enumerate(zip(diff_result.added, diff_result.removed, diff_result.same)):
            if same is not None:
                tag = 'SAME'
                token = same

                assert self.prev_data[prev_token_ptr] == self.updated_data[updated_token_ptr] == token

                prev_token_ptr += 1
                updated_token_ptr += 1
            elif added is not None and removed is not None:
                tag = 'REPLACE'
                token = (removed, added)

                assert self.prev_data[prev_token_ptr] == removed
                assert self.updated_data[updated_token_ptr] == added

                prev_token_ptr += 1
                updated_token_ptr += 1
            elif added is not None and removed is None:
                tag = 'ADD'
                token = added

                assert self.updated_data[updated_token_ptr] == added

                updated_token_ptr += 1
            elif added is None and removed is not None:
                tag = 'DEL'
                token = removed

                assert self.prev_data[prev_token_ptr] == removed

                prev_token_ptr += 1
            else:
                raise ValueError('unknown change entry')

            change_seq.append((tag, token))

        setattr(self, 'change_seq', change_seq)
