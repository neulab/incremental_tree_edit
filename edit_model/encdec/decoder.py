import re

import torch.nn as nn


class Decoder(nn.Module):
    VALID_IDENTIFIER_RE = re.compile(r'^[A-Za-z0-9_-]*$')

    def __init__(self):
        super(Decoder, self).__init__()

    @staticmethod
    def _can_only_generate_this_token(token):
        # remove the BPE delimiter
        if token.startswith('\u2581'):
            token = token[1:]

        return not Decoder.VALID_IDENTIFIER_RE.match(token)
