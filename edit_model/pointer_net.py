# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size):
        super(PointerNet, self).__init__()

        self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)

    def forward(self, src_encodings, src_token_mask, query_vec,
                log=False, valid_masked_as_one=False, return_logits=False):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(tgt_action_num, batch_size, query_vec_size)
        :return: Variable(tgt_action_num, batch_size, src_sent_len)
        """

        # (batch_size, 1, src_sent_len, query_vec_size)
        src_trans = self.src_encoding_linear(src_encodings).unsqueeze(1)
        # (batch_size, tgt_action_num, query_vec_size, 1)
        q = query_vec.permute(1, 0, 2).unsqueeze(3)

        # (batch_size, tgt_action_num, src_sent_len)
        weights = torch.matmul(src_trans, q).squeeze(3)

        # (tgt_action_num, batch_size, src_sent_len)
        weights = weights.permute(1, 0, 2)

        if src_token_mask is not None:
            # (tgt_action_num, batch_size, src_sent_len)
            if len(src_token_mask.size()) == len(weights.size()) + 1:
                src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)

            if valid_masked_as_one:
                src_token_mask = 1 - src_token_mask

            weights.data.masked_fill_(src_token_mask.bool(), -float('inf')) # byte -> bool, pytorch version upgrade

        if return_logits:
            return weights

        if log:
            ptr_weights = F.log_softmax(weights, dim=-1)
        else:
            ptr_weights = F.softmax(weights, dim=-1)

        return ptr_weights
