# coding=utf-8

import torch
import torch.nn.functional as F
import numpy as np

import torch
from torch.autograd import Variable
import numpy as np


def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None, return_log_att_weight=False):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask.bool(), -float('inf')) # byte -> bool, pytorch version upgrade
    softmaxed_att_weight = F.softmax(att_weight, dim=-1)
    if return_log_att_weight:
        log_softmaxed_att_weight = F.log_softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

    if return_log_att_weight:
        return ctx_vec, softmaxed_att_weight, log_softmaxed_att_weight
    else:
        return ctx_vec, softmaxed_att_weight


def log_sum_exp(inputs, keepdim=False, mask=None):
    """Numerically stable logsumexp on the last dim of `inputs`.
       reference: https://github.com/pytorch/pytorch/issues/2591
    Args:
        inputs: A Variable with any shape.
        keepdim: A boolean.
        mask: A mask variable of type float. It has the same shape as `inputs`.
    Returns:
        Equivalent of log(sum(exp(inputs), keepdim=keepdim)).
    """

    if mask is not None:
        mask = 1. - mask
        max_offset = -1e7 * mask
    else:
        max_offset = 0.

    s, _ = torch.max(inputs + max_offset, dim=-1, keepdim=True)

    inputs_offset = inputs - s
    if mask is not None:
        inputs_offset.masked_fill_(mask.bool(), -float('inf')) # byte -> bool, pytorch version upgrade

    outputs = s + inputs_offset.exp().sum(dim=-1, keepdim=True).log()

    if not keepdim:
        outputs = outputs.squeeze(-1)
    return outputs


def log_softmax(inputs, dim=-1, mask=None):
    if mask is not None:
        inputs.masked_fill_((1 - mask).bool(), -float('inf')) # byte -> bool, pytorch version upgrade

    return F.log_softmax(inputs, dim=dim)


def length_array_to_mask_tensor(length_array, device=None):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    mask = torch.tensor(mask, dtype=torch.uint8, device=device)

    return mask


def pad_lists(indices, pad_id, return_mask=False):
    max_len = max(len(idx_list) for idx_list in indices)
    padded_indices = []
    if return_mask: masks = []
    for idx_list in indices:
        padded_indices.append(idx_list + [pad_id] * (max_len - len(idx_list)))
        if return_mask:
            masks.append([0] * len(idx_list) + [1] * (max_len - len(idx_list)))

    if return_mask: return padded_indices, masks
    return padded_indices


def to_input_variable(sequences, vocab, device, append_boundary_sym=False, return_mask=False, pad_id=-1, batch_first=False):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if append_boundary_sym:
        sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]

    pad_id = pad_id if pad_id >= 0 else vocab['<pad>']

    word_ids = word2id(sequences, vocab)
    if batch_first:
        result = pad_lists(word_ids, pad_id, return_mask=return_mask)
        if return_mask: sents_t, masks = result
        else: sents_t = result
    else:
        sents_t, masks = input_transpose(word_ids, pad_id)

    sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)

    if return_mask:
        mask_var = torch.tensor(masks, dtype=torch.long, device=device)
        return sents_var, mask_var

    return sents_var


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab.id2word[w] for w in s] for s in sents]
    else:
        return [vocab.id2word[w] for w in sents]


def input_transpose(sents, pad_token):
    """
    transform the input List[sequence] of size (batch_size, max_sent_len)
    into a list of size (max_sent_len, batch_size), with proper padding
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return sents_t, masks


def get_sort_map(lens_array):
    """sort input by length in descending order,
    return the sorted index and the mapping between old and new positions"""

    sorted_example_ids = sorted(list(range(len(lens_array))), key=lambda x: -lens_array[x])

    example_old2new_pos_map = [-1] * len(lens_array)
    for new_pos, old_pos in enumerate(sorted_example_ids):
        example_old2new_pos_map[old_pos] = new_pos

    return sorted_example_ids, example_old2new_pos_map


def batch_iter(examples, batch_size, shuffle=False, sort_func=None, return_sort_map=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in range(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]
        # sort by the length of the change sequence in descending order
        if sort_func:
            sorted_examples_with_ids = sorted([(_idx, e) for _idx, e in enumerate(batch_examples)], key=lambda x: sort_func(x[1]))

        if return_sort_map and sort_func:
            sorted_example_ids = [x[0] for x in sorted_examples_with_ids]

            example_old2new_pos_map = [-1] * len(sorted_example_ids)
            for new_pos, old_pos in enumerate(sorted_example_ids):
                example_old2new_pos_map[old_pos] = new_pos

            sorted_examples = [x[1] for x in sorted_examples_with_ids]
            yield sorted_examples, sorted_example_ids, example_old2new_pos_map
        else:
            yield batch_examples


def anonymize_unk_tokens(prev_code, updated_code, context, vocab):
    unk_name_map = dict()

    def __to_new_token_seq(tokens):
        for token in tokens:
            if token in unk_name_map:
                new_token_name = unk_name_map[token]
            else:
                if vocab.is_unk(token):
                    new_token_name = 'UNK_%d' % len(unk_name_map)
                    unk_name_map[token] = new_token_name
                else:
                    new_token_name = token

            yield new_token_name

    new_prev_code = list(__to_new_token_seq(prev_code))
    new_context = list(__to_new_token_seq(context))
    new_updated_code = list(__to_new_token_seq(updated_code))

    return new_prev_code, new_updated_code, new_context

