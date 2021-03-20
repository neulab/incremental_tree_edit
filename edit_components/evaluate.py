import torch
import numpy as np
from edit_model.editor import Graph2IterEditEditor, Graph2TreeEditor


def evaluate_nll(model, test_set, batch_size=32, change_vectors=None, return_nll_list=False,
                 return_details=False):
    was_training = model.training
    model.eval()

    cum_nll = 0.
    cum_ppl = 0.
    cum_examples = 0.

    cum_other_nll_per_cat = dict()
    cum_examples_per_cat = dict()
    cum_nll_per_cat_by_step = dict()
    cum_examples_per_cat_by_step = dict()

    nll_dict = dict()

    if change_vectors is not None:
        assert isinstance(model, (Graph2IterEditEditor, Graph2TreeEditor))

    with torch.no_grad():
        for batch_examples in test_set.batch_iter(batch_size, shuffle=False):
            batch_change_vectors = None
            if change_vectors is not None:
                batch_change_vectors = torch.stack([change_vectors[e.change_vec_idx] for e in batch_examples])

            # neg_log_probs = -model(batch_examples)['log_probs']
            if isinstance(model, (Graph2IterEditEditor, Graph2TreeEditor)):
                results = model(batch_examples, change_vectors=batch_change_vectors)
            else:
                results = model(batch_examples)

            neg_log_probs = - results['log_probs']
            batch_code_tokens_num = torch.tensor([len(e.updated_data) for e in batch_examples],
                                                 dtype=torch.float,
                                                 device=neg_log_probs.device)

            batch_nlls = neg_log_probs.cpu().numpy()
            batch_ppls = (neg_log_probs / batch_code_tokens_num).cpu().numpy()
            for batch_id in range(len(batch_examples)):
                nll_dict[batch_examples[batch_id].id] = batch_nlls[batch_id]

            cum_ppl += batch_ppls.sum()
            cum_nll += batch_nlls.sum()
            cum_examples += len(batch_examples)

            del neg_log_probs

            if isinstance(model, Graph2IterEditEditor) and return_details:
                log_probs = results['ungated_log_probs']
                batch_edit_mask = results['batch_edit_mask']

                tgt_op_log_probs = (results['tgt_op_log_probs'] * batch_edit_mask).sum(dim=0)
                tgt_op_mask = batch_edit_mask.sum(dim=0)
                tgt_op_log_probs_by_step = torch.unbind((results['tgt_op_log_probs'] * batch_edit_mask).sum(dim=1), dim=0)
                tgt_op_mask_by_step = torch.unbind(batch_edit_mask.sum(dim=1), dim=0)
                results.update({'tgt_op_log_probs': tgt_op_log_probs,
                                'tgt_op_mask': tgt_op_mask,
                                'tgt_op_log_probs_by_step': tgt_op_log_probs_by_step,
                                'tgt_op_mask_by_step': tgt_op_mask_by_step})

                if 'tgt_node_log_probs' in results:
                    tgt_node_log_probs = results['tgt_node_log_probs']
                    node_selection_mask = results['node_selection_mask']
                    tgt_node_log_probs_by_step = torch.unbind((tgt_node_log_probs * node_selection_mask).sum(dim=1), dim=0)
                    node_selection_mask_by_step = torch.unbind(node_selection_mask.sum(dim=1), dim=0)
                    results.update({'tgt_node_log_probs': (tgt_node_log_probs * node_selection_mask).sum(dim=0),
                                    'node_selection_mask': node_selection_mask.sum(dim=0),
                                    'tgt_node_log_probs_by_step': tgt_node_log_probs_by_step,
                                    'node_selection_mask_by_step': node_selection_mask_by_step})

                if 'tgt_add_log_probs' in results:
                    tgt_add_log_probs = results['tgt_add_log_probs']
                    tgt_add_operator_mask = results['tgt_add_operator_mask']
                    tgt_add_log_probs_by_step = torch.unbind((tgt_add_log_probs * tgt_add_operator_mask).sum(dim=1), dim=0)
                    tgt_add_operator_mask_by_step = torch.unbind(tgt_add_operator_mask.sum(dim=1), dim=0)
                    results.update({'tgt_add_log_probs': (tgt_add_log_probs * tgt_add_operator_mask).sum(dim=0),
                                    'tgt_add_operator_mask': tgt_add_operator_mask.sum(dim=0),
                                    'tgt_add_log_probs_by_step': tgt_add_log_probs_by_step,
                                    'tgt_add_operator_mask_by_step': tgt_add_operator_mask_by_step})

                if 'tgt_add_subtree_log_probs' in results:
                    tgt_add_subtree_log_probs = results['tgt_add_subtree_log_probs']
                    tgt_add_subtree_operator_mask = results['tgt_add_subtree_operator_mask']
                    tgt_add_subtree_log_probs_by_step = torch.unbind((tgt_add_subtree_log_probs * tgt_add_subtree_operator_mask).sum(dim=1), dim=0)
                    tgt_add_subtree_operator_mask_by_step = torch.unbind(tgt_add_subtree_operator_mask.sum(dim=1), dim=0)
                    results.update({'tgt_add_subtree_log_probs': (tgt_add_subtree_log_probs * tgt_add_subtree_operator_mask).sum(dim=0),
                                    'tgt_add_subtree_operator_mask': tgt_add_subtree_operator_mask.sum(dim=0),
                                    'tgt_add_subtree_log_probs_by_step': tgt_add_subtree_log_probs_by_step,
                                    'tgt_add_subtree_operator_mask_by_step': tgt_add_subtree_operator_mask_by_step})

                keys = ['tgt_op_log_probs', 'tgt_node_log_probs', 'tgt_add_log_probs', 'tgt_add_subtree_log_probs']
                key_masks = ['tgt_op_mask', 'node_selection_mask', 'tgt_add_operator_mask', 'tgt_add_subtree_operator_mask']
                for key_idx, key in enumerate(keys):
                    if key in results:
                        _neg_log_probs = - results[key].cpu().numpy().sum()
                        cum_other_nll_per_cat[key] = cum_other_nll_per_cat.get(key, 0.) + _neg_log_probs

                        _count_examples = results[key_masks[key_idx]].cpu().numpy().sum()
                        cum_examples_per_cat[key] = cum_examples_per_cat.get(key, 0.) + _count_examples

                        key_by_step = key + "_by_step"
                        key_mask_by_step = key_masks[key_idx] + "_by_step"
                        _log_probs_by_step = results[key_by_step]  # a list of summed log_prob
                        _count_examples_by_step = results[key_mask_by_step] # a list of counts

                        if key_by_step not in cum_nll_per_cat_by_step:
                            cum_nll_per_cat_by_step[key_by_step] = []
                            cum_examples_per_cat_by_step[key_by_step] = []
                        for step, _log_prob in enumerate(_log_probs_by_step):
                            if _count_examples_by_step[step] == 0:
                                continue
                            if len(cum_nll_per_cat_by_step[key_by_step]) - 1 < step:
                                cum_nll_per_cat_by_step[key_by_step].append(- _log_prob.item())
                                cum_examples_per_cat_by_step[key_by_step].append(_count_examples_by_step[step].item())
                            else:
                                cum_nll_per_cat_by_step[key_by_step][step] -= _log_prob.item()
                                cum_examples_per_cat_by_step[key_by_step][step] += _count_examples_by_step[step].item()

                gated_log_probs_by_step = torch.unbind(torch.sum(log_probs * batch_edit_mask, dim=1), dim=0)
                batch_edit_mask_by_step = torch.unbind(batch_edit_mask.sum(dim=1), dim=0)
                key_by_step = "log_probs_by_step"
                if key_by_step not in cum_nll_per_cat_by_step:
                    cum_nll_per_cat_by_step[key_by_step] = []
                    cum_examples_per_cat_by_step[key_by_step] = []
                for step, _log_prob in enumerate(gated_log_probs_by_step):
                    if batch_edit_mask_by_step[step] == 0:
                        continue
                    if len(cum_nll_per_cat_by_step[key_by_step]) - 1 < step:
                        cum_nll_per_cat_by_step[key_by_step].append(- _log_prob.item())
                        cum_examples_per_cat_by_step[key_by_step].append(batch_edit_mask_by_step[step].item())
                    else:
                        cum_nll_per_cat_by_step[key_by_step][step] -= _log_prob.item()
                        cum_examples_per_cat_by_step[key_by_step][step] += batch_edit_mask_by_step[step].item()

            del results

    avg_ppl = np.exp(cum_ppl / cum_examples)
    avg_nll = cum_nll / cum_examples

    if was_training:
        model.train(was_training)

    if isinstance(model, Graph2IterEditEditor) and return_details:
        avg_other_nll_per_cat = {key: cum_other_nll_per_cat[key] / cum_examples_per_cat[key]
                                 for key in cum_other_nll_per_cat.keys()}

        avg_nll_per_cat_per_step = {}
        for key in cum_nll_per_cat_by_step.keys():
            _avg_scores = np.array(cum_nll_per_cat_by_step[key]) / np.array(cum_examples_per_cat_by_step[key])
            _print = ["%.3f (count=%d)" % (_score, cum_examples_per_cat_by_step[key][_idx])
                      for _idx, _score in enumerate(_avg_scores[:10])] # only the first 10 steps
            avg_nll_per_cat_per_step[key] = _print
        if return_nll_list:
            return avg_nll, avg_ppl, nll_dict, avg_other_nll_per_cat, avg_nll_per_cat_per_step
        else:
            return avg_nll, avg_ppl, avg_other_nll_per_cat, avg_nll_per_cat_per_step

    else:
        if return_nll_list:
            return avg_nll, avg_ppl, nll_dict
        else:
            return avg_nll, avg_ppl
