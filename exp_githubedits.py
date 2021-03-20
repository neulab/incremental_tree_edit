#!/usr/bin/env python
"""

Usage:
    exp_githubedits.py train [options] CONFIG_FILE
    exp_githubedits.py test_ppl [options] MODEL_PATH TEST_SET_PATH
    exp_githubedits.py decode_updated_data [options] MODEL_PATH TEST_SET_PATH
    exp_githubedits.py eval_csharp_fixer [options] MODEL_PATH TEST_SET_PATH
    exp_githubedits.py collect_edit_vecs [options] MODEL_PATH TEST_SET_PATH
    exp_githubedits.py imitation_learning [options] CONFIG_FILE

Options:
    -h --help                                   Show this screen
    --cuda                                      Use GPU
    --debug                                     Debug mode
    --seed=<int>                                Seed [default: 0]
    --work_dir=<dir>                            work dir [default: exp_runs/]
    --sample_size=<int>                         Sample size [default: 1]
    --beam_size=<int>                           Beam size [default: 1]
    --evaluate_ppl                              Evaluate perplexity as well
    --scorer=<str>                              Scorer for csharp fixer evaluation [default: default]
"""

import json
import os
import sys
import time
import gc
import pickle
from collections import OrderedDict, defaultdict
from docopt import docopt
from tqdm import tqdm
import numpy as np
import random
from copy import deepcopy
import math
import torch

from datasets.githubedits.common.config import Arguments
from edit_components.dataset import DataSet
from edit_components.change_entry import ChangeExample
from edit_model.encdec import *
from edit_model import nn_utils
from edit_components.vocab import Vocab, VocabEntry
from edit_components.evaluate import evaluate_nll
from edit_components.utils.decode import *
from edit_model.editor import NeuralEditor, ChangedWordPredictionMultiTask, Seq2SeqEditor, \
    Graph2TreeEditor, Graph2IterEditEditor
from edit_model.edit_encoder import SequentialChangeEncoder, GraphChangeEncoder
from trees.utils import calculate_tree_prod_f1, get_productions_str


def _extract_record(example):
    record = {'idx': example.id, 'init_code': str(example.prev_data), 'init_tree': example.prev_code_ast.root_node.to_string(),
              'tgt_code': str(example.updated_data), 'tgt_tree': example.updated_code_ast.root_node.to_string(),
              'context': str(example.context)}
    if hasattr(example, 'tgt_actions'):
        record.update({'tgt_actions': list(map(str, example.tgt_actions))})

    return record


def _load_dataset(filename, mode, model, args):
    assert mode in ('train', 'test')

    max_worker = 1
    tensorization = True
    if mode == 'train':
        max_worker = args['dataset']['num_data_load_worker']
        tensorization = args['dataset']['tensorization']

    if isinstance(model, Graph2IterEditEditor):
        filename_suffix = ""
        if not args['decoder']['copy_subtree']:
            filename_suffix += "_noCopySubtree"
        file_save_path = filename.replace('.jsonl', '%s.pkl' % filename_suffix)

        if os.path.exists(file_save_path):
            print("loading dataset from [%s]" % file_save_path, file=sys.stderr)
            begin_time = time.time()
            gc.disable()
            dataset = pickle.load(open(file_save_path, "rb"))
            gc.enable()
            print("time spent: %.3fs" % (time.time() - begin_time), file=sys.stderr)

            # check edit_type consistency
            if args['edit_encoder']['type'] == 'graph' and tensorization:
                _example = dataset.examples[0]
                if not hasattr(_example, 'change_edges'):
                    print("adding example attribute 'change_edges'...", file=sys.stderr)
                    from edit_model.edit_encoder import GraphChangeEncoder
                    for example in dataset.examples:
                        example.change_edges = GraphChangeEncoder.compute_change_edges(example)

                    # print("saving dataset to [%s]" % file_save_path, file=sys.stderr)
                    # begin_time = time.time()
                    # gc.disable()
                    # pickle.dump(dataset, open(file_save_path, "wb"), protocol=-1)
                    # gc.enable()
                    # print("time spent: %.3fs" % (time.time() - begin_time), file=sys.stderr)

            elif args['edit_encoder']['type'] == 'sequential' and tensorization:
                _example = dataset.examples[0]
                if not (hasattr(_example, 'prev_token_index') and hasattr(_example, 'updated_token_index') and
                        hasattr(_example, 'tag_index')):
                    print("adding example attributes for sequential edit encoder...", file=sys.stderr)
                    from edit_model.edit_encoder import SequentialChangeEncoder
                    for example in dataset.examples:
                        SequentialChangeEncoder.populate_aligned_token_index_and_mask(example)

                    # print("saving dataset to [%s]" % file_save_path, file=sys.stderr)
                    # begin_time = time.time()
                    # gc.disable()
                    # pickle.dump(dataset, open(file_save_path, "wb"), protocol=-1)
                    # gc.enable()
                    # print("time spent: %.3fs" % (time.time() - begin_time), file=sys.stderr)
        else:
            print("preprocessing dataset...", file=sys.stderr)
            dataset = DataSet.load_from_jsonl(filename,
                                              language=args['lang'],
                                              editor=model,
                                              max_workers=max_worker,
                                              tensorization=tensorization)
            # stats for debug
            lengths = [len(e.tgt_actions) for e in dataset.examples]
            print("average gold edit seq length: %.3f" % np.average(lengths), file=sys.stderr)

            print("saving dataset to [%s]" % file_save_path, file=sys.stderr)
            begin_time = time.time()
            gc.disable()
            pickle.dump(dataset, open(file_save_path, "wb"), protocol=-1)
            gc.enable()
            print("time spent: %.3fs" % (time.time() - begin_time), file=sys.stderr)

    else:
        dataset = DataSet.load_from_jsonl(filename,
                                          language=args['lang'],
                                          editor=model,
                                          max_workers=max_worker,
                                          tensorization=tensorization)

        if args['edit_encoder']['type'] == 'treediff':  # TreeDiff Edit Encoder
            print('TreeDiff Edit Encoder requires gold edit sequences.', file=sys.stderr)
            file_save_path = filename.replace('.jsonl', '.pkl')
            assert os.path.exists(file_save_path)
            print("loading dataset from [%s]" % file_save_path, file=sys.stderr)
            begin_time = time.time()
            gc.disable()
            dataset_w_edits = pickle.load(open(file_save_path, "rb"))
            gc.enable()
            print("time spent: %.3fs" % (time.time() - begin_time), file=sys.stderr)

            for e_idx, e in enumerate(dataset.examples):
                e.tgt_edits = dataset_w_edits.examples[e_idx].tgt_actions

    return dataset


def _train_fn(args, model, optimizer, train_set, dev_set, batch_size, work_dir,
              start_epoch_eval_decode=-1, eval_f1=False, model_name='model', optim_name='optim',
              max_epoch=None, with_gold_edits=False):
    parameters = list(model.parameters())
    if max_epoch is None:
        max_epoch = args['trainer']['max_epoch']

    print(f'training size={len(train_set)}', file=sys.stderr)
    print(f'max_epoch={max_epoch}', file=sys.stderr)

    epoch = train_iter = num_trial = 0
    # accumulate statistics on the device
    report_loss = 0.
    report_word_predict_loss = 0.
    report_examples = 0
    patience = 0
    history_dev_scores = []
    best_epoch = 0

    while True:
        model.train()
        epoch += 1
        epoch_begin = time.time()
        epoch_cum_examples = 0.

        print('', file=sys.stderr)

        eval_fn_name = 'eval_ppl'
        if start_epoch_eval_decode >= 1:
            if epoch == start_epoch_eval_decode:
                eval_fn_name = 'eval_decode'
                print('back up ppl history_dev_scores:', history_dev_scores, file=sys.stderr)
                history_dev_scores = [] # clear
                best_epoch = epoch
            elif epoch > start_epoch_eval_decode:
                eval_fn_name = 'eval_decode'

        for batch_examples in train_set.batch_iter(batch_size=batch_size, shuffle=True):
            train_iter += 1

            try:
                optimizer.zero_grad()

                results = model(batch_examples, return_change_vectors=True, with_gold_edits=with_gold_edits)
                log_probs, change_vecs = results['log_probs'], results['edit_encoding']
                loss = -log_probs.mean()

                total_loss_val = (-log_probs).sum().item()
                report_loss += total_loss_val
                report_examples += len(batch_examples)
                epoch_cum_examples += len(batch_examples)

                loss.backward()

                # clip gradient
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args['trainer']['clip_grad'])

                optimizer.step()
            except RuntimeError as e:
                err_message = getattr(e, 'message', str(e))
                if 'out of memory' in err_message:
                    print('OOM exception encountered, will skip this batch with examples:', file=sys.stderr)
                    for example in batch_examples:
                        print('\t%s' % example.id, file=sys.stderr)

                    try:
                        del loss, log_probs, change_vecs, results
                    except:
                        pass

                    gc_start = time.time()
                    gc.collect()
                    gc_time = time.time() - gc_start
                    print(f'gc took {gc_time}s', file=sys.stderr)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            del loss, log_probs, change_vecs, results

            if train_iter % args['trainer']['log_every'] == 0:
                print('[Iter %d] encoder loss=%.5f, word prediction loss=%.5f, %.2fs/epoch' %
                      (train_iter,
                       report_loss / report_examples, report_word_predict_loss / report_examples,
                       (time.time() - epoch_begin) / epoch_cum_examples * len(train_set)),
                      file=sys.stderr)

                report_loss = 0.
                report_examples = 0.
                report_word_predict_loss = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        # perform validation
        valid_begin = time.time()
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)

        if eval_fn_name == 'eval_ppl':
            dev_nll = _eval_ppl(model, dev_set, batch_size, epoch=epoch)
            dev_score = -dev_nll
        else:
            assert eval_fn_name == 'eval_decode'
            # dev_outputs = _eval_decode(model, dev_set, save_decode_results=False, decode_full_change_vecs=False,
            #                            eval_f1=eval_f1)
            dev_outputs = _eval_decode_in_batch(model, dev_set, batch_size=batch_size, save_decode_results=False,
                                                eval_f1=eval_f1)
            dev_score = dev_outputs[1]

        print('[Epoch %d] validation elapsed %d s' % (epoch, time.time() - valid_begin), file=sys.stderr)

        is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
        history_dev_scores.append(dev_score)

        if args['trainer']['save_each_epoch']:
            save_path = os.path.join(work_dir, f'{model_name}.epoch{epoch}.bin')
            print('[Epoch %d] save model to [%s]' % (epoch, save_path), file=sys.stderr)
            model.save(save_path)

        if is_better:
            patience = 0
            best_epoch = epoch
            save_path = os.path.join(work_dir, f'{model_name}.bin')
            print('save currently the best model to [%s]' % save_path, file=sys.stderr)
            model.save(save_path)

            # also save the optimizers' state
            torch.save(optimizer.state_dict(), os.path.join(work_dir, f'{optim_name}.bin'))
        elif patience < args['trainer']['patience']:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        print('current best %.3f (epoch %d)' % (max(history_dev_scores), best_epoch), file=sys.stderr)

        if patience == args['trainer']['patience']:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args['trainer']['max_num_trial']:
                print('early stop!', file=sys.stderr)
                break

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args['trainer']['lr_decay']
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(os.path.join(work_dir, f'{model_name}.bin'),
                                map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if use_cuda: model = model.cuda()

            # load optimizers
            if args['trainer']['reset_optimizer']:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(os.path.join(work_dir, f'{optim_name}.bin')))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0

        if epoch == max_epoch:
            print('reached maximum number of epochs!', file=sys.stderr)
            break

    return history_dev_scores


def _eval_ppl(model, dataset, batch_size, epoch=-1, change_vectors=None):
    epoch_info = "[Epoch %d] " % epoch if epoch >= 0 else ""
    eval_start = time.time()
    # evaluate ppl
    if isinstance(model, Graph2IterEditEditor):
        dev_nll, dev_ppl, dev_avg_other_nll_per_cat, avg_nll_per_cat_per_step = evaluate_nll(
            model, dataset, batch_size=batch_size, change_vectors=change_vectors, return_details=True)
        print('%saverage negative log likelihood=%.5f, average ppl=%.5f took %ds.' % (
            epoch_info, dev_nll, dev_ppl, time.time() - eval_start), file=sys.stderr)
        print('%sdetailed negative log likelihood=%.5f (op), %.5f (node), %.5f (add), %.5f (add_subtree).' % (
            epoch_info, dev_avg_other_nll_per_cat['tgt_op_log_probs'],
            dev_avg_other_nll_per_cat['tgt_node_log_probs'] if 'tgt_node_log_probs' in dev_avg_other_nll_per_cat else 0.0,
            dev_avg_other_nll_per_cat['tgt_add_log_probs'] if 'tgt_add_log_probs' in dev_avg_other_nll_per_cat else 0.0,
            dev_avg_other_nll_per_cat['tgt_add_subtree_log_probs'] if 'tgt_add_subtree_log_probs' in dev_avg_other_nll_per_cat else 0.0),
              file=sys.stderr)
        print('%sdetailed negative log likelihood by step:' % epoch_info, file=sys.stderr)
        print('\toverall:', avg_nll_per_cat_per_step['log_probs_by_step'], file=sys.stderr)
        print('\top:', avg_nll_per_cat_per_step['tgt_op_log_probs_by_step'], file=sys.stderr)
        if 'tgt_node_log_probs_by_step' in avg_nll_per_cat_per_step:
            print('\tnode:', avg_nll_per_cat_per_step['tgt_node_log_probs_by_step'], file=sys.stderr)
        if 'tgt_add_log_probs_by_step' in avg_nll_per_cat_per_step:
            print('\tadd:', avg_nll_per_cat_per_step['tgt_add_log_probs_by_step'], file=sys.stderr)
        if 'tgt_add_subtree_log_probs_by_step' in avg_nll_per_cat_per_step:
            print('\tadd_subtree:', avg_nll_per_cat_per_step['tgt_add_subtree_log_probs_by_step'], file=sys.stderr)
    else:
        dev_nll, dev_ppl = evaluate_nll(model, dataset, batch_size=batch_size, change_vectors=change_vectors)
        print('%saverage negative log likelihood=%.5f, average ppl=%.5f took %ds' % (
            epoch_info, dev_nll, dev_ppl, time.time() - eval_start), file=sys.stderr)

    return dev_nll


def _eval_decode(model, test_set, beam_size=1, length_norm=False, change_vectors=None,
                 debug=False, save_decode_results=True, decode_full_change_vecs=True, eval_f1=False):

    def _is_correct(_hyp, _example):
        if isinstance(model, Seq2SeqEditor):
            return _hyp.code == _example.updated_data
        elif isinstance(model, (Graph2TreeEditor, Graph2IterEditEditor)):
            return _hyp.tree == _example.updated_code_ast.root_node
        else:
            raise RuntimeError()

    def _get_f1(_hyp, _example):
        assert isinstance(model, (Graph2TreeEditor, Graph2IterEditEditor)), \
            'cannot evaluate tree prod F1 for model type %s' % type(model)
        prod2count_hyp = get_productions_str(_hyp.tree)
        prod2count_gold = get_productions_str(_example.updated_code_ast.root_node)
        return calculate_tree_prod_f1(prod2count_hyp, prod2count_gold)

    model.eval()

    batch_size = 32 if model.args['edit_encoder']['type'] == 'treediff' else 256
    print('batch_size=%d' % batch_size, file=sys.stderr)

    hits = []
    oracle_hits = []
    f1_scores = []
    decode_results = []
    count_failure = 0
    with torch.no_grad():
        # decode change vectors
        if change_vectors is None and decode_full_change_vecs:
            change_vectors = model.get_edit_encoding_by_batch(test_set.examples, batch_size=batch_size)
            print(f'decoded {change_vectors.shape[0]} entries', file=sys.stderr)

        cur_batch_start_e_idx = 0
        for e_idx, example in enumerate(tqdm(test_set.examples, file=sys.stdout, total=len(test_set))):
            if change_vectors is None and e_idx % batch_size == 0:
                cur_batch_change_vectors = model.get_edit_encoding_by_batch(test_set.examples[e_idx:e_idx+batch_size],
                                                                            batch_size=batch_size,
                                                                            quiet=True)
                cur_batch_start_e_idx = e_idx

            if change_vectors is not None:
                change_vec = change_vectors[e_idx]
                hypotheses = model.decode_updated_data(example, edit_encoding=change_vec, beam_size=beam_size,
                                                       length_norm=length_norm, debug=debug)
            else:
                change_vec = cur_batch_change_vectors[e_idx - cur_batch_start_e_idx]
                hypotheses = model.decode_updated_data(example, edit_encoding=change_vec, beam_size=beam_size,
                                                       length_norm=length_norm, debug=debug)

            if hypotheses:
                hit = _is_correct(hypotheses[0], example)
                if len(hypotheses) > 1:
                    oracle_hit = any(_is_correct(hyp, example) for hyp in hypotheses)
                else:
                    oracle_hit = hit
                if eval_f1:
                    f1 = _get_f1(hypotheses[0], example)
            else:
                oracle_hit = hit = False
                if eval_f1:
                    f1 = 0.
                count_failure += 1

            hits.append(float(hit))
            oracle_hits.append(float(oracle_hit))
            if eval_f1:
                f1_scores.append(f1)

            if hit:
                # print(example.id)
                # print('Prev:')
                # print(example.raw_prev_data)
                # print('Updated:')
                # print(example.raw_updated_data)
                mode = model.args['mode']
                if 'iter' in mode and debug:
                    results = model([example])
                    log_prob = results['log_probs'][0].item()
                    top_hyp_score = hypotheses[0].score
                    if length_norm:
                        top_hyp_score = top_hyp_score * len(hypotheses[0].edits)
                    if np.abs(top_hyp_score - log_prob) > 0.0001:  # happen when there are more than 1 valid paths
                        print(f'Warning: hyp score is different: {example.id}, hyp: {top_hyp_score}, train: {log_prob}',
                              file=sys.stderr)
                elif '2tree' in mode and debug:
                    # log_prob, debug_log = model([example], debug=True)
                    results = model([example])
                    log_prob = results['log_probs']
                    top_hyp_score = hypotheses[0].score
                    top_hyp_log = hypotheses[0].action_log
                    if np.abs(top_hyp_score.item() - log_prob[0].item()) > 0.0001:
                        print(
                            f'Warning: hyp score is different: {example.id}, hyp: {top_hyp_score.item()}, train: {log_prob[0].item()}',
                            file=sys.stderr)
                elif 'tree2seq' in mode and debug:
                    log_prob, debug_log = model([example], debug=True)
                    top_hyp_score = hypotheses[0].score
                    if np.abs(top_hyp_score - log_prob.item()) > 0.0001:
                        print(
                            f'Warning: hyp score is different: {example.id}, hyp: {top_hyp_score}, train: {log_prob[0].item()}',
                            file=sys.stderr)

            # f_log.write(f'*' * 20 +
            #             f'\nSource:\n{example.raw_prev_data}\n' +
            #             f'Target:\n{example.raw_updated_data}\n\n')

            if save_decode_results:
                hypotheses_logs = []
                for hyp in hypotheses:
                    entry = {'code': str(hyp.code) if isinstance(model, Seq2SeqEditor) else str([token.value for token
                                                                                                 in hyp.tree.descendant_tokens]),
                             'score': float(hyp.score),
                             'is_correct': _is_correct(hyp, example)}
                    if eval_f1:
                        entry['f1'] = _get_f1(hyp, example)

                    if debug and hasattr(hyp, 'action_log'):
                        entry['action_log'] = hyp.action_log
                    if 'iter' in model.args['mode']:
                        entry['edits'] = list(map(str, hyp.edits))
                        entry['tree'] = hyp.tree.to_string()
                    elif '2tree' in model.args['mode']:
                        entry['actions'] = list(map(str, hyp.actions))
                        entry['tree'] = hyp.tree.to_string()

                    hypotheses_logs.append(entry)
                decode_results.append({'example': _extract_record(example),
                                       'hypotheses_logs': hypotheses_logs})

            del hypotheses

    acc = sum(hits) / len(hits)
    oracle_acc = sum(oracle_hits) / len(oracle_hits)

    print('', file=sys.stderr)
    print(f'acc@{beam_size}={sum(hits)}/{len(hits)}={acc}', file=sys.stderr)
    print(f'oracle acc@{beam_size}={sum(oracle_hits)}/{len(oracle_hits)}={oracle_acc}',
          file=sys.stderr)
    print(f'#failure={count_failure}', file=sys.stderr)

    if eval_f1:
        avg_f1 = np.average(f1_scores)
        print(f'f1@{beam_size}={avg_f1}', file=sys.stderr)
        if 0 in hits:
            error_case_f1_scores = [score for _i, score in enumerate(f1_scores) if hits[_i] == 0]
            error_case_avg_f1 = np.average(error_case_f1_scores)
            print(f'error case f1@{beam_size}={sum(error_case_f1_scores)}/{len(error_case_f1_scores)}={error_case_avg_f1}',
                  file=sys.stderr)
        else:
            error_case_avg_f1 = 0.
        return decode_results, acc, oracle_acc, avg_f1, error_case_avg_f1

    return decode_results, acc, oracle_acc


def _eval_decode_in_batch(model, test_set, batch_size, save_decode_results=True, eval_f1=False):
    def _is_correct(_hyp, _example):
        if isinstance(model, Seq2SeqEditor):
            return _hyp.code == _example.updated_data
        elif isinstance(model, (Graph2TreeEditor, Graph2IterEditEditor)):
            return _hyp.tree == _example.updated_code_ast.root_node
        else:
            raise RuntimeError()

    def _get_f1(_hyp, _example):
        assert isinstance(model, (Graph2TreeEditor, Graph2IterEditEditor)), \
            'cannot evaluate tree prod F1 for model type %s' % type(model)
        prod2count_hyp = get_productions_str(_hyp.tree)
        prod2count_gold = get_productions_str(_example.updated_code_ast.root_node)
        return calculate_tree_prod_f1(prod2count_hyp, prod2count_gold)

    assert isinstance(model, Graph2IterEditEditor)
    model.eval()

    hits = []
    f1_scores = []
    decode_results = []
    with torch.no_grad():
        num_batches = int(math.ceil(len(test_set)*1.0/batch_size))
        for batch_idx in tqdm(range(num_batches), file=sys.stdout, total=num_batches):
            examples = test_set.examples[batch_idx*batch_size: (batch_idx+1)*batch_size]

            sorted_example_ids, example_old2new_pos = nn_utils.get_sort_map([len(c.change_seq) for c in examples])
            sorted_examples = [examples[i] for i in sorted_example_ids]
            sorted_hypotheses = model.decode_updated_data_in_batch(sorted_examples)

            hypotheses = [None for _ in range(len(examples))]
            for old_pos, new_pos in enumerate(example_old2new_pos):
                hypotheses[old_pos] = sorted_hypotheses[new_pos]

            for e_idx, hyp in enumerate(hypotheses):
                example = examples[e_idx]

                hit = _is_correct(hyp, example)
                hits.append(hit)

                if eval_f1:
                    f1 = _get_f1(hyp, example)
                    f1_scores.append(f1)

                if save_decode_results:
                    hypotheses_logs = []
                    entry = {
                        'code': str(hyp.code) if isinstance(model, Seq2SeqEditor) else str([token.value for token
                                                                                            in hyp.tree.descendant_tokens]),
                        'score': float(hyp.score),
                        'is_correct': _is_correct(hyp, example)}
                    if eval_f1:
                        entry['f1'] = _get_f1(hyp, example)

                    if 'iter' in model.args['mode']:
                        entry['edits'] = list(map(str, hyp.edits))
                        entry['tree'] = hyp.tree.to_string()
                    elif '2tree' in model.args['mode']:
                        entry['actions'] = hyp.actions
                        entry['tree'] = hyp.tree.to_string()

                    hypotheses_logs.append(entry)
                    decode_results.append({'example': _extract_record(example),
                                           'hypotheses_logs': hypotheses_logs})

            del hypotheses

    acc = sum(hits) / len(hits)
    oracle_acc = acc

    print('', file=sys.stderr)
    print(f'acc@1={sum(hits)}/{len(hits)}={acc}', file=sys.stderr)

    if eval_f1:
        avg_f1 = np.average(f1_scores)
        print(f'f1@1={avg_f1}', file=sys.stderr)
        if 0 in hits:
            error_case_f1_scores = [score for _i, score in enumerate(f1_scores) if hits[_i] == 0]
            error_case_avg_f1 = np.average(error_case_f1_scores)
            print(f'error case f1@1={sum(error_case_f1_scores)}/{len(error_case_f1_scores)}={error_case_avg_f1}',
                  file=sys.stderr)
        else:
            error_case_avg_f1 = 0.
        return decode_results, acc, oracle_acc, avg_f1, error_case_avg_f1

    return decode_results, acc, oracle_acc


def _collect_iteration_example(example_change_vec_pair, model, sampling_probability, imitation_iteration,
                               id2accumulated_edits_string=None, id2non_gold_edit_seqs=None,
                               max_trajectory_length=70, prioritize_last_edit=True, extend_stop=False, debug=False):

    example, change_vec = example_change_vec_pair

    existing_non_gold_edit_seqs = None
    if id2non_gold_edit_seqs is not None and example.id in id2non_gold_edit_seqs:
        existing_non_gold_edit_seqs = id2non_gold_edit_seqs[example.id]

    gold_edits, actual_edits, decoded_tree = model.decode_with_gold_sample(example,
                                                                           sampling_probability=sampling_probability,
                                                                           max_trajectory_length=max_trajectory_length,
                                                                           edit_encoding=change_vec,
                                                                           existing_non_gold_edit_seqs=existing_non_gold_edit_seqs,
                                                                           prioritize_last_edit=prioritize_last_edit,
                                                                           extend_stop=extend_stop,
                                                                           debug=debug)

    # decoded_tree = gold_edits[-1].meta['tree'].root_node
    hit = float(decoded_tree.to_string() == example.updated_code_ast.root_node.to_string())
    prod2count_hyp = get_productions_str(decoded_tree)
    prod2count_gold = get_productions_str(example.updated_code_ast.root_node)
    f1_score = calculate_tree_prod_f1(prod2count_hyp, prod2count_gold)

    if id2accumulated_edits_string is not None:
        edits_strings = []
        for edit in gold_edits:
            edits_strings.append(edit.meta['tree'].root_node.to_string())
        edits_string = '##EDIT##'.join(edits_strings)

        if edits_string not in id2accumulated_edits_string[example.id]:
            id2accumulated_edits_string[example.id].add(edits_string)

            new_example = ChangeExample(id=example.id + '_ITER%d' % imitation_iteration,
                                        prev_data=example.prev_data,
                                        updated_data=example.updated_data,
                                        context=example.context,
                                        tgt_actions=gold_edits)
            if model.args['edit_encoder']['type'] == 'sequential':
                SequentialChangeEncoder.populate_aligned_token_index_and_mask(new_example)
            elif model.args['edit_encoder']['type'] == 'graph':
                new_example.change_edges = GraphChangeEncoder.compute_change_edges(new_example)

            id2non_gold_edit_seqs[example.id].append(gold_edits)
            return new_example, hit
        else:
            return None, hit
    else:
        new_example = ChangeExample(id=example.id + '_ITER%d' % imitation_iteration,
                                    prev_data=example.prev_data,
                                    updated_data=example.updated_data,
                                    context=example.context,
                                    tgt_actions=gold_edits)
        if model.args['edit_encoder']['type'] == 'sequential':
            SequentialChangeEncoder.populate_aligned_token_index_and_mask(new_example)
        elif model.args['edit_encoder']['type'] == 'graph':
            new_example.change_edges = GraphChangeEncoder.compute_change_edges(new_example)

        return new_example, hit, f1_score


def _collect_iteration_example_in_batch(examples, model, sampling_probability, imitation_iteration,
                                        id2accumulated_edits_string=None, id2non_gold_edit_seqs=None, change_vecs=None,
                                        max_trajectory_length=70, prioritize_last_edit=True, extend_stop=False,
                                        extra_gold_edits=False, debug=False):

    existing_non_gold_edit_seqs_list = None
    if id2non_gold_edit_seqs is not None:
        existing_non_gold_edit_seqs_list = []
        for example in examples:
            existing_non_gold_edit_seqs = None
            if example.id in id2non_gold_edit_seqs:
                existing_non_gold_edit_seqs = id2non_gold_edit_seqs[example.id]
            existing_non_gold_edit_seqs_list.append(existing_non_gold_edit_seqs)

    gold_edits_list, actual_edits_list, decoded_tree_list = model.decode_with_gold_sample_in_batch(
        examples,
        sampling_probability=sampling_probability,
        max_trajectory_length=max_trajectory_length,
        edit_encodings=change_vecs,
        existing_non_gold_edit_seqs_list=existing_non_gold_edit_seqs_list,
        prioritize_last_edit=prioritize_last_edit,
        extend_stop=extend_stop,
        debug=debug)

    hits, new_examples = [], []
    f1_scores = []
    for e_idx, gold_edits in enumerate(gold_edits_list):
        example = examples[e_idx]
        # decoded_tree = gold_edits[-1].meta['tree'].root_node
        decoded_tree = decoded_tree_list[e_idx]

        hit = float(decoded_tree == example.updated_code_ast.root_node)
        hits.append(hit)

        prod2count_hyp = get_productions_str(decoded_tree)
        prod2count_gold = get_productions_str(example.updated_code_ast.root_node)
        f1_score = calculate_tree_prod_f1(prod2count_hyp, prod2count_gold)
        f1_scores.append(f1_score)

        if id2accumulated_edits_string is not None:
            edits_strings = []
            for edit in gold_edits:
                edits_strings.append(edit.meta['tree'].root_node.to_string())
            edits_string = '##EDIT##'.join(edits_strings)

            if edits_string not in id2accumulated_edits_string[example.id]:
                id2accumulated_edits_string[example.id].add(edits_string)

                new_example = ChangeExample(id=example.id + '_ITER%d' % imitation_iteration,
                                            prev_data=example.prev_data,
                                            updated_data=example.updated_data,
                                            context=example.context,
                                            tgt_actions=gold_edits)
                if extra_gold_edits:
                    new_example.gold_edits = example.tgt_actions
                if model.args['edit_encoder']['type'] == 'sequential':
                    SequentialChangeEncoder.populate_aligned_token_index_and_mask(new_example)
                elif model.args['edit_encoder']['type'] == 'graph':
                    new_example.change_edges = GraphChangeEncoder.compute_change_edges(new_example)

                id2non_gold_edit_seqs[example.id].append(gold_edits)
                new_examples.append(new_example)
        else:
            new_example = ChangeExample(id=example.id + '_ITER%d' % imitation_iteration,
                                        prev_data=example.prev_data,
                                        updated_data=example.updated_data,
                                        context=example.context,
                                        tgt_actions=gold_edits)
            if extra_gold_edits:
                new_example.gold_edits = example.tgt_actions
            if model.args['edit_encoder']['type'] == 'sequential':
                SequentialChangeEncoder.populate_aligned_token_index_and_mask(new_example)
            elif model.args['edit_encoder']['type'] == 'graph':
                new_example.change_edges = GraphChangeEncoder.compute_change_edges(new_example)

            new_examples.append(new_example)

    return new_examples, hits, f1_scores


def _collect_correction_iteration_example_in_batch(examples, model, imitation_iteration, id2decoded_tree_string,
                                                   max_trajectory_length=70, prioritize_last_edit=True,
                                                   extra_gold_edits=False):
    gold_edits_list, edits_weight_list, actual_decoded_trees = model.decode_with_extend_correction_in_batch(
        examples, max_trajectory_length, prioritize_last_edit=prioritize_last_edit,
        id2decoded_tree_string=id2decoded_tree_string)

    hits, new_examples = [], []
    f1_scores = []
    for e_idx in range(len(examples)):
        example = examples[e_idx]
        gold_edits = gold_edits_list[e_idx]
        edits_weight = edits_weight_list[e_idx]
        decoded_tree = actual_decoded_trees[e_idx]

        if gold_edits is None and decoded_tree is None:
            hits.append(1.)
            f1_scores.append(1.)
            continue

        hits.append(0.)
        prod2count_hyp = get_productions_str(decoded_tree)
        prod2count_gold = get_productions_str(example.updated_code_ast.root_node)
        f1_score = calculate_tree_prod_f1(prod2count_hyp, prod2count_gold)
        f1_scores.append(f1_score)

        if gold_edits is None: # repetitive example
            continue

        decoded_tree_string = decoded_tree.to_string()

        id2decoded_tree_string[example.id].add(decoded_tree_string)
        new_example = ChangeExample(id=example.id + '_ITER%d' % imitation_iteration,
                                    prev_data=example.prev_data,
                                    updated_data=example.updated_data,
                                    context=example.context,
                                    tgt_actions=gold_edits,
                                    tgt_actions_weight=edits_weight)
        if extra_gold_edits:
            new_example.gold_edits = example.tgt_actions
        if model.args['edit_encoder']['type'] == 'sequential':
            SequentialChangeEncoder.populate_aligned_token_index_and_mask(new_example)
        elif model.args['edit_encoder']['type'] == 'graph':
            new_example.change_edges = GraphChangeEncoder.compute_change_edges(new_example)

        new_examples.append(new_example)

    return new_examples, hits, f1_scores


def train(cmd_args):
    args = Arguments.from_file(cmd_args['CONFIG_FILE'], cmd_args=cmd_args)

    work_dir = cmd_args['--work_dir']
    use_cuda = cmd_args['--cuda']
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # save arguments to work dir
    f = open(os.path.join(work_dir, 'config.json'), 'w')
    f.write(args.to_string())
    f.close()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = NeuralEditor.build(args)

    print('loading datasets...', file=sys.stderr)
    train_set = _load_dataset(args['dataset']['train_file'], 'train', model, args)
    dev_set = _load_dataset(args['dataset']['dev_file'], 'train', model, args)

    print('loaded train file at [%s] (size=%d), dev file at [%s] (size=%d)' % (
        args['dataset']['train_file'], len(train_set),
        args['dataset']['dev_file'], len(dev_set)), file=sys.stderr)

    model = model.to(device)
    model.train()

    batch_size = args['trainer']['batch_size']

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    _train_fn(args, model, optimizer, train_set, dev_set, batch_size, work_dir)


def imitation_learning(cmd_args):
    args = Arguments.from_file(cmd_args['CONFIG_FILE'], cmd_args=cmd_args)

    work_dir = cmd_args['--work_dir']
    use_cuda = cmd_args['--cuda']
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # save arguments to work dir
    f = open(os.path.join(work_dir, 'config.json'), 'w')
    f.write(args.to_string())
    f.close()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('use device: %s' % device, file=sys.stderr)
    print('debug:', cmd_args['--debug'])

    max_workers = args['dataset']['num_data_load_worker']
    print('max_workers: %d' % max_workers, file=sys.stderr)

    # imitation learning core config
    max_iterations = args['imitation_learning']['iterations']
    decode_with_correction = args['imitation_learning']['decode_with_correction']
    if decode_with_correction:
        sample_size = 1
        prioritize_last_edit = args['imitation_learning']['prioritize_last_edit']
        print('imitation learning: decode_with_correction True, %d iterations, '
              'sample_size %d, prioritize_last_edit %s' % (
            max_iterations, sample_size, prioritize_last_edit), file=sys.stderr)
    else:
        beta_config = float(args['imitation_learning']['beta'])
        assert 0 <= beta_config <= 1
        sample_size = args['imitation_learning']['sample_size']
        if beta_config == 0:
            assert sample_size == 1, 'setting sample_size > 1 with beta config I(i=1) is meaningless.'
        prioritize_last_edit = args['imitation_learning']['prioritize_last_edit']
        extend_stop = args['imitation_learning']['extend_stop']
        print('imitation learning: %d iterations, beta configuration %.3f, sample_size %d, '
              'prioritize_last_edit %s, extend_stop %s' % (
            max_iterations, beta_config, sample_size, prioritize_last_edit, extend_stop), file=sys.stderr)

    start_epoch_eval_decode = args['imitation_learning']['start_epoch_eval_decode']
    max_epoch = args['imitation_learning']['max_epoch']
    print('start_epoch_eval_decode=%d, max_epoch=%d' % (start_epoch_eval_decode, max_epoch), file=sys.stderr)

    batch_size = args['trainer']['batch_size']
    train_batch_size = args['imitation_learning']['batch_size']
    print(f'batch_size={batch_size}, training batch_size={train_batch_size}', file=sys.stderr)

    model = NeuralEditor.build(args)
    assert args['mode'] == 'graph2iteredit'
    init_state_dict = deepcopy(model.state_dict())

    restore_model_path = args['imitation_learning']['load_model_path']
    print('load model checkpoint from [%s]' % restore_model_path, file=sys.stderr)
    restore_params = torch.load(restore_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(restore_params['state_dict'])

    print('loading datasets...', file=sys.stderr)
    train_set = _load_dataset(args['dataset']['train_file'], 'train', model, args)
    dev_set = _load_dataset(args['dataset']['dev_file'], 'train', model, args)
    if args['debug']:
        train_set = DataSet(train_set.examples[:1000])
        dev_set = DataSet(dev_set.examples[:1000])
    print('loaded train file at [%s] (size=%d), dev file at [%s] (size=%d)' % (
        args['dataset']['train_file'], len(train_set),
        args['dataset']['dev_file'], len(dev_set)), file=sys.stderr)
    assert max_iterations > 0

    model = model.to(device)

    # initialize training data
    if decode_with_correction:
        id2decoded_tree_string = defaultdict(set)
        accumulated_train_examples = list(train_set.examples)
    else:
        id2accumulated_edits_string, id2non_gold_edit_seqs = defaultdict(set), defaultdict(list)
        accumulated_train_examples = list(train_set.examples)
        for example in train_set.examples:
            edits_strings = []
            for edit in example.tgt_actions:
                edits_strings.append(edit.meta['tree'].root_node.to_string())
            edits_string = '##EDIT##'.join(edits_strings)
            id2accumulated_edits_string[example.id].add(edits_string)

    history_dev_scores_by_iteration = []
    for imitation_iteration in range(1, max_iterations + 1):
        print('\nimitation learning iteration %d starts' % imitation_iteration, file=sys.stderr)
        if not decode_with_correction:
            sampling_probability = 1 - beta_config**imitation_iteration
            print('1 - beta_i = %f' % sampling_probability, file=sys.stderr)

        iteration_source_examples = train_set.examples

        iteration_backup_path = os.path.join(work_dir, f'iteration_examples.iter{imitation_iteration}.pkl')
        if imitation_iteration == 1 and os.path.exists(iteration_backup_path):
            print('load iteration examples from [%s] (skip collection)' % iteration_backup_path, file=sys.stderr)
            begin_time = time.time()
            gc.disable()
            iteration_examples = pickle.load(open(iteration_backup_path, "rb"))
            gc.enable()
            print("time spent: %ds\n" % (time.time() - begin_time), file=sys.stderr)
        else:
            model.eval()
            iteration_examples = []
            hits, f1_scores = [], []
            begin_time = time.time()
            with torch.no_grad():
                for _ in range(sample_size):
                    iteration_source_dataset = DataSet(iteration_source_examples)
                    for examples in tqdm(iteration_source_dataset.batch_iter(batch_size=batch_size, shuffle=False),
                                         file=sys.stdout, total=math.ceil(len(iteration_source_dataset) * 1.0 /batch_size)):
                        if decode_with_correction:
                            _collected_iteration_examples, _hits, _f1_scores = _collect_correction_iteration_example_in_batch(
                                examples, model, imitation_iteration, id2decoded_tree_string,
                                max_trajectory_length=args['trainer']['max_change_sequence_length'],
                                prioritize_last_edit=prioritize_last_edit,
                                extra_gold_edits=args['edit_encoder']['type']=='treediff')
                        else:
                            _collected_iteration_examples, _hits, _f1_scores = _collect_iteration_example_in_batch(
                                examples, model, sampling_probability, imitation_iteration,
                                id2accumulated_edits_string, id2non_gold_edit_seqs,
                                max_trajectory_length=args['trainer']['max_change_sequence_length'],
                                prioritize_last_edit=prioritize_last_edit,
                                extend_stop=extend_stop,
                                extra_gold_edits=args['edit_encoder']['type']=='treediff',
                                debug=cmd_args['--debug'])
                        iteration_examples.extend(_collected_iteration_examples)
                        hits.extend(_hits)
                        f1_scores.extend(_f1_scores)

                # acc
                print(f'acc@1={sum(hits)}/{len(hits)}={sum(hits) / len(hits)}', file=sys.stderr)
                avg_f1 = np.average(f1_scores)
                print(f'f1@1={avg_f1}', file=sys.stderr)
                if 0 in hits:
                    error_case_f1_scores = [score for _i, score in enumerate(f1_scores) if hits[_i] == 0]
                    error_case_avg_f1 = np.average(error_case_f1_scores)
                    print(f'error case f1@1={sum(error_case_f1_scores)}/{len(error_case_f1_scores)}={error_case_avg_f1}',
                          file=sys.stderr)
                print('iteration examples collection done. time elapsed %ds.' % (time.time() - begin_time),
                      file=sys.stderr)

                if len(iteration_examples) == 0:
                    print('collected zero iteration example. imitation learning stopped!', file=sys.stderr)
                    exit()

                # save
                print('saving iteration examples to [%s]' % iteration_backup_path, file=sys.stderr)
                begin_time = time.time()
                gc.disable()
                pickle.dump(iteration_examples, open(iteration_backup_path, 'wb'), protocol=-1)
                gc.enable()
                print("time spent: %ds\n" % (time.time() - begin_time), file=sys.stderr)

        print(f'added {len(iteration_examples)} new training trajectories', file=sys.stderr)

        accumulated_train_examples.extend(iteration_examples)
        iteration_train_set = DataSet(accumulated_train_examples)

        # training
        print('\nreset model parameters and retrain', file=sys.stderr)
        model.load_state_dict(deepcopy(init_state_dict))
        model = model.to(device)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.001)

        history_dev_scores = _train_fn(args, model, optimizer, iteration_train_set, dev_set, train_batch_size, work_dir,
                                       start_epoch_eval_decode=start_epoch_eval_decode,
                                       eval_f1=True,
                                       model_name=f'model.{imitation_iteration}',
                                       optim_name=f'optim.{imitation_iteration}',
                                       max_epoch=max_epoch,
                                       with_gold_edits=args['edit_encoder']['type']=='treediff')

        is_better_iteration = history_dev_scores_by_iteration == [] or \
                              max(history_dev_scores) > max(history_dev_scores_by_iteration)
        if is_better_iteration:
            save_path = os.path.join(work_dir, 'model.bin')
            print('save currently the best model to [%s]' % save_path, file=sys.stderr)
            model.save(save_path)
        history_dev_scores_by_iteration.append(max(history_dev_scores))

        print('imitation learning iteration %d ends\n' % imitation_iteration, file=sys.stderr)


def test_ppl(args):
    sys.setrecursionlimit(7000)
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = NeuralEditor.load(model_path, use_cuda=args['--cuda'])

    # load dataset
    # print(f'loading dataset from [{test_set_path}]', file=sys.stderr)
    # test_set = DataSet.load_from_jsonl(test_set_path, language=args['lang'], editor=model)
    test_set = _load_dataset(test_set_path, 'test', model, model.args)

    _eval_ppl(model, test_set, batch_size=128)


def decode_updated_code(args):
    sys.setrecursionlimit(7000)
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']
    beam_size = int(args['--beam_size'])
    print(f'beam_size: {beam_size}')

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = NeuralEditor.load(model_path, use_cuda=args['--cuda'])

    # load dataset
    test_set = _load_dataset(test_set_path, 'test', model, args=model.args)
    
    if beam_size == 1 and isinstance(model, Graph2IterEditEditor):
        output = _eval_decode_in_batch(model, test_set, batch_size=32)
    else:
        output = _eval_decode(model, test_set, beam_size=beam_size,
                              debug=args['--debug'])
    decode_results = output[0]

    if args['--evaluate_ppl']:
        _eval_ppl(model, test_set, 128)

    filename = 'decode'
    if beam_size > 1:
        filename += f'_beam{beam_size}'
    if 'csharp_fixer' in test_set_path:
        filename += '_csharp_fixer_gold'
    else:
        basename = os.path.basename(test_set_path)
        assert basename.startswith('githubedits.')
        filename += '_' + basename.split('.')[1]
    save_decode_path = model_path + f'.{filename}.json'
    json.dump(decode_results, open(save_decode_path, 'w'), indent=4)
    print(f'saved decoding results to {save_decode_path}', file=sys.stderr)


def eval_csharp_fixer(args):
    from scipy.spatial import distance

    sys.setrecursionlimit(7000)
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']
    assert 'csharp_fixer' in test_set_path
    beam_size = int(args['--beam_size'])
    print(f'beam_size: {beam_size}')

    scorer = args['--scorer']
    print("scorer:", scorer)
    assert scorer in ('default', 'iclr19')

    seed_query_sample_size = 10 if scorer == 'iclr19' else 100
    np.random.seed(1234)

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = NeuralEditor.load(model_path, use_cuda=args['--cuda'])
    model.eval()

    def _is_correct(_hyp, _example):
        if isinstance(model, Seq2SeqEditor):
            return _hyp.code == _example.updated_data
        elif isinstance(model, (Graph2TreeEditor, Graph2IterEditEditor)):
            return _hyp.tree == _example.updated_code_ast.root_node
        else:
            raise RuntimeError()

    # load dataset
    dataset = _load_dataset(test_set_path, 'test', model, model.args)

    fixer_cats = ['CA2007', 'IDE0004', 'RCS1015', 'RCS1021', 'RCS1032', 'RCS1058', 'RCS1077', 'RCS1089', 'RCS1097',
                  'RCS1118', 'RCS1123', 'RCS1146', 'RCS1197', 'RCS1202', 'RCS1206', 'RCS1207']

    # micro average
    total_correct, total_correct_recall = 0., 0.
    total_elements = 0

    def _decode_and_compute_acc(_example_batch, _change_vec_batch):
        if isinstance(model, Graph2IterEditEditor) and beam_size == 1:
            _change_vec_batch = torch.stack(_change_vec_batch, dim=0)
            hypotheses_batch = model.decode_updated_data_in_batch(
                _example_batch, edit_encodings=_change_vec_batch)
            hypotheses_batch = [[hypothesis] for hypothesis in hypotheses_batch]
        else:
            hypotheses_batch = []
            for _example, _change_vec in zip(_example_batch, _change_vec_batch):
                hypotheses = model.decode_updated_data(_example,
                                                       edit_encoding=_change_vec,
                                                       beam_size=beam_size)
                hypotheses_batch.append(hypotheses)

        hit_batch, recall_hit_batch, hypotheses_logs_batch = [], [], []
        for _example_idx, hypotheses in enumerate(hypotheses_batch):
            _example = _example_batch[_example_idx]
            hypotheses_logs = []
            if hypotheses:
                recall_hit = any(_is_correct(hyp, _example) for hyp in hypotheses)
                hit = _is_correct(hypotheses[0], _example)
                for hyp in hypotheses:
                    entry = {
                        'code': hyp.code if isinstance(model, Seq2SeqEditor) else [token.value for token in hyp.tree.descendant_tokens],
                        'score': float(hyp.score),
                        'is_correct': _is_correct(hyp, _example)}
                    if args['--debug'] and hasattr(hyp, 'action_log'):
                        entry['action_log'] = hyp.action_log
                    if 'iter' in model.args['mode']:
                        entry['edits'] = hyp.edits
                        entry['tree'] = hyp.tree
                    elif '2tree' in model.args['mode']:
                        entry['actions'] = hyp.actions
                        entry['tree'] = hyp.tree
                    hypotheses_logs.append(entry)
            else:
                recall_hit = hit = False
                hypotheses_logs = []

            hit_batch.append(hit)
            recall_hit_batch.append(recall_hit)
            hypotheses_logs_batch.append(hypotheses_logs)

        return hit_batch, recall_hit_batch, hypotheses_logs_batch

    def count_stats(scores, name):
        max_score = max(scores)
        avg_score = np.average(scores)
        print(f'  avg {name}: {avg_score}, max {name}: {max_score}', file=sys.stderr)
        return max_score, avg_score

    with torch.no_grad():
        if model.args['edit_encoder']['type'] == 'treediff':
            feature_vecs = model.get_edit_encoding_by_batch(dataset.examples, batch_size=64)
        else:
            feature_vecs = model.get_edit_encoding_by_batch(dataset.examples, batch_size=256)
        print(f'decoded {feature_vecs.shape[0]} entries', file=sys.stderr)

        decode_results = OrderedDict()
        for fixer_id in fixer_cats:
            example_ids_under_this_category = [e.id for e in dataset.examples if e.id.startswith(fixer_id)]

            if scorer == 'iclr19':
                np.random.shuffle(example_ids_under_this_category)
            else:
                # example_ids_under_this_category.sort()
                pass

            seed_querie_ids = example_ids_under_this_category[:seed_query_sample_size]
            seed_query_accs, upper_bound_seed_query_accs = [], []
            seed_query_recalls, upper_bound_seed_query_recalls = [], []
            seed_query_neighbor_decoding_results = []

            for seed_query_id in seed_querie_ids:
                seed_query_idx = dataset.example_id_to_index[seed_query_id]
                seed_query_vec = feature_vecs[seed_query_idx]

                if scorer == 'iclr19':
                    neighbors = [(nbr_idx, distance.cosine(seed_query_vec.cpu().numpy(),
                                                           feature_vecs[nbr_idx].cpu().numpy()))
                                 for nbr_idx, e in enumerate(dataset.examples)
                                 if e.id.startswith(fixer_id)]
                else:
                    neighbors = [(dataset.example_id_to_index[neighbor_id],
                                  distance.cosine(seed_query_vec.cpu().numpy(),
                                                  feature_vecs[dataset.example_id_to_index[neighbor_id]].cpu().numpy()))
                                 for neighbor_id in seed_querie_ids if neighbor_id != seed_query_id]

                print(f'Fixer {fixer_id}, seed query {seed_query_id}, {len(neighbors)} neighbors...')

                neighbors.sort(key=lambda x: x[1])
                neighbor_decoding_results = []
                neighbor_hypotheses_logs = []

                # batch-wise decoding
                neighbor_example_batch = [dataset.examples[nbr_idx] for nbr_idx, _ in neighbors]
                hit_batch, recall_batch, hypotheses_logs_batch = _decode_and_compute_acc(
                    neighbor_example_batch, [seed_query_vec] * len(neighbor_example_batch))

                for rank_idx, (nbr_idx, dist) in enumerate(neighbors):
                    neighbor_example = dataset.examples[nbr_idx]
                    nbr_feature_vec = feature_vecs[nbr_idx]

                    # hit, recall, hypotheses_logs = _decode_and_compute_acc(neighbor_example, seed_query_vec)
                    hit, recall, hypotheses_logs = hit_batch[rank_idx], recall_batch[rank_idx], \
                                                   hypotheses_logs_batch[rank_idx]

                    neighbor_decoding_results.append(dict(id=neighbor_example.id,
                                                          hit=hit,
                                                          recall=recall,
                                                          dist=dist))

                    total_elements += 1
                    total_correct += float(hit)
                    total_correct_recall += float(recall)

                    if rank_idx < 10 or rank_idx > len(neighbors) - 10: # save samples in the highest/lowest 10 rank
                        neighbor_hypotheses_logs.append((neighbor_example, hypotheses_logs,
                                                         neighbor_decoding_results[-1]))

                acc_current_seed_query = np.average([nbr['hit'] for nbr in neighbor_decoding_results])
                recall_current_seed_query = np.average([nbr['recall'] for nbr in neighbor_decoding_results])
                seed_query_accs.append(acc_current_seed_query)
                seed_query_recalls.append(recall_current_seed_query)
                seed_query_neighbor_decoding_results.append((seed_query_id, neighbor_hypotheses_logs))

            # stats over seed queries
            print(f'Fixer {fixer_id}:', file=sys.stderr)
            max_acc, avg_acc = count_stats(seed_query_accs, 'acc')
            max_recall, avg_recall = count_stats(seed_query_recalls, 'recall')

            decode_results[fixer_id] = (seed_querie_ids, [max_acc, avg_acc, max_recall, avg_recall],
                                        seed_query_neighbor_decoding_results)

        # aggregation for all fixer cats
        aggregation = np.average([decode_results[fixer_id][1] for fixer_id in fixer_cats], axis=0)
        print('', file=sys.stderr)
        print(f'avg_acc@{beam_size}(macro)={aggregation[1]}', file=sys.stderr)
        print(f'avg_recall@{beam_size}(macro)={aggregation[3]}', file=sys.stderr)
        print(f'avg_acc@{beam_size}(micro)={total_correct / total_elements}', file=sys.stderr)
        print(f'avg_recall@{beam_size}(micro)={total_correct_recall / total_elements}', file=sys.stderr)
        if scorer == 'iclr19':
            print(f'max_acc@{beam_size}(macro)={aggregation[0]}', file=sys.stderr)
            print(f'max_recall@{beam_size}(macro)={aggregation[2]}', file=sys.stderr)

        filename = 'decode'
        if beam_size > 1:
            filename += f'_beam{beam_size}'
        save_decode_path = model_path + f'.{filename}_csharp_fixer_{scorer}.bin'
        pickle.dump(decode_results, open(save_decode_path, 'bw'))
        print(f'saved decoding results to {save_decode_path}', file=sys.stderr)


def collect_edit_vecs(args):
    model_path = args['MODEL_PATH']
    test_set_path = args['TEST_SET_PATH']

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model = NeuralEditor.load(model_path, use_cuda=args['--cuda'])
    model.eval()

    # load dataset
    dataset = _load_dataset(test_set_path, 'test', model, model.args)

    with torch.no_grad():
        if model.args['edit_encoder']['type'] == 'treediff':
            feature_vecs = model.get_edit_encoding_by_batch(dataset.examples, batch_size=64)
        else:
            feature_vecs = model.get_edit_encoding_by_batch(dataset.examples, batch_size=256)
        print(f'decoded {feature_vecs.shape[0]} entries', file=sys.stderr)

    if 'csharp_fixer' in test_set_path:
        setname = 'csharp_fixer'
    elif 'train_debug' in test_set_path:
        setname = 'train_debug'
    elif 'dev_debug' in test_set_path:
        setname = 'dev_debug'
    elif 'train' in test_set_path:
        setname = 'train'
    elif 'dev' in test_set_path:
        setname = 'dev'
    elif 'test' in test_set_path:
        setname = 'test'
    else:
        raise Exception('Unknown data source!')
    save_path = model_path.replace(".bin", f".edit_vec_{setname}.bin")
    print(f'save edit vecs to {save_path}')
    pickle.dump(feature_vecs.cpu().numpy(), open(save_path, 'wb'))


if __name__ == '__main__':
    cmd_args = docopt(__doc__)

    # seed the RNG
    seed = int(cmd_args['--seed'])
    print(f'use random seed {seed}', file=sys.stderr)
    torch.manual_seed(seed)

    use_cuda = cmd_args['--cuda']
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed * 13 // 7)
    random.seed(seed * 13 // 7)

    if cmd_args['train']:
        train(cmd_args)
    elif cmd_args['imitation_learning']:
        imitation_learning(cmd_args)
    elif cmd_args['test_ppl']:
        test_ppl(cmd_args)
    elif cmd_args['decode_updated_data']:
        decode_updated_code(cmd_args)
    elif cmd_args['eval_csharp_fixer']:
        eval_csharp_fixer(cmd_args)
    elif cmd_args['collect_edit_vecs']:
        collect_edit_vecs(cmd_args)
    else:
        raise RuntimeError(f'invalid run mode')
