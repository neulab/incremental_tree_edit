import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]


import json
import sys
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool

import numpy as np

from asdl.lang.csharp.csharp_hypothesis import CSharpHypothesis
from asdl.lang.csharp.csharp_transition import ApplyRuleAction, ApplySubTreeAction
from edit_components.change_entry import ChangeExample
from edit_model.edit_encoder import SequentialChangeEncoder, GraphChangeEncoder, BagOfEditsChangeEncoder
from edit_model.encdec import SequentialDecoder


def _encode(word_list):
    return [w.replace('\n', '-NEWLINE-') for w in word_list]


def load_one_change_entry_csharp(json_str, editor_type='seq2seq', edit_encoder_type='seq', tensorization=True, debug=False,
                                 transition_system=None, substitution_system=None, vocab=None, args=None):
    entry = json.loads(json_str)
    previous_code_chunk = _encode(entry['PrevCodeChunkTokens'])
    updated_code_chunk = _encode(entry['UpdatedCodeChunkTokens'])
    context = _encode(entry['PrecedingContext'] + ['|||'] + entry['SucceedingContext'])

    if editor_type == 'seq2seq':
        prev_code_ast_json = entry['PrevCodeAST']
        prev_code_ast = transition_system.grammar.get_ast_from_json_obj(prev_code_ast_json)

        updated_code_ast_json = entry['UpdatedCodeAST']
        updated_code_ast = transition_system.grammar.get_ast_from_json_obj(updated_code_ast_json)

        example = ChangeExample(id=entry['Id'],
                                prev_data=previous_code_chunk,
                                updated_data=updated_code_chunk,
                                raw_prev_data=entry['PrevCodeChunk'],
                                raw_updated_data=entry['UpdatedCodeChunk'],
                                context=context,
                                prev_code_ast=prev_code_ast,
                                updated_code_ast=updated_code_ast)

        # preform tensorization
        if tensorization:
            if edit_encoder_type == 'sequential':
                SequentialChangeEncoder.populate_aligned_token_index_and_mask(example)
            elif edit_encoder_type == 'graph':
                example.change_edges = GraphChangeEncoder.compute_change_edges(example)

            # SequentialChangeEncoder.populate_aligned_token_index_and_mask(example)
            SequentialDecoder.populate_gen_and_copy_index_and_mask(example, vocab, copy_token=args['decoder']['copy_token'])

    elif editor_type in ('graph2tree', 'graph2iteredit'):
        prev_code_ast_json = entry['PrevCodeAST']
        prev_code_ast = transition_system.grammar.get_ast_from_json_obj(prev_code_ast_json)

        updated_code_ast_json = entry['UpdatedCodeAST']
        updated_code_ast = transition_system.grammar.get_ast_from_json_obj(updated_code_ast_json)

        if editor_type == 'graph2tree':
            tgt_actions = transition_system.get_decoding_actions(target_ast=updated_code_ast,
                                                                 prev_ast=prev_code_ast,
                                                                 copy_identifier=args['decoder']['copy_identifier_node'])
        else:
            prev_code_ast.reindex_w_dummy_reduce()
            updated_code_ast.reindex_w_dummy_reduce()
            tgt_actions = substitution_system.get_decoding_edits_fast(prev_code_ast, updated_code_ast,
                                                                      bool_copy_subtree=args['decoder']['copy_subtree'],
                                                                      init_code_tokens=previous_code_chunk,
                                                                      bool_debug=args['debug'])

        if debug:
            # print('Prev Code')
            # print(entry['PrevCodeChunk'])
            #
            # print('Updated Code')
            # print(entry['UpdatedCodeChunk'])

            # sys.stdout.flush()
            #
            # action_paths = transition_system.get_all_decoding_action_paths(target_ast=updated_code_ast, prev_ast=prev_code_ast, sample_size=10)
            #
            action_paths = [tgt_actions]
            for tgt_actions in action_paths:
                # sanity check target decoding actions
                hyp = CSharpHypothesis()
                for decode_action in tgt_actions:
                    assert any(
                        isinstance(decode_action, cls) for cls in transition_system.get_valid_continuation_types(hyp))
                    if isinstance(decode_action, ApplyRuleAction):
                        assert decode_action.production in transition_system.get_valid_continuating_productions(hyp)

                    if isinstance(decode_action, ApplySubTreeAction) and hyp.frontier_field:
                        assert decode_action.tree.production.type in transition_system.grammar.descendant_types[
                            hyp.frontier_field.type]
                        # if decode_action.tree.production.type != hyp.frontier_field.type and decode_action.tree.production.type not in hyp.frontier_field.type.child_types:
                        #     print(decode_action.tree.production.type, hyp.frontier_field.type.child_types)

                    if hyp.frontier_node:
                        assert hyp.frontier_field == decode_action.frontier_field
                        assert hyp.frontier_node.production == decode_action.frontier_prod

                    hyp.apply_action(decode_action)
                assert hyp.tree.to_string() == updated_code_ast.root_node.to_string()
                assert hyp.tree == updated_code_ast.root_node
                assert hyp.completed

        example = ChangeExample(id=entry['Id'],
                                prev_data=previous_code_chunk,
                                updated_data=updated_code_chunk,
                                raw_prev_data=entry['PrevCodeChunk'],
                                raw_updated_data=entry['UpdatedCodeChunk'],
                                context=context,
                                prev_code_ast=prev_code_ast,
                                updated_code_ast=updated_code_ast,
                                tgt_actions=tgt_actions)

        # preform tensorization
        if tensorization:
            if edit_encoder_type == 'sequential':
                SequentialChangeEncoder.populate_aligned_token_index_and_mask(example)
            elif edit_encoder_type == 'graph':
                example.change_edges = GraphChangeEncoder.compute_change_edges(example)
    else:
        # raise ValueError('unknown dataset type')
        example = ChangeExample(id=entry['Id'],
                                prev_data=previous_code_chunk,
                                updated_data=updated_code_chunk,
                                raw_prev_data=entry['PrevCodeChunk'],
                                raw_updated_data=entry['UpdatedCodeChunk'],
                                context=context)

    return example


class DataSet:
    def __init__(self, examples):
        self.examples = examples
        self.example_id_to_index = OrderedDict([(e.id, idx) for idx, e in enumerate(self.examples)])

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            # sort by the length of the change sequence in descending order
            batch_examples.sort(key=lambda e: -len(e.change_seq))

            yield batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def get_example_by_id(self, eid):
        idx = self.example_id_to_index[eid]
        return self.examples[idx]

    @staticmethod
    def load_from_jsonl(file_path, language='csharp', editor=None,
                        editor_type=None, edit_encoder_type=None, args=None, vocab=None, transition_system=None,
                        substitution_system=None, tensorization=True, from_ipython=False, max_workers=1, debug=False):

        from edit_model.editor import Seq2SeqEditor, Graph2TreeEditor, Graph2IterEditEditor

        if editor:
            if isinstance(editor, Seq2SeqEditor):
                editor_type = 'seq2seq'
            elif isinstance(editor, Graph2TreeEditor):
                editor_type = 'graph2tree'
            elif isinstance(editor, Graph2IterEditEditor):
                editor_type = 'graph2iteredit'

            if isinstance(editor.edit_encoder, SequentialChangeEncoder):
                edit_encoder_type = 'sequential'
            elif isinstance(editor.edit_encoder, GraphChangeEncoder):
                edit_encoder_type = 'graph'
            elif isinstance(editor.edit_encoder, BagOfEditsChangeEncoder):
                edit_encoder_type = 'bag'

            if hasattr(editor, 'transition_system'):
                transition_system = editor.transition_system

            if hasattr(editor, 'substitution_system'):
                substitution_system = editor.substitution_system

            vocab = editor.vocab
            args = editor.args

        if editor_type is None:
            print("WARNING: unknown dataset type")

        if language == 'csharp':
            load_one_change_entry = load_one_change_entry_csharp
        else:
            raise Exception(f"unavailable language={language}")

        examples = []
        with open(file_path) as f:
            print('reading all lines from the dataset', file=sys.stderr)
            all_lines = [l for l in f]
            print('%d lines. Done' % len(all_lines), file=sys.stderr)

            if from_ipython:
                from tqdm import tqdm_notebook
                iter_log_func = partial(tqdm_notebook, total=len(all_lines), desc='loading dataset')
            else:
                from tqdm import tqdm
                iter_log_func = partial(tqdm, total=len(all_lines), desc='loading dataset', file=sys.stdout)

            if max_workers > 1:
                print('Parallel data loading...', file=sys.stderr)
                with Pool(max_workers) as pool:
                    processed_examples = pool.map(partial(load_one_change_entry,
                                                          editor_type=editor_type,
                                                          edit_encoder_type=edit_encoder_type,
                                                          tensorization=tensorization,
                                                          transition_system=transition_system,
                                                          substitution_system=substitution_system,
                                                          vocab=vocab, args=args),
                                                   iterable=all_lines) # chunksize=min(1000, int(len(all_lines)/max_workers))
                    for example in iter_log_func(processed_examples):
                        examples.append(example)
            else:
                for line in iter_log_func(all_lines):
                    example = load_one_change_entry(line,
                                                    editor_type=editor_type,
                                                    edit_encoder_type=edit_encoder_type,
                                                    tensorization=tensorization,
                                                    transition_system=transition_system,
                                                    substitution_system=substitution_system,
                                                    vocab=vocab, args=args)
                    examples.append(example)

        data_set = DataSet([e for e in examples if e])

        return data_set
