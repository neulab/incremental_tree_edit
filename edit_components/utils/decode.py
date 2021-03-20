#!/usr/bin/env python
"""Neural Representations of code revisions

Usage:
    decode.py rerank --mode=<str> --dataset=<file> --relevance_db=<file> [options] MODEL_PATH
    decode.py decode_change_vec --mode=<str> --dataset=<file> [options] MODEL_PATH

Options:
    -h --help                               Show this screen.
    --mode=<str>                            Mode: seq2seq|seq2tree|tree2tree_subtree_copy
    --dataset=<file>                        Dataset
    --relevance_db=<file>                   Relevance DB path
    --save_to=<file>                        Save decode results to [default: None]
    --cuda                                  Use gpu
"""

from edit_model.editor import SequentialAutoEncoder, TreeBasedAutoEncoderWithGraphEncoder
from edit_model.edit_encoder import GraphChangeEncoder
from edit_components.dataset import DataSet
from asdl.lang.csharp.csharp_transition import CSharpTransitionSystem
from asdl.lang.csharp.csharp_grammar import CSharpASDLGrammar
from asdl.lang.csharp.csharp_hypothesis import CSharpHypothesis
from edit_components.utils.utils import *
from edit_components.utils.relevance import generate_reranked_list
import urllib.request as request
from tqdm import tqdm_notebook
import json
from docopt import docopt
import pickle, datetime, time


def dump_rerank_file(args):
    dataset_file = args['--dataset']
    print(f'load dataset {dataset_file}')
    dataset = DataSet.load_from_jsonl(dataset_file, type='sequential', max_workers=10)

    model_file = args['MODEL_PATH']
    print(f'load model from {model_file}')

    if args['--mode'] == 'seq2seq':
        model_cls = SequentialAutoEncoder
    elif args['--mode'].startswith('tree2tree'):
        model_cls = TreeBasedAutoEncoderWithGraphEncoder
    else:
        model_cls = TreeBasedAutoEncoder

    model = model_cls.load(model_file)
    model.eval()
    # print(model.training)
    print(model.args)

    # feature_vecs = model.code_change_encoder.encode_code_changes(dataset.examples, code_encoder=model.sequential_code_encoder, batch_size=256)
    # print(f'decoded {feature_vecs.shape[0]} entries')

    print(f"load relevance db from {args['--relevance_db']}")
    relevance_db = pickle.load(open(args['--relevance_db'], 'rb'))

    algo_label = [x for x in model_file.split('/') if 'branch' in x][0]
    model_label = [x for x in model_file.split('/') if x.endswith('.bin')][0]
    output_folder = f"reranking/{algo_label}/{model_label}-{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"

    generate_reranked_list(model, relevance_db, dataset, output_folder)


def decode_change_vec(args):
    dataset_file = args['--dataset']
    model_file = args['MODEL_PATH']
    print(f'load model from {model_file}')

    if args['--mode'] == 'seq2seq':
        model_cls = SequentialAutoEncoder
    elif args['--mode'].startswith('tree2tree'):
        model_cls = TreeBasedAutoEncoderWithGraphEncoder
    else:
        model_cls = TreeBasedAutoEncoder

    model = model_cls.load(model_file, use_cuda=args['--cuda'])
    model.eval()
    print(model.args)

    dataset_file = args['--dataset']
    print(f'load dataset {dataset_file}')

    is_graph_change_encoder = isinstance(model_cls.code_change_encoder, GraphChangeEncoder)

    dataset = DataSet.load_from_jsonl(type='tree2tree_subtree_copy' if is_graph_change_encoder else 'sequential',
                                      transition_system=CSharpTransitionSystem(model.grammar) if '2tree' in args['--mode'] else None,
                                      max_workers=1,
                                      parallel=False,
                                      annotate_tree_change=is_graph_change_encoder)

    change_vecs = model.code_change_encoder.encode_code_changes(dataset.examples, code_encoder=model.sequential_code_encoder, batch_size=256)
    print(f'decoded {change_vecs.shape[0]} entries')

    save_to = args['--save_to']
    if save_to == 'None':
        save_to = model_file + '.change_vec.pkl'

    pickle.dump(change_vecs, open(save_to, 'wb'))
    print(f'saved decoding results to {save_to}')

    return change_vecs


if __name__ == '__main__':
    args = docopt(__doc__)

    if args['rerank']:
        dump_rerank_file(args)
    elif args['decode_change_vec']:
        decode_change_vec(args)
