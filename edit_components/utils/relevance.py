import sys, os, re
import numpy as np
from collections import OrderedDict
import sys, os, re
import numpy as np
from collections import OrderedDict
import pickle
import glob
import datetime, time
import scipy.spatial.distance as distance
from typing import List, Union, Any, Dict, Set

from edit_components.dataset import DataSet
from edit_components.change_entry import ChangeExample
from edit_components.utils.utils import get_entry_str


def get_rank_score(_candidate_scores):
    # DCG
    cum_score = 0.

    score_map = {1: 2, 2: 1, 3: 0}

    for i in range(len(_candidate_scores)):
        cand_id, cand_score = _candidate_scores[i]
        rank = i + 1

        rel_score = score_map[cand_score]
        cur_score = (np.exp(rel_score) - 1) / float(np.log2(rank + 1))
        cum_score += cur_score

    return cum_score


def load_query_results(file_path, with_score=True):
    f = open(file_path)

    line = f.readline()
    assert line.startswith('***Seed Query***')
    query_id = f.readline().strip()
    query_id = query_id[len('Id:'):].strip()
    print(f'\tseed query {query_id}', file=sys.stderr)

    while not re.match('^\d+ neighbors', line):
        line = f.readline().strip()

    f.readline()

    candidate_scores = []

    while True:
        line = f.readline()

        if not line:
            break

        e_id = line[len('Id:'):].strip()
        while not line.startswith('Score:'):
            line = f.readline()
        if with_score:
            score = int(line[len('Score:'):].strip())
        else: score = None

        line = f.readline()
        assert line.startswith('*****')

        candidate_scores.append((e_id, score))

    f.close()

    return {'seed_change_id': query_id, 'candidate_changes_and_scores': candidate_scores}


def gather_all_query_results_from_annotations(annotation_folder, with_score=True):
    relevance_data = dict()
    for annotation_file in glob.glob(annotation_folder + '/*.*', recursive=True):
        if os.path.isfile(annotation_file):
            print(f'loading annotations from {annotation_file}', file=sys.stderr)
            result = load_query_results(annotation_file, with_score=with_score)
            seed_change_id = result['seed_change_id']
            candidate_changes_and_scores = result['candidate_changes_and_scores']
            print(f'\t{len(candidate_changes_and_scores)} entries', file=sys.stderr)

            relevance_data.setdefault(seed_change_id, dict()).update({k: v for k, v in candidate_changes_and_scores})

    return relevance_data


def dcg(candidate_changes_and_scores):
    # discounted cumulative gain
    cum_score = 0.

    score_map = {1: 2, 2: 1, 3: 0}

    for i in range(len(candidate_changes_and_scores)):
        cand_id, cand_score = candidate_changes_and_scores[i]
        rank = i + 1

        rel_score = score_map[cand_score]
        cur_score = (np.exp(rel_score) - 1) / float(np.log2(rank + 1))
        cum_score += cur_score

    return cum_score


def ndcg(candidate_changes_and_scores):
    # normalized discounted cumulative gain
    ranked_candidate_changes_and_scores = sorted(candidate_changes_and_scores, key=lambda x: x[1])
    idcg_score = dcg(ranked_candidate_changes_and_scores)
    dcg_score = dcg(candidate_changes_and_scores)

    ndcg = dcg_score / idcg_score
    return ndcg


def get_nn(dataset, feature_vecs, seed_query_id=None, K=30, dist_func=distance.cosine, return_self=False, query_vec=None):
    """get the top-K nearest neighbors given a seed query"""

    if seed_query_id:
        seed_query_idx = dataset.example_id_to_index[seed_query_id]
        query_vec = feature_vecs[seed_query_idx]

    example_distances = []
    for idx in range(len(dataset.examples)):
        if seed_query_id and return_self is False and idx == seed_query_idx:
            continue

        feat = feature_vecs[idx]
        dist = dist_func(feat, query_vec)
        example_distances.append((idx, dist))
    
    example_distances.sort(key=lambda x: x[1])
    results = []
    for idx, dist in example_distances[:K]:
        change_entry = dataset.examples[idx]
        results.append((change_entry, dist))
        
    return results


def generate_top_k_query_results(dataset: DataSet, feature_vecs: List, seed_query_ids: List[str], model_name=None, eval_folder=None, K=30, dist_func=distance.cosine):
    if eval_folder is None:
        assert model_name
        model_name = model_name.replace('|', '_').replace('/', '_')
        eval_folder = f"evaluation/{model_name}/{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"

    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    print(f'output query results to {eval_folder}', file=sys.stderr)

    for seed_query_id in seed_query_ids:
        print(f'processing {seed_query_id}', file=sys.stderr)
        seed_query_example = dataset.get_example_by_id(seed_query_id)
        neighbors = get_nn(dataset, feature_vecs, seed_query_id=seed_query_id, K=30)

        f_name = seed_query_id.replace('|', '_').replace('/', '_')
        f_name = os.path.join(eval_folder, f_name)

        with open(f_name, 'w') as f:
            f.write(f'***Seed Query***\n{get_entry_str(seed_query_example)}\n\n\n')
            f.write(f'{len(neighbors)} neighbors\n\n')
            for example, dist in neighbors:
                f.write(get_entry_str(example, dist=dist) + '\n')


def dump_aggregated_query_results_from_query_results_for_annotation(annotation_folders: List[str], output_folder: str, relevance_db: Dict, dataset: DataSet):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    aggregated_query_results = dict()
    for annotation_folder in annotation_folders:
        query_results = gather_all_query_results_from_annotations(annotation_folder, with_score=False)
        for seed_query_id, candidate_changes_and_scores in query_results.items():
            aggregated_query_results.setdefault(seed_query_id, dict()).update(candidate_changes_and_scores)

    # filter out entries that has already be annotated
    for seed_query_id in aggregated_query_results:
        candidate_ids = list(aggregated_query_results[seed_query_id].keys())
        if seed_query_id in relevance_db:
            not_annotated_candidate_ids = [id for id in candidate_ids if id not in relevance_db[seed_query_id]]
            candidate_ids = not_annotated_candidate_ids

        f_name = seed_query_id.replace('|', '_').replace('/', '_')
        f_name = os.path.join(output_folder, f_name)
        seed_query_example = dataset.get_example_by_id(seed_query_id)

        with open(f_name, 'w') as f:
            f.write(f'***Seed Query***\n{get_entry_str(seed_query_example)}\n\n\n')
            f.write(f'{len(candidate_ids)} neighbors\n\n')
            np.random.shuffle(candidate_ids)
            for cand_id in candidate_ids:
                example = dataset.get_example_by_id(cand_id)
                f.write(get_entry_str(example, dist=0.0) + '\n')


def generate_reranked_list(model, relevance_db, dataset, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ranked_lists = dict()
    for seed_change_id, annotated_candidates in relevance_db.items():
        print(f'processing {seed_change_id}')
        seed_change_idx = [i for i, e in enumerate(dataset.examples) if e.id == seed_change_id][0]
        seed_change = dataset.examples[seed_change_idx]
        seed_change_feat_vec = model.code_change_encoder.encode_code_changes([seed_change], code_encoder=model.sequential_code_encoder, batch_size=1)[0]
        candidate_distances = []
        
        f_name = seed_change_id.replace('|', '_').replace('/', '_')
        f_name = os.path.join(output_folder, f_name)
        with open(f_name, 'w') as f:
            f.write(f'***Seed Query***\n{get_entry_str(seed_change)}\n\n\n')
            
            candidate_ids = list(annotated_candidates)
            candidate_examples = [dataset.get_example_by_id(x) for x in candidate_ids]
            cand_feature_vecs = model.code_change_encoder.encode_code_changes(candidate_examples, code_encoder=model.sequential_code_encoder, batch_size=256)

            for candidate_id, candidate_example, candidate_feat_vec in zip(candidate_ids, candidate_examples, cand_feature_vecs):
                # print(f'\tevaluate {candidate_id}')

                dist = distance.cosine(seed_change_feat_vec, candidate_feat_vec)
                # dist = get_distance(seed_change, candidate_example, seed_change_feat_vec, candidate_feat_vec)
                assert not np.isnan(dist)

                candidate_distances.append((candidate_id, dist))
            
            ranked_candidates = sorted(candidate_distances, key=lambda x: x[1])
            for candidate_id, dist in ranked_candidates:
                candidate_score = relevance_db[seed_change_id][candidate_id]
                candidate = dataset.get_example_by_id(candidate_id)
                f.write(get_entry_str(candidate, dist=dist, score=candidate_score) + '\n')
            
        ranked_lists[seed_change_id] = [candidate_id for candidate_id, dist in ranked_candidates]

    save_to = os.path.join(output_folder, 'ranked_lists.bin')
    pickle.dump(ranked_lists, open(save_to, 'bw'))
    print(f'save results to {save_to}')


if __name__ == '__main__':
    eval_folder = sys.argv[1]

    print(f'evaluating folder {eval_folder}', file=sys.stderr)

    files = filter(lambda x: x, os.listdir(eval_folder))
    eval_files_scores = OrderedDict()

    for eval_file in files:
        print(f'evaluating {eval_file}', file=sys.stderr)
        full_file_path = os.path.join(eval_folder, eval_file)
        f = open(full_file_path)

        line = f.readline()
        assert line.startswith('***Seed Query***')
        query_id = f.readline().strip()
        query_id = query_id[len('Id:'):].strip()
        print(f'\tseed query {query_id}', file=sys.stderr)

        while not re.match('^\d+ neighbors', line):
            line = f.readline().strip()

        f.readline()

        candidate_scores = []

        while True:
            line = f.readline()

            if not line:
                break

            e_id = line[len('Id:'):].strip()
            while not line.startswith('Score:'):
                line = f.readline()
            score = int(line[len('Score:'):].strip())
            line = f.readline()
            assert line.startswith('*****')

            candidate_scores.append((e_id, score))

        eval_files_scores[query_id] = candidate_scores

        f.close()

    print('', file=sys.stderr)
    rank_scores = []
    for query_id, candidate_scores in eval_files_scores.items():
        rank_score = get_rank_score(candidate_scores)
        print(f'{query_id}\t{rank_score}', file=sys.stderr)

        rank_scores.append(rank_score)

    print(f'\nAverage rank score: {np.average(rank_scores)}', file=sys.stderr)
