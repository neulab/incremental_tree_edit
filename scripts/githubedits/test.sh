#!/bin/bash

source activate structural_edits
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/scratch/yao.470/CMU_project/incremental_tree_edit:$PYTHONPATH

config_file=$1

work_dir=exp_githubedits_runs/FOLDER_OF_MODEL # TODO: replace FOLDER_OF_MODEL with your model dir

echo use config file ${config_file}
echo work dir=${work_dir}

mkdir -p ${work_dir}

# TODO: uncomment the test setting
# beam search
#test_file=source_data/githubedits/githubedits.test.jsonl
#OMP_NUM_THREADS=1 python -m exp_githubedits decode_updated_data \
#	  --cuda \
#    --beam_size=1 \
#    --evaluate_ppl \
#    ${work_dir}/model.bin \
#    ${test_file} 2>${work_dir}/model.bin.${filename}_test.log # dev_debug, dev, test, train_debug

# test ppl
#OMP_NUM_THREADS=1 python -m exp_githubedits test_ppl \
#	  --cuda \
#    --evaluate_ppl \
#    ${work_dir}/model.bin \
#    ${test_file} #2>${work_dir}/model.bin.decode_ppl.log


## csharp_fixer
#test_file=source_data/githubedits/csharp_fixers.jsonl
#scorer='default'
#OMP_NUM_THREADS=1 python -m exp_githubedits eval_csharp_fixer \
#	  --cuda \
#    --beam_size=1 \
#     --scorer=${scorer} \
#    ${work_dir}/model.bin \
#    ${test_file} 2>${work_dir}/model.bin.${filename}_csharp_fixer_${scorer}.log


## beam search on csharp_fixer with gold inputs
#test_file=source_data/githubedits/csharp_fixers.jsonl
#OMP_NUM_THREADS=1 python -m exp_githubedits decode_updated_data \
#	  --cuda \
#    --beam_size=1 \
#    ${work_dir}/model.bin \
#    ${test_file} 2>${work_dir}/model.bin.${filename}_csharp_fixer_gold.log


# collect edit vecs
##test_file=source_data/githubedits/csharp_fixers.jsonl
#test_file=source_data/githubedits/githubedits.test.jsonl
#OMP_NUM_THREADS=1 python -m exp_githubedits collect_edit_vecs \
#  --cuda \
#  ${work_dir}/model.bin \
#  ${test_file}

