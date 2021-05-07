# Learning Structural Edits via Incremental Tree Transformations

Code for ["Learning Structural Edits via Incremental Tree Transformations" (ICLR'21)](https://openreview.net/pdf?id=v9hAX77--cZ)


## 1. Prepare Environment
We recommend using `conda` to manage the environment:
```
conda env create -n "structural_edits" -f structural_edits.yml
conda activate structural_edits
```

Install the punkt tokenizer:
```
python
>>> import nltk
>>> nltk.download('punkt')
>>> <ctrl-D>
```

## 2. Data
Please extract the datasets and vocabulary files by:
```
cd source_data
tar -xzvf githubedits.tar.gz
```

All necessary source data has been included as the following:
```
| --source_data
|       |-- githubedits
|           |-- githubedits.{train|train_20p|dev|test}.jsonl
|           |-- csharp_fixers.jsonl
|           |-- vocab.from_repo.{080910.freq10|edit}.json
|           |-- Syntax.xml
|           |-- configs
|               |-- ...(model config json files)
```
A sample file containing 20% of the GitHubEdits training data is included as `source_data/githubedits/githubedits.train_20p.jsonl` for running small experiments.

We have generated and included the vocabulary files as well. To create your own vocabulary, see `edit_components/vocab.py`.

Copyright: The original data were downloaded from [Yin et al., (2019)](http://www.cs.cmu.edu/~pengchey/githubedits.zip).


### 2.1 Data Preprocessing and Sanity Check
Todo: when applying the model to a different programming language, how to check/debug the ASDL grammar implementation, the abstract syntactic tree preprocessing, and the ground-truth edit sequence generation?

 
## 3. Experiments
See training and test scripts in `scripts/githubedits/`. Please configure the `PYTHONPATH` environment variable in line 6.
 
### 3.1 Training
For training, uncomment the desired setting in `scripts/githubedits/train.sh` and run:
```
bash scripts/githubedits/train.sh source_data/githubedits/configs/CONFIGURATION_FILE
```
where `CONFIGURATION_FILE` is the json file of your setting. 


#### Supervised Learning
For example, if you want to train Graph2Edit + Sequence Edit Encoder on GitHubEdits's 20\% sample data, please uncomment only line 21-25 in `scripts/githubedits/train.sh` and run:
```
bash scripts/githubedits/train.sh source_data/githubedits/configs/graph2iteredit.seq_edit_encoder.20p.json
```
(Note: when you run the experiment for the first time, you might need to wait for ~15 minutes for data preprocessing.)


#### Imitation Learning
To further train the model with PostRefine imitation learning, 
please replace `FOLDER_OF_SUPERVISED_PRETRAINED_MODEL` with your model dir in `source_data/githubedits/configs/graph2iteredit.seq_edit_encoder.20p.postrefine.imitation.json`.
Uncomment only line 27-31 in `scripts/githubedits/train.sh` and run:
```
bash scripts/githubedits/train.sh source_data/githubedits/configs/graph2iteredit.seq_edit_encoder.20p.postrefine.imitation.json
```

### 3.2 Test
To test a trained model, first uncomment only the desired setting in `scripts/githubedits/test.sh` and replace `work_dir` with your model directory, 
and then run:
```
bash scripts/githubedits/test.sh
```

## 4. Reference
If you use our code and data, please cite our paper:
```
@inproceedings{yao2021learning,
    title={Learning Structural Edits via Incremental Tree Transformations},
    author={Ziyu Yao and Frank F. Xu and Pengcheng Yin and Huan Sun and Graham Neubig},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=v9hAX77--cZ}
}
```

Our implementation is adapted from [TranX](https://github.com/pcyin/tranx) and [Graph2Tree](https://github.com/microsoft/iclr2019-learning-to-represent-edits).
We are grateful to the two work!
```
@inproceedings{yin18emnlpdemo,
    title = {{TRANX}: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation},
    author = {Pengcheng Yin and Graham Neubig},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP) Demo Track},
    year = {2018}
}
@inproceedings{yin2018learning,
    title={Learning to Represent Edits},
    author={Pengcheng Yin and Graham Neubig and Miltiadis Allamanis and Marc Brockschmidt and Alexander L. Gaunt},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=BJl6AjC5F7},
}
```
