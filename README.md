# Learning Structural Edits via Incremental Tree Transformations

Code for ["Learning Structural Edits via Incremental Tree Transformations" (ICLR'21)](https://openreview.net/pdf?id=v9hAX77--cZ)

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

 
## 3. Experiments
See training and test scripts in `scripts/githubedits/`. Please configure the `PYTHONPATH` environment variable in line 6.
 
### 3.1 Training
For training, uncomment the desired setting in `scripts/githubedits/train.sh` and run:
```
bash scripts/githubedits/train.sh source_data/githubedits/configs/CONFIGURATION_FILE
```
where `CONFIGURATION_FILE` is the json file of your setting. 
Please check out the `TODO`'s in [`scripts/githubedits/train.sh`](scripts/githubedits/train.sh).


#### 3.1.1 Supervised Learning
For example, if you want to train Graph2Edit + Sequence Edit Encoder on GitHubEdits's 20\% sample data, 
please uncomment only line 22-26 in `scripts/githubedits/train.sh` and run:
```
bash scripts/githubedits/train.sh source_data/githubedits/configs/graph2iteredit.seq_edit_encoder.20p.json
```
**Note**: 
- When you run the experiment for the first time, you might need to wait for ~15 minutes for data preprocessing.
- By default, the data preprocessing includes generating and saving the target edit sequences for instances in the training data.
    However, this may cause a `out of (cpu) memory` issue. **A simple way to solve this problem is to set `--small_memory` in the `train.sh` script.**
    We explained the details in [Section 4.2 Out of Memory Issue](#42-out-of-memory-issue).
     


#### 3.1.2 Imitation Learning
To further train the model with PostRefine imitation learning, 
please replace `FOLDER_OF_SUPERVISED_PRETRAINED_MODEL` with your model dir in `source_data/githubedits/configs/graph2iteredit.seq_edit_encoder.20p.postrefine.imitation.json`.
Uncomment only line 27-31 in `scripts/githubedits/train.sh` and run:
```
bash scripts/githubedits/train.sh source_data/githubedits/configs/graph2iteredit.seq_edit_encoder.20p.postrefine.imitation.json
```
Note that `--small_memory` cannot be used in this setting.

### 3.2 Test
To test a trained model, first uncomment only the desired setting in `scripts/githubedits/test.sh` and replace `work_dir` with your model directory, 
and then run:
```
bash scripts/githubedits/test.sh
```
Please check out the `TODO`'s in [`scripts/githubedits/test.sh`](scripts/githubedits/test.sh).

## 4. FAQ

### 4.1 Applications to Other Languages
<!-- Todo: when applying the model to a different programming language, how to check/debug the ASDL grammar implementation, the abstract syntactic tree preprocessing, and the ground-truth edit sequence generation?-->
 
In principle, our framework can work with various programming languages. 
To this end, several changes are needed:
1. Implementing a language-specific `ASDLGrammar` class for the new language. 
    - This class could inherit the [`asdl.asdl.ASDLGrammar`](asdl/asdl.py) class.
    - Basic functions should include 
        - Defining the `primitive` and `composite` types, 
        - Implementing the class constructor (e.g., converting from the `.xml` or `.txt` syntax descriptions),
        - Converting the source AST data into an `asdl.asdl_ast.AbstractSyntaxTree` object.
    - Example: see the [`asdl.lang.csharp.CSharpASDLGrammar`](asdl/lang/csharp/csharp_grammar.py) class.
    - **Sanity check**: It is very helpful to implement a `demo_edits.py` file like [this one for csharp](asdl/lang/csharp/demo_edits.py) and 
        make sure you have checked out the generated ASTs and target edit sequences.
    - **Useful resource**: The [TranX](https://github.com/pcyin/tranx) library contains ASDLGrammar classes for some other languages.
        _Note that we have revised the `asdl.asdl.ASDLGrammar` class so directly using the TranX implementation may not work._
        However, this resource is still a good starting point; you may consider modify it based on the sanity check outputs.
        
2. Implementing a language-specific `TransitionSystem` class.
    - The target edit sequences (of the training data) are calculated by `trees.substitution_system.SubstitutionSystem`,
        which depends on a `asdl.transition_system.TransitionSystem` object (or its inheritor) (see [reference](trees/substitution_system.py#L16)). 
    - In our current implementation of CSharp, we have reused the `CSharpTransitionSystem` class implemented in the [Graph2Tree library](https://github.com/microsoft/iclr2019-learning-to-represent-edits).
        However, only the `get_primitive_field_actions` function of the `TransitionSystem` class is actually used by the `SubstitutionSystem` ([example](trees/substitution_system.py#L131)). 
        Therefore, for simplicity, one can only implement only this function. 
        Basically, this `get_primitive_field_actions` function defines how the leaf string should be generated 
        (e.g., multiple `GenTokenAction` actions should be taken for generating a multi-word leaf string), which we will discuss next.

3. Customizing the leaf string generation.
    - Following the last item, one may also need to customize the `GenTokenAction` action especially about whether and how the [stop signal](asdl/transition_system.py#L29) will be used.
        For CSharp, we do not use detect any stop signal as in our datasets the leaf string is typically one single-word token.
        However, it will be needed when the leaf string contains multiple words.
    - Accordingly, one may customize the [`Add`](/trees/edits.py#L61) edit action and the [`SubstitutionSystem`](trees/substitution_system.py#L170) 
        regarding how the leaf string should be added to the current tree.
        

### 4.2 Out of Memory Issue
**The issue:**
By default, the data preprocessing step will 
(1) run a dynamic programming algorithm to calculate the shortest edit sequence `(a_1, a_2, ..., a_T)`
    as the target edit sequence for each code pair `(C-, C+)`, and 
(2) save every intermediate tree graph `(g_1, g_2, ..., g_T)`, where `g_{t+1}` is the transformation result of 
    applying edit action `a_t` to tree `g_t` at time step `t`, as the input to the tree encoder (see [3.1.2 in our paper](https://openreview.net/pdf?id=v9hAX77--cZ)).
Therefore, a completely preprocessed training set has a very large size and will take up a lot of CPU memory
every time you load the data for model training.

**The solution:**
A simple solution is to avoid saving any intermediate tree graph, 
i.e., we will only save the shortest edit sequence results from (1) 
while leaving the generation of intermediate tree graphs in (2) to during the model training.
This can be done by set `--small_memory` in the [train.sh](scripts/githubedits/train.sh) script. 
_Currently this option can only be used for regular supervised learning; for imitation learning, this has to be off._

Note that there will be a trade-off between the CPU memory and the GPU utility/training speed, 
since the generation of the intermediate tree graphs is done at the CPU level.
