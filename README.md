# SemEval-2013 Task 13 Word Sense Induction for Graded and Non-Graded Senses

A PKU course project based on the "SemEval-2013 task 13 Word Sense Induction for Graded and Non-Graded Senses" competition.

Three Subtask:

1. Non-graded Word Sense Iduction Subtask
2. Graded Word Sense Induction Subtask
3. Lexical Substitution SubTask

> The link of description for each subtask is broken.

## Getting Started

> If it's your first time using nltk's wordNet API. You have to [install](#NLTK-API) first.

```sh
# install the dependencies
pip3 install -r requirements.txt

# get fastText pre-trained embedding
bash fastTextPretrainEmbedding.sh

# get the topN key file
python3 similarity.py

# evaluate
./evaluate.sh ResultFolder/top2.key
# or
python3 evaluation.py ResultFolder/top2.key
```

```sh
# plot TextCNN Model
brew install graphviz
python3 textCNN.py
```

### File Purpose

- `corpus.py` - load the SemEval xml data into `Lexelt` object
- `embedding.py` - sentence embedding based on fastText
- `similarity.py` - calculating similarity between test data sentence with wordNet definitions

## Task Overview

[Task Description in Detail](SemEval-2013-Task-13-test-data/README.md)

### [Given Data](SemEval-2013-Task-13-test-data/README.md#DIRECTORY-LAYOUT-AND-FILE-DESCRIPTIONS)

- Contexts
  - senseval2-format
  - xml-format
- Keys
  - [baselines](SemEval-2013-Task-13-test-data/README.md#BASELINES)
  - gold
  - systems
- Scoring - [Evaluation](SemEval-2013-Task-13-test-data/README.md#EVALUATION)

### [ukWaC - Corpus for WSI](SemEval-2013-Task-13-test-data/README.md#TRAINING-DATA)

- [WaCky - The Web-As-Corpus Kool Yinitiative](https://wacky.sslmit.unibo.it/doku.php)
  - [paper](https://wacky.sslmit.unibo.it/lib/exe/fetch.php?media=papers:wacky_2008.pdf)
  - [tagset](https://wacky.sslmit.unibo.it/lib/exe/fetch.php?media=tagsets:ukwac_tagset.txt)

### Evaluation

WSD F1

- Jaccard Index
- Positionally-Weighted Kendall's $\tau$
- Weighted NDCG

Sense Cluster Comparison

- Fuzzy NMI
- Fuzzy B-Cubed

#### Weighted NDCG

> DCG stands for Discounted Cumulative Gain

#### Fuzzy NMI

> NMI stands for Normalized Mutual Information

- [Wiki - Mutual information](https://en.wikipedia.org/wiki/Mutual_Information)

## Our Works

### The TopN Approach

Big Picture

1. Use the definitions of word from WordNet / or From the test data
   1. Use WordNet's explainations to represent the meaning
   2. Use the context to represent the meaning
2. Transfer sentences into vectors
3. Calculate the similarity between each definition sentences
4. For each test data export N possible sense with the most similarity
   1. Each of their weights minus the N+1th's weight

#### Embedding

- [fastText Pretrained English word vectors](https://fasttext.cc/docs/en/english-vectors.html)
- [imgarylai/bert-embedding](https://github.com/imgarylai/bert-embedding)

  ```py
  from bert_embedding import BertEmbedding

  bert_embedding = BertEmbedding()
  result = bert_embedding(sentence)
  ```

  ```txt
  ERROR: mxnet 1.4.0 has requirement numpy<1.15.0,>=1.8.2, but you'll have numpy 1.16.3 which is incompatible.
  ERROR: bert-embedding 1.0.1 has requirement numpy==1.14.6, but you'll have numpy 1.16.3 which is incompatible.
  ```

#### Construct Word Embedding to Sentence Embedding

1. Naive Adding
2. Naive Adding with Normalization
3. Padded Naive Adding with Normalization
4. TextCNN
   - [bhaveshoswal/CNN-text-classification-keras](https://github.com/bhaveshoswal/CNN-text-classification-keras)
   - [Keras Model class API](https://keras.io/models/model/)

![TextCNN Model](images/TextCNN_model.png)

### The BiLM ELMo Approach

> Follow the thought of the paper: Word Sense Induction with Neural biLM and Symmetric Patterns

![BiLM Model](images/BiLM_model.png)

## Evaluation Result

### Naive Adding Strict vs. Generalized TopN Approach Result

> The scentence embedding is simply adding by every word embeddings in the sentence

#### Strict TopN Approach Result

> Output of TopN must be N results.

| model | Embedding  | Similarity | Jac. Ind. | tau   | WDCG  | Fuzzy NMI | Fuzzy B-Cube | AVG   |
| ----- | ---------- | ---------- | --------- | ----- | ----- | --------- | ------------ | ----- |
| Top1  | fastText   | Cosine     | 23.81     | 61.84 | 29.30 | 2.81      | 50.22        | 11.89 |
| Top2  | fastText   | Cosine     | 19.62     | 59.88 | 35.98 | 8.56      | 52.10        | 21.12 |
| Top3  | fastText   | Cosine     | 19.22     | 59.61 | 37.90 | 7.15      | 42.87        | 17.51 |
| Top4  | fastText   | Cosine     | 19.20     | 59.58 | 37.47 | 6.61      | 32.07        | 14.56 |
| Top5  | fastText   | Cosine     | 19.20     | 59.50 | 36.25 | 5.92      | 24.80        | 12.11 |
| Top1  | fastText   | Euclidean  | 24.45     | 61.35 | 28.96 | 3.63      | 47.78        | 13.17 |
| Top2  | fastText   | Euclidean  | 21.25     | 60.19 | 35.47 | 7.87      | 46.06        | 19.04 |
| Top3  | fastText   | Euclidean  | 20.09     | 60.09 | 36.99 | 6.97      | 40.36        | 16.77 |
| Top4  | fastText   | Euclidean  | 19.51     | 59.90 | 36.00 | 6.46      | 34.25        | 14.87 |
| Top5  | fastText   | Euclidean  | 19.44     | 59.72 | 34.59 | 6.15      | 28.33        | 13.20 |
| Top1  | fastText   | Minkowski  | 24.45     | 61.35 | 28.96 | 3.63      | 47.78        | 13.17 |
| Top2  | fastText   | Minkowski  | 21.25     | 60.19 | 35.47 | 7.87      | 46.06        | 19.04 |
| Top3  | fastText   | Minkowski  | 20.09     | 60.09 | 36.99 | 6.97      | 40.36        | 16.77 |
| Top4  | fastText   | Minkowski  | 19.51     | 59.90 | 36.00 | 6.46      | 34.25        | 14.87 |
| Top5  | fastText   | Minkowski  | 19.44     | 59.72 | 34.59 | 6.15      | 28.33        | 13.20 |
| Top2  | Bert Model | Cosine     | 20.91     | 60.04 | 34.62 | 6.74      | 46.78        | 17.76 |

#### Generalized TopN Approach Result

> Output of TopN must be less than N results.

| model | Embedding | Similarity | Jac. Ind. | tau   | WDCG  | Fuzzy NMI | Fuzzy B-Cube | AVG   |
| ----- | --------- | ---------- | --------- | ----- | ----- | --------- | ------------ | ----- |
| Top1  | fastText  | Cosine     | 23.81     | 61.84 | 29.30 | 2.81      | 50.22        | 11.89 |
| Top2  | fastText  | Cosine     | 19.95     | 59.90 | 35.70 | 8.35      | 52.40        | 20.92 |
| Top3  | fastText  | Cosine     | 19.28     | 59.58 | 37.81 | 7.15      | 44.16        | 17.77 |
| Top4  | fastText  | Cosine     | 19.21     | 59.60 | 37.46 | 6.47      | 35.66        | 15.19 |
| Top5  | fastText  | Cosine     | 19.20     | 59.51 | 36.23 | 5.88      | 28.48        | 12.94 |

### Naive adding divided by sentence length Strict TopN Approach Result

> The scentence embedding is simply adding by every word embeddings in the sentence, and then divided by its length

| model | Embedding | Similarity | Jac. Ind. | tau   | WDCG  | Fuzzy NMI | Fuzzy B-Cube | AVG   |
| ----- | --------- | ---------- | --------- | ----- | ----- | --------- | ------------ | ----- |
| Top1  | fastText  | Cosine     | 24.52     | 62.11 | 29.89 | 3.38      | 50.37        | 13.04 |
| Top2  | fastText  | Cosine     | 19.68     | 60.30 | 36.25 | 8.92      | 52.51        | 21.64 |
| Top3  | fastText  | Cosine     | 19.23     | 59.48 | 38.24 | 7.43      | 43.24        | 17.92 |
| Top4  | fastText  | Cosine     | 19.20     | 59.61 | 37.65 | 6.74      | 32.11        | 14.71 |
| Top5  | fastText  | Cosine     | 19.20     | 59.64 | 36.20 | 6.16      | 25.00        | 12.41 |

### Padding sentence divided by max sentence length Strict TopN Approach Result

> The sentence embedding is padding to the max sentence length using "AVG" embedding.

| model | Embedding | Similarity | Jac. Ind. | tau   | WDCG  | Fuzzy NMI | Fuzzy B-Cube | AVG   |
| ----- | --------- | ---------- | --------- | ----- | ----- | --------- | ------------ | ----- |
| Top1  | fastText  | Cosine     | 23.37     | 61.38 | 29.11 | 3.73      | 52.16        | 13.95 |
| Top2  | fastText  | Cosine     | 20.52     | 59.82 | 35.28 | 8.55      | 46.58        | 19.95 |
| Top3  | fastText  | Cosine     | 19.80     | 59.76 | 37.11 | 7.88      | 37.99        | 17.30 |
| Top4  | fastText  | Cosine     | 18.66     | 59.96 | 35.99 | 7.36      | 30.82        | 15.06 |
| Top5  | fastText  | Cosine     | 18.39     | 59.79 | 34.66 | 6.95      | 25.37        | 13.28 |

### BERT Embedding generated with model not vector table

| model | Embedding | Similarity | Jac. Ind. | tau   | WDCG  | Fuzzy NMI | Fuzzy B-Cube | AVG   |
| ----- | --------- | ---------- | --------- | ----- | ----- | --------- | ------------ | ----- |
| Top2  | BERT      | Cosine     | 19.65     | 59.89 | 35.66 | 7.93      | 53.06        | 20.51 |

### Deprecated

#### Generalized TopN Approach with at least M Result

| model(M) | Embedding | Similarity | Sentence Embedding | Jac. Ind. | tau   | WDCG  | Fuzzy NMI | Fuzzy B-Cube | AVG   |
| -------- | --------- | ---------- | ------------------ | --------- | ----- | ----- | --------- | ------------ | ----- |
| Top3(2)  | fastText  | Cosine     | NaiveAdding        | 19.30     | 59.66 | 38.15 | 7.45      | 44.58        | 18.23 |
| Top4(2)  | fastText  | Cosine     | NaiveAdding        | 19.21     | 59.60 | 37.63 | 6.70      | 35.94        | 15.52 |
| Top5(2)  | fastText  | Cosine     | NaiveAdding        | 19.20     | 59.66 | 36.19 | 6.11      | 28.94        | 13.30 |
| Top3(2)  | fastText  | Cosine     | NaiveNormalized    | 19.30     | 59.66 | 38.15 | 7.45      | 44.58        | 18.23 |
| Top4(2)  | fastText  | Cosine     | NaiveNormalized    | 19.21     | 59.60 | 37.63 | 6.70      | 35.94        | 15.52 |
| Top5(2)  | fastText  | Cosine     | NaiveNormalized    | 19.20     | 59.66 | 36.19 | 6.11      | 28.94        | 13.30 |

#### Generalized TopN Approach without minus the N+1 similarity

| model | Embedding | Similarity | Sentence Embedding | Jac. Ind. | tau   | WDCG  | Fuzzy NMI | Fuzzy B-Cube | AVG   |
| ----- | --------- | ---------- | ------------------ | --------- | ----- | ----- | --------- | ------------ | ----- |
| Top2  | fastText  | Cosine     | NaiveAdding        | 20.30     | 60.35 | 37.51 | 6.62      | 52.20        | 18.59 |
| Top3  | fastText  | Cosine     | NaiveAdding        | 19.32     | 59.77 | 36.82 | 6.80      | 39.72        | 16.43 |
| Top2  | fastText  | Cosine     | NaiveAvgPadding    | 22.20     | 61.61 | 31.98 | 3.78      | 53.26        | 14.19 |
| Top3  | fastText  | Cosine     | NaiveAvgPadding    | 20.92     | 60.78 | 32.62 | 4.89      | 51.34        | 15.84 |

## Links

- [Wiki - SemEval](https://en.wikipedia.org/wiki/SemEval)
- [ACL Anthology](https://www.aclweb.org/anthology/)
  - [Lexical and Computational Semantics and Semantic Evaluation (formerly Workshop on Sense Evaluation) (\*SEMEVAL)](https://www.aclweb.org/anthology/venues/semeval/)
    - [2013](https://www.aclweb.org/anthology/events/semeval-2013/)

### SemEval 2013

- [SemEval 2013: Program](https://www.cs.york.ac.uk/semeval-2013/accepted.html)

#### Subtask 13

- [**Word Sense Induction for Graded and Non-Graded Senses**](https://www.cs.york.ac.uk/semeval-2013/task13.html)
- [**Task 13 Paper**](https://www.aclweb.org/anthology/S13-2049)
  - [link in ACL Anthology](https://www.aclweb.org/anthology/papers/S/S13/S13-2049/)
  - [SemanticScholer](https://www.semanticscholar.org/paper/SemEval-2013-Task-13%3A-Word-Sense-Induction-for-and-Jurgens-Klapaftis/0d62b1bc53f8c253915d3ba5de50b461b49b7ead)
- [**All data and system submissions**](https://www.cs.york.ac.uk/semeval-2013/task13/data/uploads/semeval-2013-task-13-test-data.zip) - i.e. the "SemEval-2013-Task-13-test-data" folder
- [Errata (Corrigendum)](https://www.cs.mcgill.ca/~jurgens/docs/semeval-2013-task13-errata.pdf)
  - [cluster-comparison-tools](https://code.google.com/archive/p/cluster-comparison-tools/)

Relative Works

- [Paper - Word Sense Induction with Neural biLM and Symmetric Patterns](https://arxiv.org/abs/1808.08518)
  - [asafamr/SymPatternWSI](https://github.com/asafamr/SymPatternWSI)
- [Paper - AutoSense Model for Word Sense Induction](https://arxiv.org/abs/1811.09242)
  - [rktamplayo/AutoSense](https://github.com/rktamplayo/AutoSense)

### WordNet 3.1

- [WordNet](http://wordnet.princeton.edu/)
  - [Download - current version](https://wordnet.princeton.edu/download/current-version)

> Version 3.1 is currently available only online.

- [**WordNet Search - 3.1**](http://wordnetweb.princeton.edu/perl/webwn)

#### NLTK API

- [**WordNet Interface**](http://www.nltk.org/howto/wordnet.html)

```py
# get wordnet (at the first time)
import nltk
nltk.download('wordnet')

# use wordnet api
from nltk.corpus import wordnet as wn
```

```py
# Synset.definition()
wn.synsets('dark') # list of Synset
wn.synsets('dark', pos=wn.NOUN)
wn.synset('dark.n.01') # a Synset

# Lemma.key()
wn.lemma('dark.n.01.dark') # a Lemma (dark in dark.n.01)
```

### Similarity

- [FIVE MOST POPULAR SIMILARITY MEASURES IMPLEMENTATION IN PYTHON](http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/)
- [Scipy - Distance computations (scipy.spatial.distance)](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

## Motivation

> WSI = Word Sense Induction
> It's a task to get the word sense

- local
  - syntactic
  - close to the disambiguate word
- global
  - thematic
  - semantic

## BiLim

### Exercise for cluster method & Language Model method

| LM   | cluster  | Jac.  | POS   | WDCG    | F-NMI   | F_BC  | AVG     |
| ---- | -------- | ----- | ----- | ------- | ------- | ----- | ------- |
| ELMo | one-hot  | 19.58 | 61.45 | 33.25   | 9.28    | 58.70 | 23.34   |
| ELMo | tf-idf   | 20.20 | 62.40 | `34.06` | `11.06` | 57.72 | `25.27` |
| ELMo | Bert     | 19.20 | 60.23 | 30.69   | 2.65    | 54.34 | 12.00   |
| ELMo | Glove    | 21.22 | 63.18 | 32.00   | 8.28    | 60.90 | 22.45   |
| ELMo | FastText | 20.14 | 62.14 | 31.47   | 7.40    | 61.39 | 21.32   |

| LM    | cluster  | Jac.  | POS     | WDCG  | F-NMI | F_BC  | AVG   |
| ----- | -------- | ----- | ------- | ----- | ----- | ----- | ----- |
| Glove | one-hot  | 19.59 | 61.66   | 32.63 | 8.66  | 59.21 | 22.64 |
| Glove | tf-idf   | 19.94 | 62.50   | 33.65 | 10.69 | 57.68 | 24.83 |
| Glove | Glove    | 21.31 | `63.87` | 31.83 | 8.08  | 60.88 | 22.18 |
| Glove | FastText | 19.81 | 61.78   | 30.92 | 6.49  | 61.08 | 19.91 |

| LM       | cluster  | Jac.    | POS   | WDCG  | F-NMI | F_BC    | AVG   |
| -------- | -------- | ------- | ----- | ----- | ----- | ------- | ----- |
| FastText | one-hot  | 19.66   | 61.79 | 32.97 | 8.78  | 58.59   | 22.68 |
| FastText | tf-idf   | 20.08   | 62.90 | 33.80 | 10.82 | 57.34   | 24.90 |
| FastText | Glove    | `21.80` | 63.49 | 31.77 | 7.79  | 60.52   | 21.72 |
| FastText | FastText | 20.73   | 62.15 | 31.37 | 7.08  | `61.11` | 20.80 |

### Exercise for cluster num

LM=ELMo, cluster=tf-idf

| LM   | clusterNum | Jac.  | POS   | WDCG    | F-NMI   | F_BC  | AVG     |
| ---- | ---------- | ----- | ----- | ------- | ------- | ----- | ------- |
| ELMo | 5          | 19.68 | 61.67 | 32.29   | 9.68    | 58.88 | 23.87   |
| ELMo | 6          | 19.93 | 61.90 | 33.28   | 10.21   | 58.25 | 24.39   |
| ELMo | 7          | 20.20 | 62.40 | `34.06` | `11.06` | 57.72 | `25.27` |
| ELMo | 8          | 20.38 | 61.72 | 34.46   | 10.88   | 56.60 | 24.82   |
| ELMo | 9          | 20.30 | 62.34 | 35.25   | 11.58   | 56.61 | 25.60   |
| ELMo | 10         | 20.39 | 62.67 | 35.23   | 10.95   | 55.94 | 24.75   |

## Trouble Shooting

1. AttributeError: '\_NamespacePath' object has no attribute 'sort'

It's difficult to solve in Intel® AI DevCloud

```bash
pip install --upgrade pip

python3 -m pip install --user --upgrade pip==9.0.3

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py --user
```
