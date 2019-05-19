# SemEval-2013 Task 13 Word Sense Induction for Graded and Non-Graded Senses

A PKU course project based on the "SemEval-2013 task 13 Word Sense Induction for Graded and Non-Graded Senses" competition.

Three Subtask:

1. Non-graded Word Sense Iduction Subtask
2. Graded Word Sense Induction Subtask
3. Lexical Substitution SubTask

> The link of description for each subtask is broken.

## Getting Started

```sh
# get fastText pre-trained embedding
bash fastTextPretrainEmbedding.sh

# get the topN key file
python3 similarity.py

# evaluate
./evaluate.sh Result/top2.key
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

### The ELMo Approach

> Follow the thought of the paper: Word Sense Induction with Neural biLM and Symmetric Patterns

## Evaluation Result

### Strict TopN Approach Result

> Output of TopN must be N results.

| model | Embedding | Similarity | Fuzzy NMI | Fuzzy B-Cube | AVG   |
| ----- | --------- | ---------- | --------- | ------------ | ----- |
| Top1  | fastText  | Cosine     | 2.81      | 50.22        | 11.89 |
| Top2  | fastText  | Cosine     | 8.56      | 52.10        | 21.12 |
| Top3  | fastText  | Cosine     | 7.15      | 42.87        | 17.51 |
| Top4  | fastText  | Cosine     | 6.61      | 32.07        | 14.56 |
| Top5  | fastText  | Cosine     | 5.92      | 24.80        | 12.11 |

### Generalized TopN Approach Result

> Output of TopN must be less than N results.

| model | Embedding | Similarity | Fuzzy NMI | Fuzzy B-Cube | AVG |
| ----- | --------- | ---------- | --------- | ------------ | --- |
| Top1  | fastText  | Cosine     |           |              |     |
| Top2  | fastText  | Cosine     |           |              |     |
| Top3  | fastText  | Cosine     |           |              |     |
| Top4  | fastText  | Cosine     |           |              |     |
| Top5  | fastText  | Cosine     |           |              |     |

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

## Motivation

> WSI = Word Sense Induction
> It's a task to get the word sense

- local
  - syntactic
  - close to the disambiguate word
- global
  - thematic
  - semantic

## Exercise

| model | Embedding | F-NMI | F_BC  | AVG   |
| ----- | --------- | ----- | ----- | ----- |
| BiLM  | ELMo      | 11.06 | 57.72 | 25.27 |
