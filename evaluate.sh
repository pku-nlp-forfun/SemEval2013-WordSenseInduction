#!/bin/bash

if [ -z "$1" ]
then
  echo "Usage: $0 [key_file]"
  exit 1
fi

echo "==================== Evaluation of $1 ===================="
echo

echo "========== WSD =========="
echo

echo "Jaccard Index"
java -jar SemEval-2013-Task-13-test-data/scoring/jaccard-index.jar  SemEval-2013-Task-13-test-data/keys/gold/all.key $1
echo

echo "Positional-Weighted Kendall's tau"
java -jar SemEval-2013-Task-13-test-data/scoring/positional-tau.jar  SemEval-2013-Task-13-test-data/keys/gold/all.key $1
echo

echo "Weighted NDCG"
java -jar SemEval-2013-Task-13-test-data/scoring/weighted-ndcg.jar  SemEval-2013-Task-13-test-data/keys/gold/all.key $1
echo

echo "========== Sense Cluster Comparisons =========="
echo

echo "Fuzzy NMI"
java -jar SemEval-2013-Task-13-test-data/scoring/fuzzy-nmi.jar  SemEval-2013-Task-13-test-data/keys/gold/all.key $1
echo

echo "Fuzzy B-Cube"
java -jar SemEval-2013-Task-13-test-data/scoring/fuzzy-bcubed.jar  SemEval-2013-Task-13-test-data/keys/gold/all.key $1
echo
