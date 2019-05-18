from corpus import loadSenseval2Format
from nltk.corpus import wordnet as wn
import numpy as np

import io  # load fastText embedding
import pickle as pkl
import os  # isfile

from tqdm import tqdm


def loadPretrainedFastText(filename: str = "Embedding/wiki-news-300d-1M"):
    if not os.path.isfile(filename + ".pkl"):
        print("Loading embedding and also dump to pickle format...")
        fin = io.open(filename+".vec", 'r', encoding='utf-8',
                      newline='\n', errors='ignore')
        n_words, dimension = map(int, fin.readline().split())
        print("loading words:", n_words, "with dimension:", dimension)
        embedding = {}
        for line in tqdm(fin, total=n_words):
            tokens = line.rstrip().split(' ')
            embedding[tokens[0]] = map(float, tokens[1:])

        with open(filename + ".pkl", 'w') as pklFile:
            pkl.dump(embedding, pklFile)
    else:
        print("Loading embedding in pickle format...")
        with open(filename + ".pkl", 'r') as pklFile:
            embedding = pkl.load(pklFile)

    return embedding


def sentenceEmbedding(sentence: str, embedding: dict):
    """
    The sentence Embedding is formed by concatting the words
    in the sentence to the maximum length.
    And then use max-pooling like TextCNN to reduce to fixed dimension representation.
    """
    returnEmbedding = []
    return returnEmbedding


def wordNetMeaningEmbeddings(word: str, pos: str, embedding: dict) -> dict:
    """
    Get all the meanings in embedding format of a word in wordNet

    return dict: {lemma_key: embedding of its meaning}
    """
    meaningEmbedding = {}
    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            if lemma.name() == word:
                meaningEmbedding[lemma.key()] = sentenceEmbedding(
                    synset.definition(), embedding)
                break

    return meaningEmbedding


def main():
    embedding = loadPretrainedFastText()
    # test the embedding
    print(embedding['dark'])

    # test the wordnet to embedding
    Dataset = loadSenseval2Format()
    wordNetMeaningEmbeddings(
        Dataset["become.v"].lemma, Dataset["become.v"].pos, embedding)


if __name__ == "__main__":
    main()
