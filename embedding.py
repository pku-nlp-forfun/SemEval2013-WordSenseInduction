from corpus import loadSenseval2Format
from nltk.corpus import wordnet as wn

import numpy as np

import io  # load fastText embedding
import pickle as pkl
import os  # isfile

from string import punctuation  # remove punctuation from string

from tqdm import tqdm


EMBEDDING = "Embedding/wiki-news-300d-1M"  # faxtText official


def getAllWord():
    """
    Get word from the test set and definition of WordNet.
    Used to reduce the pickle size of the pretrained fastText
    """
    allWord = set()
    Dataset = loadSenseval2Format()
    for lemma in Dataset.values():
        for instence in lemma.instances.values():
            for word in instence['context'].split():
                allWord.add(word.strip(punctuation))

        for synset in wn.synsets(lemma.lemma, pos=lemma.pos):
            for word in synset.definition().split():
                allWord.add(word.strip(punctuation))

    return allWord


def loadPretrainedFastText(filename: str = EMBEDDING):
    if not os.path.isfile(filename + ".pkl"):
        print("Loading embedding and also dump to pickle format...")
        fin = io.open(filename+".vec", 'r', encoding='utf-8',
                      newline='\n', errors='ignore')
        n_words, dimension = map(int, fin.readline().split())
        print("loading words:", n_words, "with dimension:", dimension)
        embedding = {}

        words = getAllWord()
        for line in tqdm(fin, total=n_words):
            tokens = line.rstrip().split(' ')
            if tokens[0] not in words:
                continue
            embedding[tokens[0]] = np.fromiter(
                map(float, tokens[1:]), np.float)

        with open(filename + ".pkl", 'wb') as pklFile:
            pkl.dump(embedding, pklFile)
    else:
        print("Loading embedding in pickle format...")
        with open(filename + ".pkl", 'rb') as pklFile:
            embedding = pkl.load(pklFile)

    return embedding


def getSentenceEmbedding(sentence: str, embedding: dict):
    """
    The sentence Embedding is formed by concatting the words
    in the sentence to the maximum length.
    And then use max-pooling like TextCNN to reduce to fixed dimension representation.
    """
    returnEmbedding = np.zeros((300, ))
    for word in sentence.split():
        try:
            returnEmbedding += embedding[word.strip(punctuation)]
        except KeyError:  # getting word which is not in embedding table
            continue

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
                meaningEmbedding[lemma.key()] = getSentenceEmbedding(
                    synset.definition(), embedding)
                break

    return meaningEmbedding


def main():
    embedding = loadPretrainedFastText()
    # test the embedding
    # print(embedding['become'])

    Dataset = loadSenseval2Format()
    # test the sentence to embedding
    sentence = list(list(Dataset.values())[
        0].instances.values())[0]['context']
    print("sentence embedding:", sentence,
          getSentenceEmbedding(sentence, embedding))

    # test the wordnet to embedding
    meaningEmbedding = wordNetMeaningEmbeddings(
        Dataset["become.v"].lemma, Dataset["become.v"].pos, embedding)
    # test the definition string embedding
    print("wordnet meaning embedding:",  list(meaningEmbedding.keys())
          [0], list(meaningEmbedding.values())[0])


if __name__ == "__main__":
    main()
