from embedding import loadPretrainedFastText, getSentenceEmbedding, wordNetMeaningEmbeddings
from corpus import loadSenseval2Format
import numpy as np
from scipy.spatial import distance
from operator import itemgetter

import os

RESULT_PATH = "Result"


def sentenceSimilarity(sentenceEmbedding: np.array, definitionEmbedding: np.array):
    # may test different similarity method
    return 1/distance.cosine(sentenceEmbedding, definitionEmbedding)


def sentenceVsDefinitions(sentence: str, definitions: dict, embedding: dict):
    similarityDict = {}
    for lemma_key, definitionEmbedding in definitions.items():
        similarityDict[lemma_key] = sentenceSimilarity(
            getSentenceEmbedding(sentence, embedding), definitionEmbedding)
    return similarityDict


def topNSimilarity(lemma, N: int, embedding: dict):
    definitions = wordNetMeaningEmbeddings(lemma.lemma, lemma.pos, embedding)
    result = {}
    for instance in lemma.instances:
        similarityDict = sentenceVsDefinitions(
            instance['context'], definitions, embedding)
        result[instance["id"]] = sorted(
            similarityDict.items(), key=itemgetter(1), reverse=True)[:N]

    return result


def topNcorpusVsWordnet(Dataset, embedding, N: int):
    checkFileAndRemove(N)  # remove previous result

    print("Dealing with top %d result..." % N)
    for case, lemma in Dataset.items():
        print("Processing", case)
        top2Similarity = topNSimilarity(lemma, N, embedding)
        outputTopNResult(case, top2Similarity, N)


def outputTopNResult(case: str, topNResult: dict, N: int):
    filename = "top%d.key" % N
    with open(os.path.join(RESULT_PATH, filename), "a") as result:
        for num, similarityTupleList in topNResult.items():
            result.write(case + ' ' + case + '.' + num)
            for lemma_key, weight in similarityTupleList:
                result.write(' ' + lemma_key + '/' + str(weight))
            result.write("\n")


def checkFileAndRemove(N: int):
    filename = "top%d.key" % N
    if os.path.isfile(os.path.join(RESULT_PATH, filename)):
        os.remove(os.path.join(RESULT_PATH, filename))


def main():
    os.makedirs(RESULT_PATH, exist_ok=True)

    Dataset = loadSenseval2Format()
    embedding = loadPretrainedFastText()
    for i in [2, 3, 4, 5]:
        topNcorpusVsWordnet(Dataset, embedding, i)


if __name__ == "__main__":
    main()
