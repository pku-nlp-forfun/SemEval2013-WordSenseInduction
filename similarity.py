from embedding import loadPretrainedFastText, getSentenceEmbedding, wordNetMeaningEmbeddings, EMBEDDING
from corpus import loadSenseval2Format
import numpy as np
from operator import itemgetter
from collections import defaultdict
from configure import Similarity, SentenceEmbedding, SentenceEmbeddingString, EmbeddingMethod, EmbeddingMethodString
import os
from evaluation import evaluateFromFile

######## Setting ########

SIMILARITY = "Cosine"  # Cosine, Euclidean, Minkowski

# if the result need to be just N (True) or can be less than N (False)
STRICT_N = True
LEASTN = 1  # at least N result (do not less than 1)
THRESHOLD = 0.1  # If STRICT_N is False, then it will get rid off the weight which is less than THRESHOLD

# NaiveAdding, NaiveNormalized, NaiveAvgPadding, TextCNN
SENTENCE_EMBEDDING = SentenceEmbedding.TextCNN

######## Setting ########

TOPN = [1, 2, 3, 4, 5]  # the N value to test

RESULT_PATH = "Result" + SIMILARITY + EmbeddingMethodString[EMBEDDING] + \
    SentenceEmbeddingString[SENTENCE_EMBEDDING]


def sentenceSimilarity(sentenceEmbedding: np.array, definitionEmbedding: np.array):
    """
    Calculate the similarity between two sentences

    (may test different similarity method)
    """
    return Similarity[SIMILARITY](sentenceEmbedding, definitionEmbedding)


def sentenceVsDefinitions(sentence: str, definitions: dict, embedding: dict, maxSentenceLen: int):
    """
    The similarity between the sentence and all the definitions
    """
    similarityDict = {}
    for lemma_key, definitionEmbedding in definitions.items():
        similarityDict[lemma_key] = sentenceSimilarity(
            getSentenceEmbedding(sentence, embedding, maxSentenceLen, method=SENTENCE_EMBEDDING), definitionEmbedding)
    return similarityDict


def topNSimilarity(lemma, N: int, embedding: dict):
    """
    For each Lemma, calculate the similarity with all the definition of the Lemma itself.
    And select the top N weight as result.
    (the top N result has all minus the weight of the top N+1th result)
    """
    definitions = wordNetMeaningEmbeddings(
        lemma.lemma, lemma.pos, embedding, lemma.max_sentence_len, method=SENTENCE_EMBEDDING)
    result = defaultdict(list)
    for instance in lemma.instances:
        similarityDict = sentenceVsDefinitions(
            instance['context'], definitions, embedding, lemma.max_sentence_len)
        candidates = sorted(similarityDict.items(),
                            key=itemgetter(1), reverse=True)[:N+1]
        # get the wiehgt of the N+1th definition
        thresholdWeight = candidates[-1][1]
        for lemma_key, weight in candidates[:N]:
            if not STRICT_N and len(result[instance["id"]]) >= LEASTN and weight - thresholdWeight < THRESHOLD:
                # skip the weight that is too less when disable STRICT_N (but at least with 1 result)
                break
            result[instance["id"]].append(
                (lemma_key, weight - thresholdWeight))

    return result


def topNcorpusVsWordnet(Dataset, embedding, N: int):
    """
    Calculating top N similarity and output result
    """
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

    print("Current Setting:")
    print("Similarity:", SIMILARITY)
    if STRICT_N:
        print("Strict TopN mode")
    else:
        print("Generalized TopN mode with THRESHOLD:", THRESHOLD)
    print("Embedding:", EmbeddingMethodString[EMBEDDING])
    print("Sentence Embedding:", SentenceEmbeddingString[SENTENCE_EMBEDDING])

    Dataset = loadSenseval2Format()
    embedding = loadPretrainedFastText()
    for i in TOPN:
        topNcorpusVsWordnet(Dataset, embedding, i)
        filename = "top%d.key" % i
        evaluateFromFile(os.path.join(RESULT_PATH, filename),
                         (os.path.join(RESULT_PATH, "evaluate.log")))


if __name__ == "__main__":
    main()
