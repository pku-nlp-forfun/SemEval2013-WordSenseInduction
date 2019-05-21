from corpus import loadSenseval2Format
from nltk.corpus import wordnet as wn
from bert_embedding import BertEmbedding

import numpy as np

import io  # load fastText embedding
import pickle as pkl
import os  # isfile

from string import punctuation  # remove punctuation from string

from tqdm import tqdm
import torch

from configure import SentenceEmbedding, EmbeddingMethod
from pytorch_pretrained_bert import BertModel, BertTokenizer


EMBEDDING = EmbeddingMethod.BERT_TORCH  # FastText, BERT
EMBEDDING_FILE = "Embedding/wiki-news-300d-1M"  # faxtText official
bert_dir = '/Users/gunjianpan/Desktop/git/bert'
bert = BertModel.from_pretrained(bert_dir)
tokenizer = BertTokenizer.from_pretrained(f'{bert_dir}/uncased_L-12_H-768_A-12/vocab.txt')

def getAllWord():
    """
    Get word from the test set and definition of WordNet.
    Used to reduce the pickle size of the pretrained fastText
    """
    allWord = set()
    Dataset = loadSenseval2Format()
    for lemma in Dataset.values():
        for instence in lemma.instances:
            for word in instence['context'].split():
                allWord.add(word.strip(punctuation))

        for synset in wn.synsets(lemma.lemma, pos=lemma.pos):
            for word in synset.definition().split():
                allWord.add(word.strip(punctuation))

    return allWord


def loadPretrainedFastText(filename: str = EMBEDDING_FILE):
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

        # calculate average embedding
        embedding['AVG'] = np.mean(list(embedding.values()))

        with open(filename + ".pkl", 'wb') as pklFile:
            pkl.dump(embedding, pklFile)
    else:
        print("Loading embedding in pickle format...")
        with open(filename + ".pkl", 'rb') as pklFile:
            embedding = pkl.load(pklFile)

    return embedding


def getSentenceEmbedding(sentence: str, embedding: dict, maxSentenceLen: int, method: int = SentenceEmbedding.NaiveAdding):
    """
    The sentence Embedding is formed by concatting the words
    in the sentence to the maximum length.
    And then use max-pooling like TextCNN to reduce to fixed dimension representation.
    """

    if EMBEDDING == EmbeddingMethod.FastText:
        returnEmbedding = np.zeros((300, ))
        if method == SentenceEmbedding.NaiveAdding:
            for word in sentence.split():
                try:
                    returnEmbedding += embedding[word.strip(punctuation)]
                except KeyError:  # getting word which is not in embedding table
                    continue
        elif method == SentenceEmbedding.NaiveNormalized:
            sentenceLen = 0
            for word in sentence.split():
                try:
                    returnEmbedding += embedding[word.strip(punctuation)]
                    sentenceLen += 1
                except KeyError:  # getting word which is not in embedding table
                    continue
            returnEmbedding /= sentenceLen
        elif method == SentenceEmbedding.NaiveAvgPadding:
            sentenceLen = 0
            for word in sentence.split():
                try:
                    returnEmbedding += embedding[word.strip(punctuation)]
                except KeyError:  # getting word which is not in embedding table
                    returnEmbedding += embedding['AVG']
                sentenceLen += 1
            for _ in range(maxSentenceLen - sentenceLen):
                # padding the sentence to maxSentenceLen
                returnEmbedding += embedding['AVG']
            returnEmbedding /= maxSentenceLen

    elif EMBEDDING == EmbeddingMethod.BERT:
        bert_embedding = BertEmbedding()  # dimension should be (768,)
        returnEmbedding = np.zeros((768, ))
        if method == SentenceEmbedding.NaiveAdding:
            embeddingList = bert_embedding(sentence.split())
            for _, arrayList in embeddingList:
                returnEmbedding += arrayList[0]
    elif EMBEDDING == EmbeddingMethod.BERT_TORCH:
        ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))])
        returnEmbedding = bert(ids, output_all_encoded_layers=False)[-1][0].detach().numpy()

    return returnEmbedding


def wordNetMeaningEmbeddings(word: str, pos: str, embedding: dict, maxSentenceLen: int) -> dict:
    """
    Get all the meanings in embedding format of a word in wordNet

    return dict: {lemma_key: embedding of its meaning}
    """
    meaningEmbedding = {}
    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            if lemma.name() == word:
                meaningEmbedding[lemma.key()] = getSentenceEmbedding(
                    synset.definition(), embedding, maxSentenceLen)
                break

    return meaningEmbedding


def main():
    embedding = loadPretrainedFastText()
    # test the embedding
    # print(embedding['become'])

    Dataset = loadSenseval2Format()
    # test the sentence to embedding
    sentence = list(Dataset.values())[
        0].instances[0]['context']
    print("sentence embedding:", sentence,
          getSentenceEmbedding(sentence, embedding, list(Dataset.values())[
              0].max_sentence_len))

    # test the wordnet to embedding
    meaningEmbedding = wordNetMeaningEmbeddings(
        Dataset["become.v"].lemma, Dataset["become.v"].pos, embedding, Dataset["become.v"].max_sentence_len)
    # test the definition string embedding
    print("wordnet meaning embedding:",  list(meaningEmbedding.keys())
          [0], list(meaningEmbedding.values())[0])


if __name__ == "__main__":
    main()
