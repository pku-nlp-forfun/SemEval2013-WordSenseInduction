from enum import Enum
from scipy.spatial import distance

Similarity = {
    "Cosine": lambda u, v: 1/distance.cosine(u, v),
    "Euclidean": lambda u, v: 1/distance.euclidean(u, v),
    "Minkowski": lambda u, v: 1/distance.minkowski(u, v)
}


class SentenceEmbedding(Enum):
    NaiveAdding = 0  # just add all the embedding in the sentence
    NaiveNormalized = 1  # divide the NaiveAdding result by word number
    NaiveAvgPadding = 2  # padding the sentence to same length using average embedding
    TextCNN = 3  # output max pooling result as the output embedding


SentenceEmbeddingString = {
    SentenceEmbedding.NaiveAdding: "NaiveAdding",
    SentenceEmbedding.NaiveNormalized: "NaiveNormalized",
    SentenceEmbedding.NaiveAvgPadding: "NaiveAvgPadding",
    SentenceEmbedding.TextCNN: "TextCNN"
}


class EmbeddingMethod(Enum):
    FastText = 0
    BERT = 1
    BERT_TORCH = 2


EmbeddingMethodString = {
    EmbeddingMethod.FastText: "FastText",
    EmbeddingMethod.BERT: "BERT",
    EmbeddingMethod.BERT_TORCH: "BERT_TORCH"
}
