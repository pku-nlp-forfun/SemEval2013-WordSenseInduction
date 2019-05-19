from enum import Enum
from scipy.spatial import distance

Similarity = {
    "Cosine": lambda u, v: 1/distance.cosine(u, v),
    "Euclidean": lambda u, v: 1/distance.euclidean(u, v),
    "Minkowski": lambda u, v: 1/distance.minkowski(u, v)
}


class SentenceEmbedding(Enum):
    NaiveAdding = 0
    BackPadding = 1
    FrontPadding = 2


SentenceEmbeddingString = {
    SentenceEmbedding.NaiveAdding: "NaiveAdding",
    SentenceEmbedding.BackPadding: "BackPadding",
    SentenceEmbedding.FrontPadding: "FrontPadding"
}
