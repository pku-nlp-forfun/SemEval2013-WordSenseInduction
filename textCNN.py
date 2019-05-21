from keras.layers import Input, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from typing import List
import numpy as np


class TextCNN:
    def __init__(self, max_sentence_len: int, embedding_dim: int, vocabulary_size: int, filter_sizes: List[int] = [3, 4, 5], num_filters: int = 512):
        """
        The layes of model is determined by the length of the filter_sizes list.
        """
        input_layer = Input(shape=(max_sentence_len, embedding_dim))
        reshape = Reshape((max_sentence_len, embedding_dim, 1))(input_layer)

        max_pool_layers = []
        for filter_size in filter_sizes:
            conv_layer = Conv2D(num_filters, kernel_size=(filter_size, embedding_dim),
                                padding='valid', kernel_initializer='normal', activation='relu')(reshape)
            max_pool_layers.append(MaxPool2D(pool_size=(
                max_sentence_len - filter_size + 1, 1), strides=(1, 1), padding='valid')(conv_layer))

        concatenated_tensor = Concatenate(axis=1)(max_pool_layers)

        output_layer = Flatten()(concatenated_tensor)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        # optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,
        #                  epsilon=1e-08, decay=0.0)
        # self.model.compile(optimizer)

    def getSentenceEmbedding(self, padded_sentence_matrix):
        output = self.model.predict(padded_sentence_matrix, batch_size=1)
        return np.reshape(output, (np.size(output), ))


def main():
    textCNN_model = TextCNN(30, 300, 5000)
    plot_model(textCNN_model.model, "images/TextCNN_model.png", show_shapes=True, show_layer_names=False)


if __name__ == "__main__":
    main()
