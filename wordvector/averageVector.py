from gensim.models import Word2Vec
import numpy as np


def get_average_word_vector(sentence, model, num_features):
    """Calculate the average word vector for a sentence."""
    words = sentence.split()
    feature_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0
    for word in words:
        if word in model.wv.key_to_index:
            num_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    return feature_vector / num_words if num_words > 0 else feature_vector
