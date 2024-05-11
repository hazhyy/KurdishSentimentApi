from gensim.models import Word2Vec
import numpy as np


def get_average_word_vector(
    sentence,
    model,
    num_features,
    stopwords_path="./stop_words.txt",
):
    """Calculate the average word vector for a sentence, excluding stop words."""
    # Load stop words from file
    try:
        with open(stopwords_path, "r", encoding="utf-8") as file:
            stopwords = set(file.read().split())
    except Exception as e:
        print(f"Failed to open file: {e}")
    words = sentence.split()
    feature_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0
    for word in words:
        if (
            word not in stopwords and word in model.wv.key_to_index
        ):  # Check if word is not a stop word
            print(word)
            num_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    return feature_vector / num_words if num_words > 0 else feature_vector
