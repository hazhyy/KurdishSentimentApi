from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from gensim.models import Word2Vec
from wordvector import averageVector
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class TextModel(BaseModel):
    text: str
    model: str


stopwords_path = "./wordvector/stop_words.txt"
word2vec_model = Word2Vec.load("./wordvector/kurdish-word2vec-75279.model")
rf_sentiment = load("./models/LogisticModel40000.joblib")
lr_loaded = load("./models/LogisticModel40000.joblib")
lr_score = load("./models/logisticModelSentiment.joblib")


class Text(BaseModel):
    text: str


sentiments = {
    1: {"text": "بەتاڵ", "english": "empty"},
    2: {"text": "دڵتەنگ", "english": "sadness"},
    3: {"text": "پەرۆش", "english": "enthusiasm"},
    4: {"text": "ئاسایی", "english": "neutral"},
    5: {"text": "نیگەران", "english": "worry"},
    6: {"text": "سوپرایز", "english": "surprise"},
    7: {"text": "خۆشەویستی", "english": "love"},
    8: {"text": "خۆشی", "english": "fun"},
    9: {"text": "ڕق", "english": "hate"},
    10: {"text": "بەختەوەر", "english": "happiness"},
    11: {"text": "بێزار", "english": "boredom"},
    12: {"text": "ئارام", "english": "relief"},
    13: {"text": "تووڕە", "english": "anger"},
}


def get_word_predictions(text: str, model: str):
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()
    words = text.split()  # Split the text into words
    word_vectors = []
    print(model)
    for word in words:
        if word not in stopwords:  # Check if the word is not in stopwords
            vector = averageVector.get_average_word_vector(
                word, word2vec_model, 100, stopwords_path=stopwords_path
            )
            if model == "lr":
                prediction = lr_loaded.predict([vector])
            elif model == "rf":
                prediction = rf_sentiment.predict([vector])
            word_vectors.append(
                {
                    "word": word,
                    "prediction_class": int(prediction[0]),
                    "sentiment": sentiments[int(prediction[0])]["text"],
                }
            )
    return word_vectors


@app.post("/predict/")
async def predict_sentiment(text_model: TextModel):
    try:
        word_vectors = get_word_predictions(text_model.text, text_model.model)

        new_vector = averageVector.get_average_word_vector(
            text_model.text, word2vec_model, 100, stopwords_path=stopwords_path
        )

        if text_model.model == "lr":
            prediction = lr_loaded.predict([new_vector])
        elif text_model.model == "rf":
            prediction = rf_sentiment.predict([new_vector])
        return {
            "labeled_prediction_class": int(prediction[0]),
            "labeled_sentiment": sentiments[prediction[0]]["text"],
            "labeled_prediction_by_word": word_vectors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/")
async def predict_score(text: Text):
    try:

        new_vector = averageVector.get_average_word_vector(
            text.text, word2vec_model, 100, stopwords_path=stopwords_path
        )

        prediction = lr_score.predict([new_vector])
        return {
            "score": int(prediction[0]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
