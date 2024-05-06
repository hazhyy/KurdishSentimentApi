from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from gensim.models import Word2Vec
from wordvector import averageVector
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

word2vec_model = Word2Vec.load("./wordvector/kurdish-word2vec-75279.model")
lr_loaded = load("./models/LogisticModel9000.joblib")


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


@app.post("/predict/")
async def predict_sentiment(text: Text):
    try:
        new_vector = averageVector.get_average_word_vector(
            text.text, word2vec_model, 100
        )
        prediction = lr_loaded.predict([new_vector])
        return {
            "prediction": int(prediction[0]),
            "text": sentiments[prediction[0]]["text"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
