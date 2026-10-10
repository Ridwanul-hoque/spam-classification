
import string
from pathlib import Path

import joblib
import nltk
from fastapi import FastAPI
from pydantic import BaseModel, Field
from nltk.corpus import stopwords


nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)

app = FastAPI(title="Spam Classifier API")


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


class MessageInput(BaseModel):
    message: str = Field(min_length=1)


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Spam Classifier API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(news_input: MessageInput):
    clean = preprocess(news_input.message)

    pred = pipeline.predict([clean])[0]
    proba = pipeline.predict_proba([clean])[0]
    conf = round(float(max(proba)), 4)

    return PredictionOutput(
        prediction=str(pred),
        confidence=conf
    )
