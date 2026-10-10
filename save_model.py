
import csv
import io
import string
import zipfile
import urllib.request
from pathlib import Path

import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)

# Find the project directory and model output location.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "pipeline.pkl"


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def load_dataset():

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "ML-Inference-API-Project"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        archive_data = response.read()


    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        with archive.open("SMSSpamCollection") as dataset_file:
            dataset_text = io.TextIOWrapper(
                dataset_file,
                encoding="utf-8"
            )

            reader = csv.reader(dataset_text, delimiter="\t")

            messages = []
            labels = []

            for row in reader:
                if len(row) != 2:
                    continue

                label, message = row

                if label not in ("ham", "spam"):
                    continue

                messages.append(preprocess(message))
                labels.append(label)

    return messages, labels


def train_and_save_model():
    messages, labels = load_dataset()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])

    pipeline.fit(messages, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("Model training completed successfully.")
    print(f"Training messages: {len(messages)}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
