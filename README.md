
# ML Inference API with FastAPI

## Project Description

This project provides a REST API for classifying SMS messages as spam or ham using a trained machine learning model.

The API is built with FastAPI and uses a saved scikit-learn pipeline for predictions.

## Model Used

- NLP text classification
- TF-IDF vectorization
- Multinomial Naive Bayes

## Prediction Task

The model predicts whether an SMS message is spam or ham.

## Input Features

The API accepts one input feature:

- `message`: The SMS message to classify, supplied as a string.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns the API name and status message |
| GET | `/health` | Returns the API health status |
| POST | `/predict` | Predicts whether a message is spam or ham |

## Project Structure

```text
ml-inference-api/
├── main.py
├── save_model.py
├── test_api.py
├── model/
│   └── pipeline.pkl
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Python virtual environment.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train and Save the Model

Run:

```bash
python save_model.py
```

The script downloads the public dataset, trains the model, and saves the pipeline to `model/pipeline.pkl`.

## Run the API

Start the server:

```bash
fastapi dev main.py
```

The API runs locally at:

http://localhost:8000

## Swagger UI

Open the interactive API documentation:

http://localhost:8000/docs

## Example Prediction Request

POST `/predict`

```json
{
  "message": "Congratulations! You have won a free prize!"
}
```

## Example Prediction Response

The following is an illustrative response. Actual predictions and confidence values depend on the trained model.

```json
{
  "prediction": "spam",
  "confidence": 0.9876
}
```

## Test the API

Keep the FastAPI server running and open a second terminal.

Run:

```bash
python test_api.py
```

The script tests the health endpoint, a spam message, a ham message, and invalid input.

## Swagger UI Screenshot

<img width="1920" height="968" alt="Screenshot 2026-10-10 at 3 50 37 PM" src="https://github.com/user-attachments/assets/3467432c-90aa-47eb-a3a6-6baba3948e91" />
<img width="1261" height="784" alt="Screenshot 2026-10-10 at 3 54 09 PM" src="https://github.com/user-attachments/assets/414c2611-f857-4cea-b736-81939fc2c4f6" />



NB. All this is running Locally not hosted
