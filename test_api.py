
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = "http://localhost:8000"


def send_request(url, method="GET", data=None):
    headers = {}

    body = None

    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        url,
        data=body,
        headers=headers,
        method=method
    )

    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as error:
        return error.code, json.loads(
            error.read().decode("utf-8")
        )


# Test 1: Health check
print("=" * 50)
print("Test 1: Health Check")

status, result = send_request(f"{BASE_URL}/health")

print(f"Status  : {status}")
print(f"Response: {result}")


# Test 2: Spam message
print()
print("=" * 50)
print("Test 2: Spam Message")

data = {
    "message": (
        "Congratulations! You have won a free iPhone. "
        "Click here to claim now!"
    )
}

status, result = send_request(
    f"{BASE_URL}/predict",
    method="POST",
    data=data
)

print(f"Message : {data['message']}")
print(f"Status  : {status}")
print(f"Response: {result}")


# Test 3: Ham message
print()
print("=" * 50)
print("Test 3: Ham Message")

data = {
    "message": "Hey, are we still meeting for lunch tomorrow?"
}

status, result = send_request(
    f"{BASE_URL}/predict",
    method="POST",
    data=data
)

print(f"Message : {data['message']}")
print(f"Status  : {status}")
print(f"Response: {result}")


# Test 4: Invalid input (wrong field name)
print()
print("=" * 50)
print("Test 4: Invalid Input (wrong field name)")

data = {
    "text": "This is wrong field name"
}

status, result = send_request(
    f"{BASE_URL}/predict",
    method="POST",
    data=data
)

print(f"Status  : {status}")
print(f"Response: {result}")
