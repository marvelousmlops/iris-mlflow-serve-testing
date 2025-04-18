"""Tests for mlflow serve."""

import json
import pathlib

import pandas as pd
import requests
from loguru import logger

BASE_URL = "http://127.0.0.1:5088"

test_data = {
    "SepalLengthCm": 6.1,
    "SepalWidthCm": 2.8,
    "PetalLengthCm": 4.7,
    "PetalWidthCm": 1.2,
}

pandas_df = pd.DataFrame([test_data])

payload_dataframe_split = json.dumps(
    {"dataframe_split": pandas_df.to_dict(orient="split")}
)
payload_dataframe_records = json.dumps(
    {"dataframe_records": pandas_df.to_dict(orient="records")}
)


def test_inference_server_health() -> None:
    """Test health endpoint of inference server. Verifies the service returns 200 status code."""
    response = requests.get(f"{BASE_URL}/health")
    logger.info(f"Received {response.status_code}.")
    assert response.status_code == 200


def test_inference_server_ping() -> None:
    """Test ping endpoint functionality. Checks if server responds with successful status code."""
    response = requests.get(f"{BASE_URL}/ping")
    logger.info(f"Received {response.status_code}.")
    assert response.status_code == 200


def test_inference_server_version() -> None:
    """Test version endpoint response. Validates status code and matches expected version string."""
    response = requests.get(f"{BASE_URL}/version")
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    assert response.text == "2.17.0"


def test_inference_server_invocations_with_dataframe_split() -> None:
    """Test model invocations using split dataframe format. Verifies successful response and valid prediction format."""
    response = requests.post(
        f"{BASE_URL}/invocations",
        data=payload_dataframe_split,
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    logger.info(f"Received {response.json()}")
    value = response.json()["predictions"]
    assert isinstance(value, list)


def test_inference_server_invocations_with_dataframe_records() -> None:
    """Test model invocations using records format. Validates response status and prediction values format."""
    response = requests.post(
        f"{BASE_URL}/invocations",
        data=payload_dataframe_records,
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200

    predictions = response.json()["predictions"]
    assert all(isinstance(pred, str) for pred in predictions)  # Ensure all are strings
    assert all(
        pred in ["setosa", "versicolor", "virginica"] for pred in predictions
    )  # Validate class names


def test_inference_server_invocations_with_dataframe_records_should_fail_when_contact_request_violation() -> (
    None
):
    """Test that inference server invocations with incomplete DataFrame records fail as expected.

    Drops each column from the DataFrame in turn and verifies that the server returns a 400 error.
    """
    for col in pandas_df.columns.to_list():
        tmp_df = pandas_df.drop(columns=[col])

        tmp_payload_dataframe_records = json.dumps(
            {"dataframe_records": tmp_df.to_dict(orient="records")}
        )
        logger.info(f"Testing with {col} dropped.")
        response = requests.post(
            f"{BASE_URL}/invocations",
            data=tmp_payload_dataframe_records,
            headers={"Content-Type": "application/json"},
            timeout=2,
        )
        logger.info(
            f"Received {response.status_code} with response of '{response.text}'."
        )
        assert response.status_code == 400


def test_infererence_server_invocations_with_full_dataframe() -> None:
    """Test model predictions with complete dataset. Validates response status and prediction class membership."""
    CUR_DIR = pathlib.Path(__file__).parent
    test_set = pd.read_csv(f"{CUR_DIR.as_posix()}/test_data/test_set.csv")
    input_data = test_set.drop(columns=["Id", "Species"])
    input_data = input_data.where(input_data.notna(), None)  # noqa
    input_data = input_data.to_dict(orient="records")
    payload = json.dumps({"dataframe_records": input_data})

    response = requests.post(
        f"{BASE_URL}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    logger.info(f"Received {response.json()}")

    predictions = response.json()["predictions"]
    assert all(isinstance(pred, str) for pred in predictions)  # Ensure all are strings
    assert all(
        pred in ["setosa", "versicolor", "virginica"] for pred in predictions
    )  # Validate class names
