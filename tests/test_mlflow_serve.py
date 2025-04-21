"""Tests for mlflow serve."""

import json
import pathlib

import pandas as pd
import requests
from loguru import logger

BASE_URL = "http://127.0.0.1:5088"

test_data = {
    "sepal length (cm)": 6.1,
    "sepal width (cm)": 2.8,
    "petal length (cm)": 4.7,
    "petal width (cm)": 1.2,
}

pandas_df = pd.DataFrame([test_data])

payload_dataframe_split = json.dumps(
    {"dataframe_split": pandas_df.to_dict(orient="split")}
)

def test_inference_server_invocations() -> None:
    """Test model invocations using split dataframe format. 
       Verifies successful response and valid prediction format."""
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
    """Test model predictions with complete dataset. 
       Validates response status and prediction class membership."""
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

