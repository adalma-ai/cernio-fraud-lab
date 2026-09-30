"""Minimal client for the hosted TabPFN REST API.

The token is read from TABPFN_TOKEN and is never written to disk or logs.
"""

from __future__ import annotations

import csv
import io
import os
from typing import Any

import requests

BASE_URL = "https://api.priorlabs.ai"
MODEL_PATH = "v3.5_default"


class TabPFNAPIError(RuntimeError):
    """Raised with a sanitized API error."""


def _csv(rows: list[list[float]] | list[int], header: list[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row if isinstance(row, list) else [row])
    return buffer.getvalue().encode()


class TabPFNRestClassifier:
    """Fit once and return class-1 probabilities from TabPFN-3.5."""

    def __init__(self, token: str | None = None, timeout: int = 180, session: Any | None = None) -> None:
        self.token = token or os.environ.get("TABPFN_TOKEN")
        if not self.token:
            raise TabPFNAPIError("Set TABPFN_TOKEN before running the hosted TabPFN benchmark.")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.fitted_train_set_id: str | None = None
        self.feature_names: list[str] = []

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(f"{BASE_URL}{path}", headers=self.headers, json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            try:
                body = response.json()
                detail = body.get("message") or body.get("error_code") or "request failed"
            except ValueError:
                detail = "request failed"
            raise TabPFNAPIError(f"TabPFN API {path} returned {response.status_code}: {detail}")
        return response.json()

    def _put(self, info: dict[str, Any], content: bytes) -> None:
        response = self.session.put(
            info["signed_urls"][0],
            data=content,
            headers=info.get("required_headers", {}),
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise TabPFNAPIError(f"Signed upload returned {response.status_code}")

    def fit(self, x: list[list[float]], y: list[int], feature_names: list[str]) -> "TabPFNRestClassifier":
        self.feature_names = feature_names
        prep = self._post(
            "/tabpfn/prepare_train_set_upload",
            {"x_train_info": {"format": "csv"}, "y_train_info": {"format": "csv"}},
        )
        self._put(prep["x_train_info"], _csv(x, feature_names))
        self._put(prep["y_train_info"], _csv(y, ["label"]))
        fitted = self._post(
            "/tabpfn/fit",
            {
                "train_set_upload_id": prep["train_set_upload_id"],
                "task": "classification",
                "tabpfn_config": {"model_path": MODEL_PATH},
            },
        )
        self.fitted_train_set_id = fitted["fitted_train_set_id"]
        return self

    def predict_proba(self, x: list[list[float]]) -> list[float]:
        if not self.fitted_train_set_id:
            raise TabPFNAPIError("Call fit before predict_proba.")
        prep = self._post(
            "/tabpfn/prepare_test_set_upload",
            {"fitted_train_set_id": self.fitted_train_set_id, "x_test_info": {"format": "csv"}},
        )
        self._put(prep["x_test_info"], _csv(x, self.feature_names))
        result = self._post(
            "/tabpfn/predict",
            {
                "test_set_upload_id": prep["test_set_upload_id"],
                "fitted_train_set_id": self.fitted_train_set_id,
                "task_config": {"task": "classification", "predict_params": {"output_type": "probas"}},
            },
        )
        prediction = result.get("prediction")
        if not isinstance(prediction, list):
            raise TabPFNAPIError("TabPFN API returned an unexpected prediction payload.")
        probabilities = []
        for item in prediction:
            if isinstance(item, list):
                probabilities.append(float(item[-1]))
            elif isinstance(item, dict):
                probabilities.append(float(item.get("1", item.get(1, 0.0))))
            else:
                probabilities.append(float(item))
        return probabilities
