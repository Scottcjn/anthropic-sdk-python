from __future__ import annotations

import httpx

from anthropic._exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _response(status_code: int, *, request_id: str | None = "req_123") -> httpx.Response:
    headers = {}
    if request_id is not None:
        headers["request-id"] = request_id

    return httpx.Response(status_code, headers=headers, request=_request())


def test_api_status_error_exposes_response_details() -> None:
    body = {"error": {"message": "rate limited"}}
    response = _response(429, request_id="req_rate_limit")

    error = APIStatusError("Too many requests", response=response, body=body)

    assert str(error) == "Too many requests"
    assert error.message == "Too many requests"
    assert error.body == body
    assert error.request is response.request
    assert error.response is response
    assert error.status_code == 429
    assert error.request_id == "req_rate_limit"


def test_api_status_error_allows_missing_request_id() -> None:
    response = _response(500, request_id=None)

    error = APIStatusError("Server error", response=response, body="not json")

    assert error.status_code == 500
    assert error.request_id is None
    assert error.body == "not json"


def test_status_error_subclass_keeps_literal_status_code() -> None:
    response = _response(429)

    error = RateLimitError("Rate limit exceeded", response=response, body=None)

    assert error.status_code == 429
    assert isinstance(error, APIStatusError)


def test_response_validation_error_uses_default_and_custom_message() -> None:
    response = _response(200)

    default_error = APIResponseValidationError(response, body={"unexpected": True})
    custom_error = APIResponseValidationError(response, body=None, message="Bad response shape")

    assert default_error.message == "Data returned by API invalid for expected schema."
    assert default_error.status_code == 200
    assert default_error.response is response
    assert default_error.body == {"unexpected": True}
    assert custom_error.message == "Bad response shape"
    assert str(custom_error) == "Bad response shape"


def test_connection_errors_attach_request_without_response_body() -> None:
    request = _request()

    connection_error = APIConnectionError(request=request)
    timeout_error = APITimeoutError(request=request)

    assert connection_error.request is request
    assert connection_error.body is None
    assert connection_error.message == "Connection error."
    assert timeout_error.request is request
    assert timeout_error.body is None
    assert "Request timed out or interrupted" in timeout_error.message
