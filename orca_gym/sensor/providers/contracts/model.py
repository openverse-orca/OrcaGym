"""Validate physical input requests through the bundled Orca Sensor executable."""

from __future__ import annotations

from typing import Any

from ._rpc import XmlModelError, request as tool_request


def validate_bound_site_request(request: dict[str, Any]) -> list[int]:
    return tool_request("validate_bound_site_request", {"request": request}, error_type=XmlModelError)


def validate_raycast_request(request: dict[str, Any]) -> None:
    tool_request("validate_raycast_request", {"request": request}, error_type=XmlModelError)
