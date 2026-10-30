"""Public SDK release metadata and a lazy client for release-profile checks."""

from __future__ import annotations

from ._rpc import SDKCompatibilityError, request

SDK_VERSION = "1.0.0"


def resolve_sdk_version(value: str | None = None, *, legacy: bool = False) -> str:
    """Ask the installed tool which immutable SDK release the package targets."""
    return request("resolve_sdk_version", {"value": value, "legacy": legacy}, error_type=SDKCompatibilityError)


def load_profile(sdk_version: str | None = None) -> dict:
    """Return compatibility metadata checked by the external runtime."""
    return request("load_profile", {"sdk_version": sdk_version}, error_type=SDKCompatibilityError)
