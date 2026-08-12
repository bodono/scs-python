#!/usr/bin/env python3
"""Generate CIBW_SKIP based on Python 3.15 SciPy wheel availability."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

try:
    import tomllib
except ImportError:
    from pip._vendor import tomli as tomllib

try:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.tags import Tag, platform_tags
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from packaging.version import InvalidVersion, Version
except ImportError:
    # pip always vendors packaging, including on a fresh setup-python install.
    from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
    from pip._vendor.packaging.tags import Tag, platform_tags
    from pip._vendor.packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from pip._vendor.packaging.version import InvalidVersion, Version


PYTHON_VERSION = "3.15"
PYTHON_TAG = "cp315"
ABIS = ("cp315", "cp315t")
SCIPY_JSON_URL = "https://pypi.org/pypi/scipy/json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configured_skips() -> list[str]:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    skip = config["tool"]["cibuildwheel"].get("skip", [])

    if isinstance(skip, str):
        return skip.split()
    if isinstance(skip, list) and all(isinstance(item, str) for item in skip):
        return skip.copy()
    raise TypeError("tool.cibuildwheel.skip must be a string or list of strings")


def scipy_releases() -> dict[str, list[dict[str, object]]]:
    request = urllib.request.Request(
        SCIPY_JSON_URL,
        headers={"User-Agent": "scs-python cibuildwheel dependency gate"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["releases"]


def supports_python(file_info: dict[str, object]) -> bool:
    requires_python = file_info.get("requires_python")
    if not requires_python:
        return True
    try:
        return SpecifierSet(str(requires_python)).contains(PYTHON_VERSION)
    except InvalidSpecifier:
        return False


def compatible_wheel(
    releases: dict[str, list[dict[str, object]]], abi: str
) -> str | None:
    supported_tags = {
        Tag(PYTHON_TAG, abi, platform_tag) for platform_tag in platform_tags()
    }
    candidates: list[tuple[Version, list[dict[str, object]]]] = []

    for release, files in releases.items():
        try:
            version = Version(release)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease:
            continue
        if any(
            not file_info.get("yanked", False) and supports_python(file_info)
            for file_info in files
        ):
            candidates.append((version, files))

    if not candidates:
        return None

    _, files = max(candidates, key=lambda candidate: candidate[0])
    for file_info in files:
        if (
            file_info.get("packagetype") != "bdist_wheel"
            or file_info.get("yanked", False)
            or not supports_python(file_info)
        ):
            continue

        filename = str(file_info["filename"])
        try:
            wheel_tags = parse_wheel_filename(filename)[3]
        except InvalidWheelFilename:
            continue
        if supported_tags.intersection(wheel_tags):
            return filename

    return None


def main() -> None:
    skips = configured_skips()
    releases = scipy_releases()

    for abi in ABIS:
        wheel = compatible_wheel(releases, abi)
        if wheel:
            print(f"SciPy wheel available for {abi}: {wheel}", file=sys.stderr)
        else:
            print(
                f"No compatible stable SciPy wheel for {abi}; skipping it",
                file=sys.stderr,
            )
            skips.append(f"{abi}-*")

    print(" ".join(dict.fromkeys(skips)))


if __name__ == "__main__":
    main()
