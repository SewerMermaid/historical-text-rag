from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceManifest:
    id: str
    name: str
    protocol: str
    endpoint: str
    rights_note: str
    options: dict[str, Any]

    @property
    def status(self) -> str:
        """Normalized availability: enabled, conditional, or disabled."""
        value = self.options.get("status", True)
        if value is True:
            return "enabled"
        if value is False:
            return "disabled"
        if str(value).lower() == "conditional":
            return "conditional"
        raise ValueError(f"Invalid status {value!r} for source {self.id!r}")


@dataclass(frozen=True)
class CountryManifest:
    id: str
    name: str
    language: str
    language_name: str
    native_query_default: str
    publication_cutoff: int
    cutoff_note: str
    sources: tuple[SourceManifest, ...]

    def source(self, source_id: str) -> SourceManifest:
        for source in self.sources:
            if source.id == source_id:
                return source
        choices = ", ".join(source.id for source in self.sources)
        raise ValueError(f"Unknown source {source_id!r} for {self.id}; choose one of: {choices}")


def _manifest_root() -> Path:
    configured = os.environ.get("HISTORY_RAG_DATABASES")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "america" / "manifest.json").is_file():
            raise ValueError(f"HISTORY_RAG_DATABASES has no America manifest: {root}")
        return root
    # package/databases/america/scripts/american_history_rag -> repository/databases
    installed = Path(__file__).resolve().parents[3]
    if (installed / "america" / "manifest.json").is_file():
        return installed
    checkout = Path.cwd() / "databases"
    if (checkout / "america" / "manifest.json").is_file():
        return checkout.resolve()
    raise RuntimeError(
        "Cannot locate country manifests; run from the checkout or set HISTORY_RAG_DATABASES"
    )


def country_manifest(country: str) -> CountryManifest:
    if not re.fullmatch(r"[a-z][a-z0-9-]*", country):
        raise ValueError(f"Invalid country identifier: {country!r}")
    path = _manifest_root() / country / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Unsupported country {country!r}; expected manifest at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("id") != country:
        raise ValueError(f"Manifest ID {raw.get('id')!r} does not match directory {country!r}")
    return CountryManifest(
        id=raw["id"], name=raw["name"], language=raw["language"],
        language_name=raw["language_name"],
        native_query_default=raw["native_query_default"],
        publication_cutoff=int(raw["publication_cutoff"]),
        cutoff_note=raw["cutoff_note"],
        sources=tuple(SourceManifest(
            id=source["id"], name=source["name"], protocol=source["protocol"],
            endpoint=source["endpoint"], rights_note=source["rights_note"],
            options=dict(source.get("options", {})),
        ) for source in raw.get("sources", [])),
    )


def data_dir(country: str) -> Path:
    country_manifest(country)
    return _manifest_root() / country / "data"


def supported_countries() -> tuple[str, ...]:
    root = _manifest_root()
    countries = sorted(path.parent.name for path in root.glob("*/manifest.json"))
    if "america" in countries:
        countries.remove("america")
        countries.insert(0, "america")
    return tuple(countries)
