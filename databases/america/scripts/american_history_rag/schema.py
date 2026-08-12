from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    source: str
    title: str
    text: str
    url: str
    date: str = ""
    authors: list[str] = field(default_factory=list)
    place: str = ""
    genre: str = ""
    rights: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    id: str
    document_id: str
    source: str
    title: str
    text: str
    url: str
    date: str = ""
    authors: list[str] = field(default_factory=list)
    place: str = ""
    genre: str = ""
    rights: str = ""
    ordinal: int = 0

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

