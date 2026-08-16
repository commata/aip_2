from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import shutil
import sys
from pathlib import Path
from typing import NoReturn


RULE_XML_NAME = "Rule_forTraining.xml"


def _exit_with_rule_xml_error(message: str) -> NoReturn:
    print(f"[bt_rule_manager] {message}", file=sys.stderr)
    raise SystemExit(1)


def _resolve_rule_xml_source(
    rule_xml_path: str | Path | None,
    workspace_root: Path,
) -> Path:
    default_source = (workspace_root / RULE_XML_NAME).resolve()

    if not rule_xml_path:
        if default_source.exists():
            return default_source
        _exit_with_rule_xml_error(
            f"Rule XML path is empty and fallback does not exist: {default_source}"
        )

    source = Path(rule_xml_path)
    if not source.is_absolute():
        source = workspace_root / source
    source = source.resolve()

    if source.is_dir():
        source = (source / RULE_XML_NAME).resolve()

    if source.suffix.lower() == ".xml" and source.exists():
        return source

    if default_source.exists():
        print(
            "[bt_rule_manager] "
            f"Rule XML not found or not an .xml file: {source}. "
            f"Using fallback: {default_source}",
            file=sys.stderr,
        )
        return default_source

    _exit_with_rule_xml_error(
        "Rule XML not found and fallback is unavailable. "
        f"requested={source}, fallback={default_source}"
    )


def _resolve_rule_targets(
    workspace_root: Path,
    aliases: list[str] | tuple[str, ...] | None,
) -> list[Path]:
    names = [RULE_XML_NAME, *(aliases or ())]
    targets: list[Path] = []
    for name in names:
        candidate = Path(name)
        if candidate.name != str(candidate) or candidate.suffix.lower() != ".xml":
            raise ValueError(f"Rule XML alias must be a plain .xml filename: {name!r}")
        target = workspace_root / candidate.name
        if target not in targets:
            targets.append(target)
    return targets


@contextmanager
def activate_rule_xml(
    rule_xml_path: str | Path | None,
    workspace_root: str | Path,
    *,
    aliases: list[str] | tuple[str, ...] | None = None,
) -> Iterator[None]:
    """Temporarily activate a BT rule XML under every DLL-required filename."""
    workspace_root = Path(workspace_root).resolve()
    source = _resolve_rule_xml_source(rule_xml_path, workspace_root)
    targets = _resolve_rule_targets(workspace_root, aliases)
    snapshots: dict[Path, bytes | None] = {}
    activated: list[Path] = []
    for target in targets:
        if source == target.resolve():
            print(
                f"[bt_rule_manager] active Rule XML already in place: {target}",
                file=sys.stderr,
            )
            continue
        snapshots[target] = target.read_bytes() if target.exists() else None
        shutil.copy2(source, target)
        activated.append(target)
        print(
            f"[bt_rule_manager] activated Rule XML: {source} -> {target}",
            file=sys.stderr,
        )
    try:
        yield
    finally:
        for target in reversed(activated):
            previous = snapshots[target]
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(previous)
