"""Standard-library file operations executed inside the repository sandbox."""

import re
from pathlib import Path


def normalized_text(text):
    """Return normalized text and the original character spans it came from."""
    translations = {
        **dict.fromkeys("\u2018\u2019\u201a\u201b", "'"),
        **dict.fromkeys("\u201c\u201d\u201e\u201f", '"'),
        **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"),
        **dict.fromkeys(
            "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000",
            " ",
        ),
    }
    chars, spans = [], []
    index = 0
    while index < len(text):
        end = index + 1
        char = text[index]
        if char == "\r":
            if end < len(text) and text[end] == "\n":
                end += 1
            char = "\n"
        chars.append(translations.get(char, char))
        spans.append((index, end))
        index = end
    normalized = "".join(chars)
    removed = set()
    for match in re.finditer(r"[ \t]+(?=\n|$)", normalized):
        removed.update(range(match.start(), match.end()))
    return (
        "".join(char for i, char in enumerate(chars) if i not in removed),
        [span for i, span in enumerate(spans) if i not in removed],
    )


def find_matches(text, old):
    matches = []
    start = 0
    while True:
        index = text.find(old, start)
        if index < 0:
            return matches
        matches.append((index, index + len(old)))
        start = index + 1


def apply_edits(original, edits, path):
    """Resolve every edit against the original before constructing any changes."""
    if not edits:
        raise ValueError("Provide at least one edit.")
    replacements = []
    normalized = None
    for edit in edits:
        old, new = edit["old_text"], edit["new_text"]
        if not old:
            raise ValueError("old_text must not be empty; use write to create a file.")
        matches = find_matches(original, old)
        if not matches:
            if normalized is None:
                normalized = normalized_text(original)
            needle, _ = normalized_text(old)
            if needle:
                matches = [
                    (normalized[1][start][0], normalized[1][end - 1][1])
                    for start, end in find_matches(normalized[0], needle)
                ]
        if not matches:
            raise ValueError(
                f"Could not find the exact text in {path}. "
                "The old text must match exactly including all whitespace and newlines."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Found multiple matches in {path}. Provide more context in old_text "
                "to identify a unique replacement region."
            )
        replacements.append((*matches[0], new))
    replacements.sort(key=lambda item: item[0])
    # Sandbox Python versions may predate itertools.pairwise.
    for previous, current in zip(replacements, replacements[1:]):  # noqa: RUF007
        if current[0] < previous[1]:
            raise ValueError(f"Replacement regions overlap in {path}.")
    result = original
    for start, end, new in reversed(replacements):
        result = result[:start] + new + result[end:]
    return result


def execute_operation(payload):
    path = Path(payload["path"])
    operation = payload["operation"]
    if operation == "write":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(payload["content"])
        return f"Wrote {path}."
    with path.open(encoding="utf-8", newline="") as stream:
        original = stream.read()
    if operation == "read":
        lines = original.splitlines()
        start = payload["offset"] - 1
        selected = lines[start : start + payload["limit"]]
        numbered = "\n".join(
            f"{index}: {line}" for index, line in enumerate(selected, start + 1)
        )
        return f"{path} ({len(lines)} lines total)\n{numbered}"
    if operation == "edit":
        updated = apply_edits(original, payload["edits"], path)
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(updated)
        return f"Applied {len(payload['edits'])} edit(s) to {path}."
    raise ValueError(f"Unknown file operation: {operation}")
