"""Apply search/replace edits to a file's text.

Models are much more reliable at "replace this snippet with that" than at
re-typing a whole file, and the edit is a few lines instead of the whole
file, which keeps answers inside the token budget.
"""
import difflib


class EditError(ValueError):
    """An edit could not be applied. The message is shown to the model."""


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n")


def _closest_line(text: str, search: str) -> str:
    first_line = next(
        (line.strip() for line in search.splitlines() if line.strip()), ""
    )

    if not first_line:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    matches = difflib.get_close_matches(first_line, lines, n=1, cutoff=0.6)

    return matches[0] if matches else ""


def apply_edits(content: str, edits: list, path: str = "file") -> str:
    """Return `content` with every edit applied, in order.

    Each edit has `search` (text that must occur exactly once) and
    `replace`. Line endings are matched loosely so a CRLF file can be
    edited with LF snippets; the file's own style is kept.
    """
    if not edits:
        raise EditError(f"No edits given for {path}.")

    uses_crlf = "\r\n" in content
    text = _normalise(content)

    for number, edit in enumerate(edits, start=1):
        search = _normalise(edit.search)
        replace = _normalise(edit.replace)

        if not search.strip():
            raise EditError(
                f"Edit {number} for {path} has an empty search text. "
                "To create or fully rewrite a file, use `content` instead."
            )

        count = text.count(search)

        if count == 0:
            hint = _closest_line(text, search)
            message = (
                f"Edit {number} for {path}: the search text was not found. "
                "It must match the file exactly, including indentation."
            )

            if hint:
                message += f" Closest existing line: {hint!r}"

            raise EditError(message)

        if count > 1:
            raise EditError(
                f"Edit {number} for {path}: the search text occurs "
                f"{count} times. Include more surrounding lines so it "
                "matches exactly once."
            )

        text = text.replace(search, replace, 1)

    if uses_crlf:
        text = text.replace("\n", "\r\n")

    return text
