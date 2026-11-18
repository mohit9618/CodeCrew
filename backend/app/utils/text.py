def clamp_text(
    text: str,
    limit: int,
    label: str = "text",
    keep: str = "head"
) -> str:
    """Shorten text so a prompt cannot blow the model's token budget.

    keep="head" keeps the start of the text (task descriptions, files).
    keep="tail" keeps the end (test output, where pytest prints its
    failure summary last).
    """
    text = "" if text is None else str(text)

    if len(text) <= limit:
        return text

    print(
        f"[Prompt] {label} truncated from {len(text):,} "
        f"to {limit:,} characters to fit the token budget."
    )

    if keep == "tail":
        return f"... ({label} truncated)\n" + text[-limit:]

    return text[:limit] + f"\n... ({label} truncated)"
