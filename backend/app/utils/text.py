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


def split_budget(
    available: int,
    wanted: dict[str, int],
    weights: dict[str, float]
) -> dict[str, int]:
    """Divide `available` characters between named prompt parts.

    `wanted` is how many characters each part would like; `weights` is the
    share (summing to 1) each may use at first. Space a short part does not
    use is handed to the parts that still want more, biggest share first.
    The result never adds up to more than `available`.
    """
    limits = {
        name: min(size, int(available * weights[name]))
        for name, size in wanted.items()
    }

    leftover = available - sum(limits.values())

    for name in sorted(wanted, key=lambda key: -weights[key]):
        if leftover <= 0:
            break

        extra = min(wanted[name] - limits[name], leftover)

        if extra > 0:
            limits[name] += extra
            leftover -= extra

    return limits
