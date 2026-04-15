"""
Parse model answer from chemistry MCQ output and compare to golden_answer.

Expected model output format:
  <reasoning>...</reasoning>
  <answer>A</answer>

Golden format: list of {"Answer": "A"} (uppercase letter A/B/C/D).
"""
import re


def parse_mcq_answer(text: str) -> str | None:
    """
    Extract the MCQ answer letter (A/B/C/D) from model output.

    Parsing priority (highest to lowest):
      1. <answer>A</answer>
      2. <answer>A  (unclosed tag, truncated output)
      3. Bare letter on its own line (last occurrence)
      4. "Answer: A" / "answer is A" (last occurrence)

    Returns uppercase letter or None if unparseable.
    """
    # 1. Primary: <answer>A</answer>
    match = re.search(r"<answer>\s*([A-Da-d])\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).upper()

    # 2. Fallback: <answer>A without closing tag
    match = re.search(r"<answer>\s*([A-Da-d])\s*$", text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()

    # 3. Bare letter on its own line — scan from the END
    lines = [line.strip() for line in text.strip().splitlines()]
    for line in reversed(lines):
        if re.fullmatch(r"[A-Da-d]", line):
            return line.upper()

    # 4. "Answer: A" / "answer is A" — use the LAST match
    matches = list(re.finditer(r"[Aa]nswer(?:\s+is)?[:\s]+([A-Da-d])\b", text))
    if matches:
        return matches[-1].group(1).upper()

    return None


def check_correctness_one(response: str, golden_answer: list) -> tuple[bool, str | None]:
    """
    Compare model output to golden_answer (list of {"Answer": "X"} dicts).
    Returns (correct: bool, parsed_pred: str | None).
    """
    parsed = parse_mcq_answer(response)
    if not golden_answer:
        return parsed is None, parsed
    expected = (golden_answer[0].get("Answer") or "").strip().upper()
    if not expected:
        return parsed is None, parsed
    correct = parsed is not None and parsed == expected
    return correct, parsed
