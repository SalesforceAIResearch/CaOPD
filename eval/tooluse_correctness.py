"""
Parse first tool call (Action + Action Input) from model output and compare to golden_answer.

Golden format: list of {"Action": "tool_name", "Action_Input": "{}"}.
Accuracy is evaluated by matching the ground-truth API call, accounting for
variations in argument ordering (JSON is normalized with sort_keys).
"""
import re
import json


def _normalize_action_input(s: str) -> str:
    """Normalize JSON string for comparison: strip, parse and re-dump to handle spacing/key order."""
    s = (s or "").strip()
    if not s:
        return "{}"
    try:
        obj = json.loads(s)
        return json.dumps(obj, sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        return s.strip()


def _extract_json_after_prefix(text: str, prefix: str) -> str:
    """Find prefix (e.g. 'Action Input:') and extract balanced {...} after it."""
    idx = text.find(prefix)
    if idx < 0:
        return ""
    start_brace = text.find("{", idx)
    if start_brace < 0:
        return "{}"
    depth = 0
    for i in range(start_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start_brace : i + 1]
    return "{}"


def parse_first_tool_call(text: str) -> dict | None:
    """
    Extract first Action and Action Input from model output.

    Returns {"Action": str, "Action_Input": str} or None if not found.
    """
    action_match = re.search(r"Action:\s*([^\n]+)", text, re.IGNORECASE)
    if not action_match:
        return None
    action_name = action_match.group(1).strip()

    raw_input = _extract_json_after_prefix(text, "Action Input:")
    if not raw_input:
        raw_input = _extract_json_after_prefix(text, "Action_Input:")
    action_input = _normalize_action_input(raw_input or "{}")

    return {"Action": action_name, "Action_Input": action_input}


def normalize_golden(g: dict) -> dict:
    """Normalize golden entry: support both 'Action_Input' and 'Action Input' keys."""
    action = (g.get("Action") or g.get("action") or "").strip()
    inp = g.get("Action_Input") or g.get("Action Input") or "{}"
    if isinstance(inp, dict):
        inp = json.dumps(inp, sort_keys=True)
    else:
        inp = _normalize_action_input(str(inp))
    return {"Action": action, "Action_Input": inp}


def check_correctness_one(response: str, golden_answer: list) -> tuple[bool, dict | None]:
    """
    Compare model output to golden_answer (list of Action/Action_Input dicts).
    Returns (correct: bool, parsed_pred: dict | None).
    """
    parsed = parse_first_tool_call(response)
    if not golden_answer:
        return parsed is None, parsed
    golden_first = normalize_golden(golden_answer[0])
    if parsed is None:
        return False, None
    pred_norm = {
        "Action": (parsed.get("Action") or "").strip(),
        "Action_Input": _normalize_action_input(parsed.get("Action_Input") or "{}"),
    }
    correct = (
        pred_norm["Action"] == golden_first["Action"]
        and pred_norm["Action_Input"] == golden_first["Action_Input"]
    )
    return correct, parsed
