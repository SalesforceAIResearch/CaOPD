"""
Extract verbal confidence from model output.

Supports multiple formats:
  - Inline "Confidence: 0.75" (preferred; matches CaOPD training format)
  - <confidence>0.75</confidence> tag
  - Various fallback patterns (percentage, 1-10 scale, etc.)
"""
import re


def _normalize_value(val: float, raw: str) -> float:
    """Map value to [0, 1].  ``raw`` is the matched string (e.g. '8', '85%')."""
    if "%" in raw or (val > 10 and val <= 100):
        return val / 100.0
    if 1 < val <= 10:
        return val / 10.0
    if 0 <= val <= 1:
        return val
    if val > 100:
        return val / 100.0
    return val / 10.0


def confidence_extractor(response: str) -> tuple:
    """
    Returns ``(format_ok, confidence_float)``.

    ``format_ok``: 1 if a valid number in [0, 1] was found, else 0.
    Prefers inline ``Confidence: X.XX`` (last occurrence); then ``<confidence>`` tags
    and other fallbacks.
    """
    def _try_extract(text: str):
        # 1) Inline "Confidence: X.XX" (preferred; same format as training)
        inline_matches = list(re.finditer(r"Confidence:\s*([\d.]+)", text, re.IGNORECASE))
        if inline_matches:
            raw = inline_matches[-1].group(1).strip()
            try:
                val = max(0.0, min(1.0, float(raw)))
                return 1, val
            except ValueError:
                m = re.search(r"[\d.]+", raw)
                if m:
                    return 1, max(0.0, min(1.0, float(m.group())))

        # 2) Strict: <confidence>...</confidence>
        conf_matches = re.findall(r"<confidence>(.*?)</confidence>", text, re.DOTALL | re.MULTILINE)
        if conf_matches:
            last = conf_matches[-1].strip()
            if last:
                try:
                    return 1, _normalize_value(float(last), last)
                except ValueError:
                    m = re.search(r"-?\d+(?:\.\d+)?", last)
                    if m:
                        return 1, _normalize_value(float(m.group()), last)
            return 0, 0.0

        # 3) Fallback: <confidence>X without closing tag
        fallback = list(re.finditer(r"<confidence>\s*(\d+(?:\.\d+)?)\s*%?", text))
        if fallback:
            val = float(fallback[-1].group(1))
            return 1, _normalize_value(val, fallback[-1].group(0))

        return None

    result = _try_extract(response)
    if result is not None:
        return result
    return 0, 0.0
