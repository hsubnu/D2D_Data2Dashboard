import json

def serialize_memory(mem) -> str:
    """
    Normalise `state["memory"]` to a JSON-string list.

    • Accepts None, str, list, or dict.
    • Always returns a string like '[]' or '["…"]'.
    """
    if mem in (None, "None"):
        return "[]"

    if isinstance(mem, str):
        try:
            parsed = json.loads(mem)
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
            return json.dumps([parsed], ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps([mem], ensure_ascii=False)

    if isinstance(mem, list):
        return json.dumps(mem, ensure_ascii=False)

    if isinstance(mem, dict):
        return json.dumps([mem], ensure_ascii=False)

    # fallback
    return "[]"
