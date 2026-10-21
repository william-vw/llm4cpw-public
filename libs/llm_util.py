# ════════════════════════════════════════════════════════════════════════════
# Model resolution
# ════════════════════════════════════════════════════════════════════════════

# Maps pipeline token → (openai_model_id, reasoning_effort | None)
# reasoning_effort is forwarded as a kwarg only when not None.
_MODEL_MAP: dict[str, tuple[str, str | None]] = {
    # GPT-4 family — no reasoning_effort
    "GPT_4":                    ("gpt-4",    None),
    "GPT_4_32K":                ("gpt-4-32k", None),
    "GPT_4o":                   ("gpt-4o",   None),
    "GPT_4o1":                  ("gpt-4o",   None),
    # GPT-5.2 family
    "GPT_5o2":                  ("gpt-5.2",  "medium"),
    "GPT_5o2_low_reasoning":    ("gpt-5.2",  "low"),
    "GPT_5o2_medium_reasoning": ("gpt-5.2",  "medium"),
    "GPT_5o2_high_reasoning":   ("gpt-5.2",  "high"),
    # GPT-5.4 family
    "GPT_5o4":                  ("gpt-5.4",  "medium"),
    "GPT_5o4_medium_reasoning": ("gpt-5.4",  "medium"),
    "GPT_5o4_high_reasoning":   ("gpt-5.4",  "high"),
    "GPT_5o4_xhigh_reasoning":  ("gpt-5.4",  "xhigh"),
}

def resolve_gpt_model(model_str: str) -> tuple[str, str | None]:
    """Return (openai_model_id, reasoning_effort_or_None)."""
    if model_str in _MODEL_MAP:
        return _MODEL_MAP[model_str]

    # Passthrough for raw OpenAI IDs (contain "-" or ".")
    # Detect reasoning_effort from trailing suffix convention
    if any(c in model_str for c in ("-", ".")):
        effort = None
        for suffix, eff in [("xhigh", "xhigh"), ("high", "high"),
                             ("medium", "medium"), ("low", "low")]:
            if model_str.endswith(f"-{suffix}"):
                model_str = model_str[: -(len(suffix) + 1)]
                effort = eff
                break
        return model_str, effort

    print(f"[GPTDirect] Warning: unknown model token {model_str!r}, "
          "falling back to gpt-4o", flush=True)
    return "gpt-4o", None