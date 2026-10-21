#!/usr/bin/env python3
"""
gpt_direct_responses_api.py
============================
"GPTDirect" framework — asks the OpenAI model to produce a valid BPMN 2.0
XML file directly from a clinical process description.

Talks to the OpenAI Responses API. Instructions and input text are always
supplied externally via --instructions-file/--input-file (e.g. the
instruct_tmpl.txt/input_tmpl.txt pair assembled by batch_runner.py) — there
is no built-in system prompt.

API convention:
    import openai
    openai.api_key = "..."
    openai.responses.create(
        model=...,
        instructions=...,           # system-level prompt (was messages[0], role="system")
        input=[...],                # user/assistant turns (was the rest of messages[])
        temperature=1.0,            # must be 1 for reasoning models
        max_output_tokens=100000,   # Responses API's equivalent of max_completion_tokens
        reasoning={"effort": "high"},  # only for reasoning-capable models
    )

Retry modes (--retry-mode):
  none     Single attempt. Output saved as-is regardless of validity.
           Pure zero-shot baseline — no second chances.
  repair   One fresh generation + up to MAX_REPAIR repair rounds feeding
           validation errors back to the model. No re-generation.
  full     MAX_FRESH fresh generations × MAX_REPAIR repairs each.
           Most robust mode.

DB framework label: GPT-Direct-none | GPT-Direct-repair | GPT-Direct-full

Stdout (last two lines on success):
  GPT_DIRECT_RETRIES: <summary>
  <absolute/path/to/process.bpmn>

Exits 0 on success, 1 on failure.
"""

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import openai
from libs.template import instantiate_values
from libs.llm_util import resolve_gpt_model

MAX_FRESH  = 3
MAX_REPAIR = 3

RETRY_MODES = ("none", "repair", "full")


# ════════════════════════════════════════════════════════════════════════════
# API call — single place that talks to OpenAI
# ════════════════════════════════════════════════════════════════════════════

def _call_api(model_id: str, reasoning_effort: str | None,
              messages: list, timeout: int = 3600) -> str:
    """
    Call the Responses API.

    `messages` uses the same role/content shape the Chat Completions version
    used (a leading role="system" entry followed by user/assistant turns).
    Here the system entry is peeled off into `instructions` and the rest is
    passed as `input`, since that's the Responses API's equivalent split.

    Key facts about reasoning models (gpt-5.2, gpt-5.4, o3, o4-mini):
    - max_output_tokens is a SHARED budget for BOTH reasoning tokens AND
      visible output tokens. With xhigh effort the model can spend 50k–100k
      tokens "thinking" before producing a single character of XML.
    - If the budget runs out during reasoning, output_text is empty and
      status is "incomplete" (incomplete_details.reason="max_output_tokens").
      This is the root cause of blank responses.
    - OpenAI recommends reserving at least 25k tokens for reasoning alone.
    - Known gpt-5.4 bug: passing max_output_tokens + reasoning effort="none"
      causes the API to ignore "none" and reason anyway, burning the entire
      budget invisibly → empty output.

    What we do here:
    - max_output_tokens = 100_000 (ample for heavy reasoning + BPMN XML)
    - timeout = 3600s (1 hour per call) — gpt-5.4 xhigh can take 20–60 min
    - Check status/incomplete_details and raise a clear human-readable error
    - Log the reasoning-token vs output-token split on every call so you can
      tune budgets over time
    """
    system_msgs = [m["content"] for m in messages if m["role"] == "system"]
    instructions = system_msgs[0] if system_msgs else None
    input_items  = [m for m in messages if m["role"] != "system"]

    kwargs: dict = dict(
        model=model_id,
        input=input_items,
        temperature=1.0,
        max_output_tokens=100_000,
        timeout=timeout,
    )
    if instructions:
        kwargs["instructions"] = instructions
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = openai.responses.create(**kwargs)

    status  = response.status
    content = response.output_text or ""

    # Log token breakdown — crucial for diagnosing reasoning budget issues
    usage = response.usage
    reasoning_toks = "?"
    if usage:
        out_detail = getattr(usage, "output_tokens_details", None)
        if out_detail:
            reasoning_toks = getattr(out_detail, "reasoning_tokens", "?")
        print(
            f"  [tokens] input={usage.input_tokens}  "
            f"output={usage.output_tokens}  "
            f"reasoning≈{reasoning_toks}  "
            f"status={status}",
            flush=True,
        )

    if status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
        if reason == "max_output_tokens":
            raise RuntimeError(
                f"status='incomplete' (reason={reason}): the max_output_tokens budget was "
                f"exhausted before any visible output was produced. "
                f"Reasoning tokens consumed most of the budget "
                f"(reasoning≈{reasoning_toks} of {usage.output_tokens if usage else '?'} "
                f"total output tokens). "
                f"Lower reasoning_effort or increase max_output_tokens further."
            )
        raise RuntimeError(f"status='incomplete' (reason={reason}).")

    if not content.strip():
        raise RuntimeError(
            f"Model returned empty content (status={status!r}). "
            "Reasoning tokens likely exhausted the token budget. "
            "Try lowering reasoning_effort or increasing max_output_tokens."
        )

    return content


# ════════════════════════════════════════════════════════════════════════════
# XML extraction & validation
# ════════════════════════════════════════════════════════════════════════════

def _extract_xml(text: str) -> str:
    """Strip markdown fences and locate the start of the XML."""
    text = re.sub(r"```(?:xml|bpmn)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    for marker in ("<?xml", "<definitions"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:].strip()
    return text


def _validate_bpmn(xml_str: str) -> tuple[bool, str]:
    """Lightweight structural check. Returns (is_valid, error_message)."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"

    tag_local = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag_local != "definitions":
        return False, f"Root element must be <definitions>, got <{tag_local}>"

    bpmn_ns = "http://www.omg.org/spec/BPMN/20100524/MODEL"
    processes = (root.findall(f"{{{bpmn_ns}}}process")
                 or root.findall("process"))
    if not processes:
        return False, "No <process> element found inside <definitions>."

    task_types = ("task", "userTask", "serviceTask",
                  "startEvent", "endEvent",
                  "exclusiveGateway", "parallelGateway")
    has_node = any(
        root.find(f".//{{{bpmn_ns}}}{t}") is not None
        or root.find(f".//{t}") is not None
        for t in task_types
    )
    if not has_node:
        return False, "Process contains no tasks, gateways, or events."

    return True, ""


# ════════════════════════════════════════════════════════════════════════════
# Generation modes
# ════════════════════════════════════════════════════════════════════════════

def _single_attempt(model_id: str, reasoning_effort: str | None,
                    task_text: str, instructions: str) -> tuple[str, str]:
    """retry_mode=none — one call, save output regardless of validity."""
    print("[GPTDirect / none] Single attempt, no retries …", flush=True)
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user",   "content": task_text},
    ]
    raw = _call_api(model_id, reasoning_effort, messages)
    if not raw.strip():
        raise RuntimeError("Model returned an empty response.")

    xml_str = _extract_xml(raw)
    ok, err = _validate_bpmn(xml_str)
    if ok:
        print("  ✔ Response is valid BPMN.", flush=True)
    else:
        print(f"  ⚠ Validation issues (saved anyway): {err}", flush=True)
    return xml_str, "mode=none"


def _repair_mode(model_id: str, reasoning_effort: str | None,
                 task_text: str, instructions: str) -> tuple[str, str]:
    """retry_mode=repair — one generation + up to MAX_REPAIR repair rounds."""
    print("[GPTDirect / repair] One generation + repair loop …", flush=True)
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user",   "content": task_text},
    ]
    raw = _call_api(model_id, reasoning_effort, messages)
    repairs_done = 0

    repair_tmpl_path = os.path.join("data", "cpw", "prmpt", "repair_tmpl.txt")

    for repair in range(1, MAX_REPAIR + 1):
        xml_str = _extract_xml(raw)
        ok, err = _validate_bpmn(xml_str)
        if ok:
            print(f"  ✔ Valid after {repairs_done} repair(s).", flush=True)
            return xml_str, f"mode=repair,repairs={repairs_done}"

        print(f"  ✗ Repair {repair}/{MAX_REPAIR}: {err}", flush=True)
        if repair == MAX_REPAIR:
            print("  ⚠ Max repairs reached — saving last output.", flush=True)
            return xml_str, f"mode=repair,repairs={repairs_done},invalid"

        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": instantiate_values(repair_tmpl_path, { "{error}": err })
        })
        raw = _call_api(model_id, reasoning_effort, messages)
        repairs_done += 1

    raise RuntimeError("Repair loop exited unexpectedly.")  # unreachable


# def _full_mode(model_id: str, reasoning_effort: str | None,
#                task_text: str, instructions: str) -> tuple[str, str]:
#     """retry_mode=full — MAX_FRESH × MAX_REPAIR attempts."""
#     for fresh in range(1, MAX_FRESH + 1):
#         print(f"[GPTDirect / full] Fresh attempt {fresh}/{MAX_FRESH} …", flush=True)
#         messages = [
#             {"role": "system", "content": instructions},
#             {"role": "user",   "content": task_text},
#         ]
#         try:
#             raw = _call_api(model_id, reasoning_effort, messages)
#         except Exception as e:
#             print(f"  ✗ API call failed: {e}", flush=True)
#             continue

#         for repair in range(1, MAX_REPAIR + 1):
#             xml_str = _extract_xml(raw)
#             ok, err = _validate_bpmn(xml_str)
#             if ok:
#                 repairs = repair - 1
#                 print(f"  ✔ Valid (fresh={fresh}, repairs={repairs})", flush=True)
#                 return xml_str, f"mode=full,fresh={fresh},repairs={repairs}"

#             print(f"  ✗ Repair {repair}/{MAX_REPAIR}: {err}", flush=True)
#             if repair == MAX_REPAIR:
#                 break

#             messages.append({"role": "assistant", "content": raw})
#             messages.append({
#                 "role": "user",
#                 "content": REPAIR_PROMPT_TEMPLATE.format(error=err),
#             })
#             try:
#                 raw = _call_api(model_id, reasoning_effort, messages)
#             except Exception as e:
#                 print(f"    ✗ Repair API call failed: {e}", flush=True)
#                 break

#         print(f"  All repairs exhausted on fresh {fresh} — retrying.", flush=True)

#     raise RuntimeError(
#         f"GPTDirect (full): all {MAX_FRESH} fresh × {MAX_REPAIR} repair attempts failed."
#     )


def generate_bpmn(task_text: str, model_id: str, reasoning_effort: str | None,
                  instructions: str, retry_mode: str = "full") -> tuple[str, str]:
    """Entry point: generate BPMN. Returns (xml_string, retries_summary)."""
    if retry_mode == "none":
        return _single_attempt(model_id, reasoning_effort, task_text, instructions)
    elif retry_mode == "repair":
        return _repair_mode(model_id, reasoning_effort, task_text, instructions)
    else:
        raise f"unsupported mode: {retry_mode}"
        # return _full_mode(model_id, reasoning_effort, task_text, instructions)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="GPTDirect BPMN generator (Responses API).")
    p.add_argument("--instructions-file", required=True,
                   help="File whose content becomes the Responses API 'instructions'.")
    p.add_argument("--input-file",        required=True,
                   help="File whose content becomes the task text placed in "
                        "the Responses API 'input'.")
    p.add_argument("--api-key",      default=None)
    p.add_argument("--model",        default="GPT_4o")
    p.add_argument("--run-dir",      required=True,
                   help="Existing (or to-be-created) folder to write process.bpmn into.")
    p.add_argument(
        "--retry-mode",
        choices=RETRY_MODES,
        default="full",
        help="none | repair | full  (see module docstring for details)",
    )
    args = p.parse_args()

    # ── API key ────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No OpenAI API key. Set OPENAI_API_KEY or pass --api-key.",
              file=sys.stderr)
        sys.exit(1)
    openai.api_key = api_key

    # ── Model ──────────────────────────────────────────────────────────────
    model_id, reasoning_effort = resolve_gpt_model(args.model)
    print(
        f"[GPTDirect] model={model_id}  "
        f"reasoning_effort={reasoning_effort!r}  "
        f"retry_mode={args.retry_mode}",
        flush=True,
    )

    # ── Task text ──────────────────────────────────────────────────────────
    def _read(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    # Sanitize: remove characters that are illegal in JSON strings.
    # Null bytes, Unicode surrogates, and most control characters (except
    # tab/newline/carriage-return) will cause a 400 BadRequestError at the
    # OpenAI API even though Python can hold them in a str.
    def _sanitize(text: str) -> str:
        cleaned = []
        for ch in text:
            cp = ord(ch)
            # Allow normal whitespace (tab=9, newline=10, CR=13)
            if cp in (9, 10, 13):
                cleaned.append(ch)
                continue
            # Drop null bytes and C0/C1 control characters
            if cp < 32 or (0x7F <= cp <= 0x9F):
                continue
            # Drop Unicode surrogates (lone surrogates are invalid in JSON)
            if 0xD800 <= cp <= 0xDFFF:
                continue
            cleaned.append(ch)
        return "".join(cleaned)

    def _sanitized(text: str, label: str) -> str:
        original_len = len(text)
        cleaned      = _sanitize(text)
        if len(cleaned) != original_len:
            removed = original_len - len(cleaned)
            print(f"[GPTDirect] ⚠ Sanitized {label}: removed {removed} "
                  f"illegal character(s) (null bytes / surrogates / control chars).",
                  flush=True)
        return cleaned

    instructions_text = _sanitized(_read(args.instructions_file), "instructions text")
    task_text         = _sanitized(_read(args.input_file), "input text")

    # ── Generate ───────────────────────────────────────────────────────────
    try:
        xml_str, retries = generate_bpmn(
            task_text, model_id, reasoning_effort,
            instructions=instructions_text,
            retry_mode=args.retry_mode,
        )
    except RuntimeError as e:
        print("GPT_DIRECT_RETRIES: failed", flush=True)
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # ── Write BPMN ─────────────────────────────────────────────────────────
    run_dir = args.run_dir

    bpmn_path = os.path.join(run_dir, "process.bpmn")
    with open(bpmn_path, "w", encoding="utf-8") as f:
        if not xml_str.startswith("<?xml"):
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml_str)

    print(f"GPT_DIRECT_RETRIES: {retries}", flush=True)
    print(bpmn_path, flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
