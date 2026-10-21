#!/usr/bin/env python3
import os, ast, re, io
import xml.etree.ElementTree as ET
from openai import OpenAI
from libs.llm_util import resolve_gpt_model
import pandas as pd

# ------------------------------------------------------------------
# Initialize OpenAI client (requires OPENAI_API_KEY in your environment)
# ------------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_MODEL = "gpt-5.4-medium"
DEFAULT_REASONING_EFFORT = "medium"   # low | medium | high | xhigh | None

def align(a_path: str, b_path: str, out_path: str, model=DEFAULT_MODEL, verbose=False, print_map=False):
    set_a = extract_activity_names(a_path)
    set_b = extract_activity_names(b_path)

    # first alignment
    init_resp = original_alignment(set_a, set_b, model, verbose)
    if verbose:
        print("initial alignment:", init_resp)
        print("\n")

    # revision
    rev_resp = revise_alignment(set_a, set_b, init_resp, model, verbose)
    if verbose:
        print("revised alignment:", rev_resp)
        print("\n")

    map = pd.read_csv(io.StringIO(rev_resp))
    if print_map:
        folder = os.path.dirname(out_path)
        map.to_csv(os.path.join(folder, "label_align.csv"))

    # if no match, use original label from A
    mask = map['Best match in B']=='No match'
    map.loc[mask, 'Best match in B'] = map[mask]['Label from A']

    final_map = map.set_index('Label from A')['Best match in B'].to_dict()
    if verbose:
        print("final mapping:")
        for a, c in final_map.items():
            print(f"- {a} -> {c}")

    # apply to BPMN
    update_activity_names(a_path, out_path, final_map)

    
def _call_responses_api(instructions: str, prompt: str, model: str,
                         reasoning_effort: str | None,
                         max_output_tokens: int = 100_000,
                         timeout: int = 3600) -> str:
    """
    Call the OpenAI Responses API and return the output text.

    reasoning_effort must be None for non-reasoning models (e.g. gpt-4o).
    For reasoning models, temperature must be 1 and max_output_tokens is a
    shared budget for both invisible reasoning tokens and visible output —
    an exhausted budget surfaces as status='incomplete' with empty output.
    """
    kwargs: dict = dict(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": prompt}],
        temperature=1.0 if reasoning_effort else 0,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
    )
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**kwargs)

    status  = response.status
    content = response.output_text or ""

    if status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else "unknown"
        raise RuntimeError(
            f"status='incomplete' (reason={reason}). "
            "Lower reasoning_effort or increase max_output_tokens."
        )
    if not content.strip():
        raise RuntimeError(f"Model returned empty content (status={status!r}).")

    return content

def get_namespace(tag):
    if tag.startswith('{'):
        return tag[1:].split('}', 1)[0]
    return ''

def extract_activity_names(file_path):
    try:
        tree = ET.parse(file_path)
    except Exception as e:
        raise RuntimeError(f"Error parsing {file_path}: {e}")
    root = tree.getroot()
    ns = get_namespace(root.tag)
    activity_tags = [
        'task','userTask','scriptTask','serviceTask','businessRuleTask',
        'manualTask','sendTask','receiveTask','subProcess','callActivity'
    ]
    names = []
    for tag in activity_tags:
        full_tag = f"{{{ns}}}{tag}"
        for elt in root.findall(f'.//{full_tag}'):
            name = elt.get('name')
            if name:
                names.append(name)
    return names

def original_alignment(set_a, set_b, model, verbose):
    model, reasoning_effort = resolve_gpt_model(model)
    if verbose:
        print(f"[Getting alignment] model={model}  reasoning_effort={reasoning_effort}")

    prompt = f"""
I have two sets of activity labels. I want align labels from Set A to the labels from Set B if they mean the same thing (it is acceptable if a few details are different). A label from set A is aligned with at most one label in Set B, and a label in Set B is aligned to at most one label from Set A. Given a label from Set A, if none of the labels in Set B mean the same thing, say 'No match'. Return your answer as a CSV table with three columns: (Label from A), (Best match in B), and (Explanation). Don't use any markdown syntax. Here are the labels:
Set A: {set_a}
Set B: {set_b}
"""
    text = _call_responses_api(
        instructions="You are a helpful assistant that aligns labels.",
        prompt=prompt,
        model=model,
        reasoning_effort=reasoning_effort,
    )

    return text

def revise_alignment(set_a, set_b, initial_response, model, verbose):
    model, reasoning_effort = resolve_gpt_model(model)
    if verbose:
        print(f"[Getting alignment revision] model={model}  reasoning_effort={reasoning_effort}")

    prompt = f"""
Below are two sets of activity labels. Your prior task was to align labels from Set A to the labels from Set B if they mean the same thing (it is acceptable if a few details are different). A label from set A is aligned with at most one label in Set B, and a label in Set B is aligned to at most one label from Set A. Given a label from Set A, if none of the labels in Set B mean the same thing, say 'No match'.

Below is the original alignment response you gave, together with the original sets:

{initial_response}

Set A: {set_a}
Set B: {set_b}

Please revise your alignment, ensuring that all alignments are correct and the full activity names are used. 
Return your answer as a CSV table with three columns: (Label from A), (Best match in B), and (Explanation).
Don't use any markdown syntax.
"""
    return _call_responses_api(
        instructions="You are a helpful assistant that aligns labels.",
        prompt=prompt,
        model=model,
        reasoning_effort=reasoning_effort,
    )

# ------------------------------------------------------------------
# 4) Apply mapping to rename activities
# ------------------------------------------------------------------
def update_activity_names(input_file, output_file, mapping):
    tree = ET.parse(input_file)
    root = tree.getroot()
    for elt in root.iter():
        name = elt.get('name')
        if name in mapping:
            elt.set('name', mapping[name])
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)
    print(f"Updated BPMN file saved as: {output_file}")

# ------------------------------------------------------------------
# 5) Main flow
# ------------------------------------------------------------------
if __name__ == "__main__":
    file_ai   = "/Users/wvw/git/pm/llm4cpw/pipeline/out/stroke 1-gptdirect_gpt_5.6_low-mg-ks/Run-stroke 1-gptdirect_gpt_5.6_low-mg-ks-attempt1-20260910_132614/process.bpmn"
    file_gold = "/Users/wvw/git/pm/llm4cpw/pipeline/data/ref/Stroke 1.bpmn"
    out_file  = "/Users/wvw/git/pm/llm4cpw/pipeline/out/stroke 1-gptdirect_gpt_5.6_low-mg-ks/Run-stroke 1-gptdirect_gpt_5.6_low-mg-ks-attempt1-20260910_132614/process-test.bpmn"

    align(file_ai, file_gold, out_file, model=DEFAULT_MODEL, verbose=True)