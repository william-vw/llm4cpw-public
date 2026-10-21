#!/usr/bin/env python3
# batch_runner.py
"""
Headless batch runner — invokes pipeline.py the same way streamlit_runner.py
does (build a JSON manifest, launch `python -u pipeline.py --batch-manifest ...`
as a subprocess, stream its stdout), without needing the Streamlit UI.

Run with:  OPENAI_API_KEY=sk-... python batch_runner.py
"""

import re
import json
import os
import subprocess
import sys
import tempfile
import argparse

# ── Constants (mirrors streamlit_runner.py) ─────────────────────────────────
MAX_WORKERS           = 1
NR_RUNS               = 1
ONLY_SUCCESSFUL       = True
MAX_RETRY_MULTIPLIER  = 3

# ── Manifest — one GPTDirect run over real QBPs ─────────────────────────────
MANIFESTS = {
    'GPTDirect': {
        "framework":            "GPTDirect",
        "runs":                 NR_RUNS,
        "name":                 "Run",
        "model":                "gpt-5.6-low",
        "mapping_model":        "gpt-5.6-low",
        "use_dedup":            False,
        "only_successful":      ONLY_SUCCESSFUL,
        "max_retry_multiplier": MAX_RETRY_MULTIPLIER,
        "gpt_direct_retry_mode": "repair"
    },
}

INSTRUCT_TMPL_CONFIG = {        
        "{syntax rules}": os.path.join('data', 'prmt', "rules", "bpmn syntax.txt"),
        "{general modelling rules}": os.path.join('data', 'prmt', "rules", "general modelling.txt"),
        "{cpw modelling rules}": os.path.join('data', 'prmt', "rules", "cpw modelling.txt"),
        "{knowledge statements}": lambda cpw: os.path.join('data', 'prmt', "knowledge", f"{cpw}.txt")
}

INPUT_TMPL_CONFIG = {
    "{cpw text}":  lambda cpw: os.path.join('data', "cpw", f"{cpw}.txt")
}


def main(sys_config, cpw, prompt_config, verbose=False, test=None, runs=NR_RUNS):
    system = sys_config['system']; model = sys_config['model']
    id = f"{cpw}-{system}_{model.replace("-", "_")}".lower()
    if prompt_config['incl_model_guide']:
        id += "-mg"
    if prompt_config['incl_knowl']:
        id += "-ks"

    # root                             = "out"
    root                             = os.path.join("out")
    exp_dir                          = os.path.join(root, id)
    if os.path.exists(exp_dir):
        pass
        # print(f"experiment directory already exists: {exp_dir}")
        # return 1
    else:
        os.mkdir(exp_dir)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY before running batch_runner.py")

    env = os.environ.copy()
    env["OPENAI_API_KEY"]           = api_key

    # - given the CPW & LLM, setup parameters
    llm_man = MANIFESTS[system]
    cur_man = llm_man.copy()
    cur_man['id']                   = id
    cur_man['qbp']                  = cpw
    cur_man["model"]                = model
    cur_man["runs"]                 = runs

    # instantiate prompts
    if prompt_config['incl_model_guide']:
        if prompt_config['incl_knowl']:
            instruct_tmpl = 'instruct_tmpl-2'
        else:
            instruct_tmpl = 'instruct_tmpl-1'
    else:
        instruct_tmpl = 'instruct_tmpl-0'
    instruct_tmpl = os.path.join('data', 'prmt', f"{instruct_tmpl}.txt")
    input_tmpl = os.path.join('data', 'prmt', "input_tmpl.txt")

    # resolve per-CPW paths; pipeline.py instantiates these into each run folder
    def _resolve(config):
        return {term: (path(cpw) if callable(path) else path) for term, path in config.items()}

    task_templates = [
        { "tmpl": instruct_tmpl, "subst": _resolve(INSTRUCT_TMPL_CONFIG), "out": "instruct.txt" },
        { "tmpl": input_tmpl,    "subst": _resolve(INPUT_TMPL_CONFIG),    "out": "input.txt" },
    ]

    # set reference BPMN
    cur_man["task_templates"]       = task_templates
    cur_man["gold_bpmn_filename"]   = f"{cpw}.bpmn"
    cur_man["gold_bpmn"]            = os.path.join("data", "ref", cur_man["gold_bpmn_filename"])
    cur_man["ltl_cnstr_filename"]   = f"{cpw}.json"
    cur_man["ltl_cnstr"]            = os.path.join("data", "prmt", "knowledge", "cnstrs", cur_man["ltl_cnstr_filename"])
    cur_man["cnstr_ontology"]       = os.path.join("data", "prmt", "knowledge", "cnstrs", "constraints.owl")

    # setup test
    if test is not None:
        cur_man["test"]             = os.path.join("data", "test", f"{test}.bpmn")

    # experiment-specific files
    manifest_path                   = os.path.join(exp_dir, "batch_manifest.json")
    #results_db                      = os.path.join(exp_dir, f"{cpw}.sqlite")
    results_db                      = os.path.join(root, f"results_db.sqlite")

    # print(cur_man)

    # store manifest
    cur_man = [ cur_man ] # TODO
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(cur_man, f, indent=2)

    cmd = [
        "python", "-u", "pipeline.py",
        "--batch-manifest",       manifest_path,
        "--max-workers",          str(MAX_WORKERS),
        "--results-db",           results_db,
        "--output-dir",           exp_dir,
        "--max-retry-multiplier", str(MAX_RETRY_MULTIPLIER),
        "--verbose",              str(verbose)
    ]
    if prompt_config['incl_model_guide']:
        cmd.append("--incl-model-guide")
    if prompt_config['incl_knowl']:
        cmd.append("--incl-knowl")
    if ONLY_SUCCESSFUL:
        cmd.append("--only-successful")

    print(f"Launching: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    batch_id = None
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        print(line)

        m = re.match(r"BATCH_ID (\w+)", line)
        if m:
            batch_id = m.group(1)

        m = re.search(r"PROGRESS (\d+)/(\d+)", line)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            print(f"  -> progress: {x}/{y}")

    proc.wait()

    if proc.returncode not in (0, None):
        print(f"Pipeline exited with code {proc.returncode}.")
    if batch_id:
        print(f"Batch complete. batch_id={batch_id}  db={results_db}")

    return proc.returncode or 0

def parse_args():
    p = argparse.ArgumentParser(description="Run a batch pipeline job for a given CPW.")
    p.add_argument("-c", "--cpw", help="CPW identifier (matches data/cpw/<cpw>.txt), or a comma-separated list of them")
    p.add_argument("-mg", "--incl_model_guide", type=bool, nargs='?', const=True, default=False, help="Include modelling guidelines")
    p.add_argument("-ks", "--incl_knowl", type=bool, nargs='?', const=True, default=False, help="Include knowledge statements")
    p.add_argument("-t", "--test", help="Test identifier (matches data/test/<test>.bpmn); use _mapped suffix to avoid mapping step as well", required=False, default=None)
    p.add_argument("-s", "--system", help=f"System identifier, i.e., one of {','.join(MANIFESTS.keys())}")
    p.add_argument("-m", "--model", help=f"Foundational model suffixed by effort (e.g., gpt-5.4-medium)")
    p.add_argument("-r", "--runs", type=int, default=NR_RUNS, help=f"Number of successful runs per CPW (default {NR_RUNS})")
    p.add_argument("-v", "--verbose", type=bool, nargs='?', const=True, default=False, help="Print detailed step stdouts (pipeline)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys_config = { 'system': args.system, 'model': args.model }
    prompt_config = { 'incl_model_guide': args.incl_model_guide, 'incl_knowl': args.incl_knowl }

    cpws = [cpw.strip() for cpw in args.cpw.split(",") if cpw.strip()]
    exit_code = 0
    for cpw in cpws:
        print(f"\n\n=== CPW: {cpw} ===")
        exit_code = main(sys_config, cpw, prompt_config, args.verbose, args.test, args.runs) or exit_code
    sys.exit(exit_code)
