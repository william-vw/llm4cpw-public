#!/usr/bin/env python3
# pipeline.py
"""
Pipeline for the GPTDirect framework.

Key features:
  - Batch mode via --batch-manifest (JSON list of experiment items)
  - Parallel execution via --max-workers (dynamic retry-aware)
  - Retry logic: --only-successful retries failed runs until target successes reached
  - Incremental DB writes (crash-safe WAL mode)
  - Progress lines: PROGRESS X/Y [attempts=Z]

Manifest item format:
{
  "id": "a1b2c3d4",          # experiment id
  "framework": "GPTDirect",
  "runs": 5,                 # target runs (successful ones if only_successful)
  "name": "PipelineRun",
  "model": "GPT_5o2",
  "mapping_model": "gpt-5.2",
  "only_successful": false,  #  if true, keep retrying failed attempts until N successful runs are accumulated
  "max_retry_multiplier": 3.0, # (only relevant if only_successful=true) e.g. if runs=5, max_retry_multiplier=3.0; up to 15 attempts total before giving up
  "cpw": "CPW1",             # process name (matches the reference model)
  "task_templates": [        # instantiated into the run folder; must produce exactly
    {"tmpl": "...", "subst": {...}, "out": "instruct.txt"},   # [instructions, input]
    {"tmpl": "...", "subst": {...}, "out": "input.txt"}       # (gpt_direct_responses_api.py
  ],                         # has no built-in system prompt and requires both files)
  "gold_bpmn": "/tmp/id__CPW1.bpmn" | null,
  "gold_bpmn_filename": "CPW1.bpmn" | null
}
"""

import argparse
import copy
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import dataclasses
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from queue import Queue
from threading import Thread
from libs.bpmn_to import bpmn_to_svg
sys.path.insert(0, "..")
from libs.bpmn_utils import add_incoming_outgoing, add_dummy_di
from libs.template import instantiate_files
from libs.cmp4proc.ltl.lola.ltl_check import check_constraints as check_ltl_cnstrs
from libs.cmp4proc.activ_cnstr.activ_cnstr_check import check_constraints as check_activ_cnstrs
from rdflib import Graph, Literal, Namespace, RDF, RDFS

from libs.automatedActivityMapping import align
from libs.cmp4proc.bpmn_comparison.struct_simil import calculate_similarity as calc_struct_simil
from libs.cmp4proc.bpmn_highlighter import render_diff



# ════════════════════════════════════════════════════════════════════════════
# Args
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BPMN generation + similarity pipeline (batch mode)."
    )
    # Batch
    p.add_argument("--batch-manifest", default=None,
                   help="JSON list of experiment items.")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Parallel worker threads.")
    p.add_argument("--batch-id", default=None,
                   help="Optional batch grouping id; auto-generated if omitted.")
    p.add_argument("--results-db", default="resultsDb.sqlite",
                   help="SQLite database path.")
    p.add_argument("--output-dir", default="output",
                   help="Directory for output files.")

    # Retry / reliability
    p.add_argument("--only-successful", action="store_true",
                   help="Retry failed runs until target successes are reached.")
    p.add_argument("--max-retry-multiplier", type=float, default=3.0,
                   help="max_attempts = target_runs * multiplier (default 3.0).")
    p.add_argument("--verbose", type=bool, default=False, help="Print detailed step stdouts.")
    p.add_argument("--incl-model-guide", action="store_true",
                   help="Modelling guidelines were included in the prompt (recorded in the db).")
    p.add_argument("--incl-knowl", action="store_true",
                   help="Knowledge statements were included in the prompt (recorded in the db).")

    return p.parse_args()



# ════════════════════════════════════════════════════════════════════════════
# Overall constraint score (LTL + other constraints together)
# ════════════════════════════════════════════════════════════════════════════

_CNSTR_NS = Namespace("http://notation3.org/constraints.owl#")


def load_constraint_map(ontology_path: str, model_name: str):
    """Return (total, {label: class name}) for the constraints of `model_name` in the ontology.
    The class name is the constraint's rdf:type in the constraints namespace, without namespace."""
    g = Graph()
    g.parse(str(ontology_path), format="turtle")

    candidates = list(g.subjects(RDFS.label, Literal(model_name))) + [_CNSTR_NS[model_name.replace(" ", "_")]]
    model = next((m for m in candidates if (m, RDF.type, _CNSTR_NS.ProcessModel) in g), None)
    if model is None:
        raise KeyError(f"No process model named {model_name!r} in the ontology")

    constraints = set(g.objects(model, _CNSTR_NS.hasConstraint))
    label_to_class = {}
    for c in constraints:
        label = g.value(c, RDFS.label)
        label = str(label) if label is not None else str(c).rsplit("#", 1)[-1]
        classes = [str(t).rsplit("#", 1)[-1] for t in g.objects(c, RDF.type)
                   if str(t).startswith(str(_CNSTR_NS)) and t != _CNSTR_NS.ProcessModel]
        label_to_class[label] = classes[0] # if classes else None
    return label_to_class

# ════════════════════════════════════════════════════════════════════════════
# Database helpers
# ════════════════════════════════════════════════════════════════════════════

def db_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.row_factory = sqlite3.Row
    return conn


def db_init(conn: sqlite3.Connection):
    c = conn.cursor()

    # ── runs: one row per generation attempt ─────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id            TEXT    NOT NULL,
        job_id              TEXT    NOT NULL,
        timestamp           TEXT,
        cpw_name            TEXT,
        framework           TEXT,
        model               TEXT,
        attempt_number      INTEGER,
        status              TEXT,           -- ok | ok_no_gold | failed_generation |
                                            -- failed_mapping | failed_similarity | failed_error
        error               TEXT,
        total_seconds       REAL,
        gen_seconds         REAL,
        map_seconds         REAL,
        fix_seconds         REAL,
        layout_seconds      REAL,
        struct_sim_seconds  REAL,
        render_diff_seconds REAL,
        ltl_seconds         REAL,
        cnstr_seconds       REAL,
        svg_seconds         REAL,

        -- Artefact paths
        gold_bpmn           TEXT,
        ltl_cnstr           TEXT,
        cnstr_ontology      TEXT,
        generated_bpmn      TEXT,
        mapped_bpmn         TEXT,

        -- Similarity (always computed when gold BPMN provided)
        node_sim            REAL,
        struct_sim          REAL,
        ged                 INTEGER,
        ltl_score           REAL,
        ltl_conf            TEXT,
        cnstr_score         REAL,
        cnstr_conf          TEXT,
        cpw_succ_cnstrs INTEGER,        -- constraints (ltl + cnstr) that hold
        cpw_total_cnstrs      INTEGER,        -- constraints of the process model in the ontology
        cpw_score       REAL,           -- cpw_succ_cnstrs / cpw_total_cnstrs

        -- Prompt configuration
        incl_model_guide    INTEGER,        -- 1 if modelling guidelines were in the prompt
        incl_knowl          INTEGER,        -- 1 if knowledge statements were in the prompt
        optim_prmt          INTEGER DEFAULT 0,

        UNIQUE(batch_id, job_id)
    )""")

    # ── batches: one row per pipeline invocation ─────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS batches (
        batch_id        TEXT PRIMARY KEY,
        started_at      TEXT,
        finished_at     TEXT,
        status          TEXT,
        total_target    INTEGER,
        total_attempts  INTEGER,
        total_successes INTEGER,
        results_db      TEXT
    )""")

    # migrate existing databases
    existing = {row[1] for row in c.execute("PRAGMA table_info(runs)")}
    for col, typ in (("cpw_succ_cnstrs", "INTEGER"), ("cpw_total_cnstrs", "INTEGER"), ("cpw_score", "REAL")):
        if col not in existing:
            c.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")

    conn.commit()


def db_upsert_run(conn: sqlite3.Connection, r: dict):
    conn.execute("""
    INSERT OR REPLACE INTO runs (
        batch_id, job_id, timestamp,
        cpw_name, framework, model, attempt_number,
        status, error, total_seconds, gen_seconds, map_seconds,
        fix_seconds, layout_seconds, struct_sim_seconds, render_diff_seconds,
        ltl_seconds, cnstr_seconds, svg_seconds,
        gold_bpmn, ltl_cnstr, cnstr_ontology, generated_bpmn, mapped_bpmn,
        node_sim, struct_sim, ged, ltl_score, ltl_conf,
        cnstr_score, cnstr_conf,
        cpw_succ_cnstrs, cpw_total_cnstrs, cpw_score,
        incl_model_guide, incl_knowl, optim_prmt
    ) VALUES (
        :batch_id, :job_id, datetime('now'),
        :cpw_name, :framework, :model, :attempt_number,
        :status, :error, :total_seconds, :gen_seconds, :map_seconds,
        :fix_seconds, :layout_seconds, :struct_sim_seconds, :render_diff_seconds,
        :ltl_seconds, :cnstr_seconds, :svg_seconds,
        :gold_bpmn, :ltl_cnstr, :cnstr_ontology, :generated_bpmn, :mapped_bpmn,
        :node_sim, :struct_sim, :ged, :ltl_score, :ltl_conf,
        :cnstr_score, :cnstr_conf,
        :cpw_succ_cnstrs, :cpw_total_cnstrs, :cpw_score,
        :incl_model_guide, :incl_knowl, :optim_prmt
    )""", r)
    conn.commit()


def db_batch_start(conn: sqlite3.Connection, batch_id: str, total_target: int):
    conn.execute("""
    INSERT OR IGNORE INTO batches (batch_id, started_at, status, total_target)
    VALUES (?, datetime('now'), 'running', ?)
    """, (batch_id, total_target))
    conn.commit()


def db_batch_finish(conn: sqlite3.Connection, batch_id: str,
                    total_attempts: int, total_successes: int):
    conn.execute("""
    UPDATE batches
    SET status='finished', finished_at=datetime('now'),
        total_attempts=?, total_successes=?
    WHERE batch_id=?
    """, (total_attempts, total_successes, batch_id))
    conn.commit()


# ════════════════════════════════════════════════════════════════════════════
# Misc helpers
# ════════════════════════════════════════════════════════════════════════════

def _safe_makedirs(path: str):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Framework runners
# ════════════════════════════════════════════════════════════════════════════

def _run_streaming(cmd, cwd=None, timeout=None, verbose=False) -> str:
    """
    Run `cmd`, printing stdout/stderr live as it arrives (when verbose),
    while also returning the full captured output for parsing.
    Raises subprocess.CalledProcessError / TimeoutExpired like subprocess.run.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if verbose:
        print(">> STDOUT:")

    lines = []
    for line in proc.stdout:
        lines.append(line)
        if verbose:
            print(line, end="")

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise

    output = "".join(lines)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=output)
    return output


def run_gpt_direct(task_file, model: str, code_root: str,
                   run_dir: str, verbose, retry_mode: str = "repair",
                   timeout: int = None):
    """
    Run the GPTDirect framework — calls gpt_direct_responses_api.py as a
    subprocess. `task_file` must be a 2-element [instructions_file,
    input_file] list; gpt_direct_responses_api.py has no built-in system
    prompt and always requires both files.
    retry_mode ∈ {"none", "repair", "full"}.
    Returns (bpmn_path: str, retries_summary: str).

    timeout is intentionally None by default: gpt_direct_responses_api.py
    applies its own per-call API timeout (1 hour) which is the correct
    governing limit. Reasoning models (gpt-5.4 xhigh) can legitimately take
    20–60 min per call, so a subprocess-level timeout would fire before the
    model finishes.
    """
    script = os.path.join(code_root, "gpt_direct_responses_api.py")
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"gpt_direct_responses_api.py not found at {script}. "
            "Place gpt_direct_responses_api.py in the repo root."
        )

    if not isinstance(task_file, (list, tuple)) or len(task_file) != 2:
        raise ValueError(
            "GPTDirect task_file must be a 2-element "
            f"[instructions_file, input_file] list, got {task_file!r}."
        )
    instructions_file, input_file = task_file

    cmd = [
        sys.executable, "-u", script,
        "--instructions-file", instructions_file,
        "--input-file",        input_file,
        "--model",             model,
        "--run-dir",           run_dir,
        "--retry-mode",        retry_mode,
    ]

    output = _run_streaming(cmd, timeout=timeout, verbose=verbose)

    gpt_retries = None
    bpmn_path   = None
    for line in output.strip().splitlines():
        if line.startswith("GPT_DIRECT_RETRIES:"):
            gpt_retries = line.split("GPT_DIRECT_RETRIES:", 1)[1].strip()
        elif line.endswith(".bpmn"):
            bpmn_path = line.strip()

    return bpmn_path, gpt_retries


# ════════════════════════════════════════════════════════════════════════════
# Job building
# ════════════════════════════════════════════════════════════════════════════

def _make_job(eid, cpw, framework, run_idx, task_templates, gold_bpmn,
              gold_bpmn_filename, ltl_cnstr, ltl_cnstr_filename, 
              cnstr_ontology, test, name, model, mapping_model,
              only_successful, max_retry_multiplier,
              target_runs, output_dir,
              verbose, gpt_direct_retry_mode="repair",
              incl_model_guide=False, incl_knowl=False) -> dict:
    return {
        "job_id":                   f"{eid}:{cpw}:attempt{run_idx}",
        "id":                       eid,
        "cpw":                      cpw,
        "framework":                framework,
        "run_idx":                  run_idx,
        "task_templates":           task_templates,
        "gold_bpmn":                gold_bpmn,
        "gold_bpmn_filename":       gold_bpmn_filename,
        "ltl_cnstr":                ltl_cnstr,
        "ltl_cnstr_filename":       ltl_cnstr_filename,
        "cnstr_ontology":           cnstr_ontology,
        "test":                     test,
        "name":                     name,
        "model":                    model,
        "mapping_model":            mapping_model,
        "only_successful":          only_successful,
        "max_retry_multiplier":     max_retry_multiplier,
        "target_runs":              target_runs,
        "output_dir":               output_dir,
        "verbose":                  verbose,
        "gpt_direct_retry_mode":    gpt_direct_retry_mode,
        "incl_model_guide":         incl_model_guide,
        "incl_knowl":               incl_knowl,
    }


def make_retry_job(original_job: dict, attempt_idx: int) -> dict:
    j = copy.deepcopy(original_job)
    j["run_idx"] = attempt_idx
    j["job_id"]  = f"{original_job['id']}:{original_job['cpw']}:attempt{attempt_idx}"
    return j


def load_jobs(args) -> list:
    jobs: list = []

    output_dir = args.output_dir
    if args.batch_manifest:
        with open(args.batch_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list):
            raise ValueError("--batch-manifest must be a JSON array.")

        for item in manifest:
            eid       = item.get("id") or "noid"
            framework = item.get("framework", "")
            if framework != "GPTDirect":
                raise ValueError(f"Invalid framework in manifest: {framework!r}")

            task_templates = item.get("task_templates")
            if not task_templates:
                continue

            target_runs = int(item.get("runs", 1))
            cpw = item["cpw"]

            # Per-manifest item overrides; fall back to global args
            only_succ = item.get("only_successful", args.only_successful)
            mult      = item.get("max_retry_multiplier", args.max_retry_multiplier)

            for run_idx in range(1, target_runs + 1):
                jobs.append(_make_job(
                    eid=eid, cpw=cpw, framework=framework,
                    run_idx=run_idx, task_templates=task_templates,
                    gold_bpmn=item.get("gold_bpmn"),
                    gold_bpmn_filename=item.get("gold_bpmn_filename"),
                    ltl_cnstr=item.get("ltl_cnstr"),
                    ltl_cnstr_filename=item.get("ltl_cnstr_filename"),
                    cnstr_ontology=item.get("cnstr_ontology"),
                    test=item.get("test", None),
                    name=item.get("name", "PipelineProject"),
                    model=item.get("model", None),
                    mapping_model=item.get("mapping_model", None),
                    only_successful=only_succ,
                    max_retry_multiplier=mult,
                    target_runs=target_runs,
                    output_dir=output_dir,
                    verbose=args.verbose,
                    incl_model_guide=item.get("incl_model_guide", args.incl_model_guide),
                    incl_knowl=item.get("incl_knowl", args.incl_knowl),
                    gpt_direct_retry_mode=item.get("gpt_direct_retry_mode", "repair")
                ))
        return jobs

    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Execute one job
# Returns a result dict; does NOT touch the DB.
# ════════════════════════════════════════════════════════════════════════════

def run_one_job(job: dict) -> dict:
    t0          = time.time()
    eid         = job["id"]
    cpw         = job["cpw"]
    run_idx     = job["run_idx"]
    framework   = job["framework"]
    job_id      = job["job_id"]

    def jprint(msg: str):
        print(f"[{job_id}] {msg}", flush=True)

    name_prefix     = job.get("name")          or "PipelineProject"
    model           = job.get("model")
    mapping_model   = job.get("mapping_model")
    task_templates  = job.get("task_templates")
    gold_bpmn       = job.get("gold_bpmn")
    gold_bpmn_filename    = job.get("gold_bpmn_filename")
    ltl_cnstr       = job.get("ltl_cnstr")
    ltl_cnstr_filename       = job.get("ltl_cnstr_filename")
    cnstr_ontology  = job.get("cnstr_ontology")
    test            = job.get("test")
    output_dir      = job.get("output_dir")
    verbose         = job.get("verbose")
    incl_model_guide = int(bool(job.get("incl_model_guide")))
    incl_knowl      = int(bool(job.get("incl_knowl")))

    db_model  = model
    run_name  = f"{name_prefix}-{eid}-attempt{run_idx}"

    # Run folder: <output_dir>/<run_name>-<timestamp> (relative output_dir is
    # resolved against this file's directory).
    ts       = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    os.makedirs(out_root, exist_ok=True)
    run_dir  = os.path.join(out_root, f"{run_name}-{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Instantiate task text files (e.g., instructions + input) into the run folder
    task_file = []
    for t in task_templates:
        out = os.path.join(run_dir, t["out"])
        with open(out, "w") as fh:
            fh.write(instantiate_files(t["tmpl"], t["subst"], None))
        task_file.append(out)

    jprint(f"=== Attempt {run_idx} for '{cpw}' [{framework} {model} (map: {mapping_model})] ===")

    gen_bpmn        = None
    fix_seconds         = -1
    layout_seconds      = -1
    struct_sim_seconds  = -1
    render_diff_seconds = -1
    ltl_seconds         = -1
    cnstr_seconds       = -1
    svg_seconds         = -1
    cpw_succ_cnstrs = None
    cpw_total_cnstrs      = None
    cpw_score       = None

    # ── Scaffold for early-exit returns ─────────────────────────────────────
    def _result(status: str, error: str = None) -> dict:
        return {
            "job_id":           job_id,
            "cpw_name":         cpw,
            "framework":        framework,
            "model":            db_model,
            "attempt_number":   run_idx,
            "status":           status,
            "error":            error,
            "total_seconds":    round(time.time() - t0, 2),
            "gen_seconds":      -1,
            "map_seconds":      -1,
            "fix_seconds":      fix_seconds,
            "layout_seconds":   layout_seconds,
            "struct_sim_seconds": struct_sim_seconds,
            "render_diff_seconds": render_diff_seconds,
            "ltl_seconds":      ltl_seconds,
            "cnstr_seconds":    cnstr_seconds,
            "svg_seconds":      svg_seconds,
            "gold_bpmn":        gold_bpmn_filename or (os.path.basename(gold_bpmn) if gold_bpmn else None),
            "ltl_cnstr":        ltl_cnstr_filename,
            "cnstr_ontology":   cnstr_ontology,
            "generated_bpmn":   gen_bpmn,
            "mapped_bpmn":      None,
            "node_sim":         None,
            "struct_sim":       None,
            "ged":              None,
            "ltl_score":        None,
            "ltl_conf":         None,
            "cnstr_score":      None,
            "cnstr_conf":       None,
            "cpw_succ_cnstrs": None,
            "cpw_total_cnstrs":      None,
            "cpw_score":        None,
            "incl_model_guide": incl_model_guide,
            "incl_knowl":       incl_knowl,
            "optim_prmt":       0,
        }

    # ── Step 1: Generate BPMN ─────────────────────────────────────────────
    jprint("── Step 1: Generate BPMN ─────────────────────────────────────────────")
    t0_gen = time.time()
    if test is not None:
        if not os.path.isfile(test):
            jprint(f"✗ Test BPMN not found: {test}")
            return _result("failed_generation", f"test_bpmn_not_found: {test}")
        # work on a copy in the run folder so that derived files (pnml, lola, svg) land there
        gen_bpmn = os.path.join(run_dir, os.path.basename(test))
        shutil.copyfile(test, gen_bpmn)
        jprint(f"✔ Loaded test: {test}")
    else:
        try:
            retry_mode = job.get("gpt_direct_retry_mode", "repair")
            code_root  = os.path.dirname(os.path.abspath(__file__))
            gen_bpmn, _ = run_gpt_direct(
                task_file, model, code_root, run_dir, verbose, retry_mode=retry_mode
            )
    
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            jprint(f"✗ Generation subprocess failed:\n{stderr or '(no stderr)'}")
            return _result("failed_generation", f"subprocess_error: {stderr}")
        except subprocess.TimeoutExpired:
            jprint("✗ Generation timed out.")
            return _result("failed_generation", "timeout")
        except Exception as e:
            jprint(f"✗ Generation error: {e}")
            return _result("failed_generation", str(e))

        if not gen_bpmn or not os.path.exists(gen_bpmn):
            jprint("✗ Generated BPMN path is missing or does not exist.")
            return _result("failed_generation", "bpmn_file_not_found")

        jprint(f"✔ Generated: {gen_bpmn}")

    gen_seconds = round(time.time() - t0_gen, 2)

    # Determine framework label for DB
    retry_mode   = job.get("gpt_direct_retry_mode", "repair")
    framework_db = f"GPT-Direct-{retry_mode}"

    # No gold BPMN → log generation only, no similarity
    if not gold_bpmn:
        return _result("ok_no_gold")

    map_seconds = -1
    if test is not None:
        # test mode: the given BPMN is checked as-is (no mapping, fixing or layout)
        mapped_bpmn = gen_bpmn
        jprint("✔ Test mode: skipping mapping, BPMN fixing and layout")
    else:
        # ── Step 2: Activity mapping ──────────────────────────────────────────
        jprint("── Step 2: Activity mapping ──────────────────────────────────────────")
        t0_map = time.time()
        mapped_bpmn = None
        try:
            mapped = gen_bpmn.replace(".bpmn", "_mapped.bpmn")
            _safe_makedirs(mapped)
            align(gen_bpmn, gold_bpmn, mapped, model=mapping_model, verbose=False, print_map=True)
            if not os.path.exists(mapped):
                raise FileNotFoundError(f"Mapped BPMN not created at: {mapped}")
            mapped_bpmn = mapped
            jprint(f"✔ Mapped:    {mapped}")
        except Exception as e:
            jprint(f"✗ Activity mapping failed: {e}")
            return _result("failed_mapping", str(e))

        map_seconds = round(time.time() - t0_map, 2)

        # -- Step 3: fixing BPMN XML --
        jprint("── Step 3: Fix BPMN XML ─────────────────")
        # downstream code requires gateway elements to have incoming and outgoing elements
        # we don't ask the LLM since this is technically duplication and we don't want to add a potential source of errors
        jprint("- Fixing gateways")

        try:
            t0_fix = time.time()
            mapped_fixed_bpmn = f"{mapped_bpmn[:-5]}_fixed.bpmn"
            add_incoming_outgoing(mapped_bpmn, mapped_fixed_bpmn)
            fix_seconds = round(time.time() - t0_fix, 2)
        except Exception as e:
            jprint(f"✗ Fixing BPMN XML failed: {e}")
            return _result("failed_fix_bmpn", str(e))

        # add layout to BPMN files
        jprint("- Adding layout")
        mapped_layout_bpmn = f"{mapped_fixed_bpmn[:-5]}_layout.bpmn"
        try:
            t0_layout = time.time()
            add_dummy_di(mapped_fixed_bpmn, mapped_layout_bpmn)
            layout_seconds = round(time.time() - t0_layout, 2)
            jprint(f"Layout saved to {mapped_layout_bpmn}")
        except Exception as e:
            jprint(f"✗ Adding layout failed: {e}")
            return _result("failed_add_layout", str(e))

        mapped_bpmn = mapped_layout_bpmn

    # ── Step 4: Similarity ─────────────────────────────────────
    jprint("── Step 4: Similarity ─────────────────────────────────────")
    # With knowledge constraints in the prompt (incl_knowl, set by batch_runner), 
    # compare against the reference model with reified activities (<name>_reif.bpmn), if any
    has_cnstrs = bool(incl_knowl)
    sim_ref = gold_bpmn
    if has_cnstrs:
        reif_bpmn = os.path.splitext(gold_bpmn)[0] + "_reif.bpmn"
        if os.path.exists(reif_bpmn):
            sim_ref = reif_bpmn
            jprint(f"→ Using reified reference model: {reif_bpmn}")
    t0_struct = time.time()
    (node_sim, struct_sim, ged, diff), _ = calc_struct_simil(sim_ref, mapped_bpmn)[0]
    struct_sim_seconds = round(time.time() - t0_struct, 2)
    jprint(f"→ Structural similarity: {struct_sim}")
    out_folder = os.path.dirname(mapped_bpmn)
    try:
        t0_render = time.time()
        render_diff(sim_ref, mapped_bpmn, diff, backend='pm4py', show_output=False, 
                    diag1_out=os.path.join(out_folder, "cmp_ref.svg"), 
                    diag2_out=os.path.join(out_folder, "cmp_out.svg"))
        render_diff_seconds = round(time.time() - t0_render, 2)
    except Exception as e:
        jprint(f"✗ Calculating similarity failed: {e}")
        return _result("failed_calc_simil", str(e))

    # ── Step 5: Constraint conformance ─────────────────────────────────

    jprint("── Step 5: Constraint conformance ─────────────────────────────────────")

    # get constraint category from label
    def _label_cat(name): # in case of underscore, drop the part afterwards
        stripped = name.rsplit("_", 1)[0] if "_" in name else name
        return stripped # if cpw_cnstrs.get(stripped) == "PrioritizedConditional" else name

    # LTL constraints
    try:
        t0_ltl = time.time()
        ltl_res = check_ltl_cnstrs(mapped_bpmn, ltl_cnstr)
        # individual constraints to be ignored 
        ignore_indiv = set(); ignore_cat = set() # keeps empty if LLM constraints are given
        if not has_cnstrs: # (only if constraints are not given to LLM)
            # ignore these (prioritized condition) constraints, since they rely on giving the constraints to LLM
            ignore_indiv = {"cap6_1", "cap6_2", "copd3_1", "copd3_2"}
            ignore_cat = {_label_cat(n) for n in ignore_indiv}
            ltl_res.update_results([r for r in ltl_res.results if r.name not in ignore_indiv])
        # this is the fine-grained score; not per constraint category, but per individual constraint
        ltl_score = ltl_res.score(); ltl_conf = ltl_res.summary()
        ltl_seconds = round(time.time() - t0_ltl, 2)
        jprint(f"LTL time: {ltl_seconds}s")
        jprint(f"→ LTL score: {ltl_score}")
        jprint(f"→ LTL conformance: {ltl_conf}")
    except Exception as e:
        jprint(f"✗ LTL constraint check failed: {e}")
        return _result("failed_ltl_check", str(e))

    # other constraints
    try:
        if not cnstr_ontology:
            raise ValueError("no constraints ontology given (cnstr_ontology)")
        t0_cnstr = time.time()
        cnstr_res = check_activ_cnstrs(cnstr_ontology, mapped_bpmn, cpw)
        # this is the fine-grained score; not per constraint category, but per individual constraint
        cnstr_score = cnstr_res.score(); cnstr_conf = cnstr_res.summary()
        cnstr_seconds = round(time.time() - t0_cnstr, 2)
        jprint(f"→ Constraints (other) score: {cnstr_score}")
        jprint(f"→ Constraints (other) conformance: {cnstr_conf}")
    except Exception as e:
        jprint(f"✗ Constraint check (other) failed: {e}")
        return _result("failed_cnstr_check", str(e))

    # ── Step 5b: Overall constraint score (ltl + cnstr) ────────────────
    jprint("── Step 5b: Overall constraint score ─────────────────────────────────")
    try:
        # returns constraint categories for this CPW
        cpw_cnstrs = load_constraint_map(cnstr_ontology, cpw)
        # e.g. for CAP: {"cap1": "DecisionalDependency",
        # "cap2": "DecisionalDependency", "cap3": "DecisionalDependency", "cap4": "Conflict",
        # "cap5": "EntryPoint", "cap6": "PrioritizedConditional"}

         # get constraint categories to be ignored for this CPW
        ignore_cpw = { n for n in ignore_cat if n in cpw_cnstrs }
        # leave ignored's out of total constraints
        cpw_total_cnstrs = len(cpw_cnstrs) - len(ignore_cpw)

        # get (unique) failed constraint categories for this CPW
        # (set of categories; if an individual of a category failed, entire category failed)
        failed_cat = {_label_cat(r.name) for res in (ltl_res, cnstr_res) for r in res.filter_results(False)}
        # (assume that failed_cat is in cpw_cnstrs)
        cpw_succ_cnstrs = cpw_total_cnstrs - len(failed_cat)
        cpw_score = cpw_succ_cnstrs / cpw_total_cnstrs if cpw_total_cnstrs else 1.0
        jprint(f"→ Overall: {cpw_succ_cnstrs}/{cpw_total_cnstrs} constraints hold (score {cpw_score})")
    except Exception as e:
        jprint(f"✗ Overall constraint score failed: {e}")
        return _result("failed_cpw_score", str(e))

    # ── Step 6: Convert to SVG ──────────────────────────────────
    jprint("── Step 6: Convert BPMN to SVG ─────────────────")
    if test is not None:
        jprint("✔ Test mode: skipping SVG conversion")
    else:
        mapped_bpmn_svg = f"{mapped_bpmn[:-5]}.svg"
        try:
            t0_svg = time.time()
            bpmn_to_svg(mapped_bpmn, mapped_bpmn_svg)
            svg_seconds = round(time.time() - t0_svg, 2)
        except Exception as e:
            jprint(f"✗ Conversion to SVG failed: {e}")
            return _result("failed_conv_svg", str(e))

    # ── All done ─────────────────────────────────────────────────────────
    return {
        "job_id":           job_id,
        "cpw_name":         cpw,
        "framework":        framework_db,
        "model":            db_model,
        "attempt_number":   run_idx,
        "status":           "ok",
        "error":            None,
        "total_seconds":    round(time.time() - t0, 2),
        "gen_seconds":      gen_seconds,
        "map_seconds":      map_seconds,
        "fix_seconds":      fix_seconds,
        "layout_seconds":   layout_seconds,
        "struct_sim_seconds": struct_sim_seconds,
        "render_diff_seconds": render_diff_seconds,
        "ltl_seconds":      ltl_seconds,
        "cnstr_seconds":    cnstr_seconds,
        "svg_seconds":      svg_seconds,
        "gold_bpmn":        gold_bpmn_filename or os.path.basename(gold_bpmn),
        "ltl_cnstr":        ltl_cnstr_filename,
        "cnstr_ontology":   cnstr_ontology,
        "generated_bpmn":   gen_bpmn,
        "mapped_bpmn":      mapped_bpmn,
        "node_sim":         node_sim,
        "struct_sim":       struct_sim,
        "ged":              ged,
        "ltl_score":        ltl_score,
        "ltl_conf":         ltl_conf,
        "cnstr_score":      cnstr_score,
        "cnstr_conf":       cnstr_conf,
        "cpw_succ_cnstrs": cpw_succ_cnstrs,
        "cpw_total_cnstrs":   cpw_total_cnstrs,
        "cpw_score":    cpw_score,
        "incl_model_guide": incl_model_guide,
        "incl_knowl":       incl_knowl,
        "optim_prmt":       0,
    }


# ════════════════════════════════════════════════════════════════════════════
# DB writer thread (single serialised writer — crash-safe)
# ════════════════════════════════════════════════════════════════════════════

def db_writer_loop(db_path: str, batch_id: str, total_target: int, q: Queue):
    conn = db_connect(db_path)
    try:
        db_init(conn)
        db_batch_start(conn, batch_id, total_target)
        attempts  = 0
        successes = 0
        while True:
            item = q.get()
            if item is None:
                break
            r = item.get("result")
            if not r:
                continue
            r = dict(r)
            r["batch_id"] = batch_id
            try:
                db_upsert_run(conn, r)
                attempts += 1
                if r.get("status") == "ok":
                    successes += 1
            except Exception as e:
                print(f"[DB] Write error: {e}", flush=True)
    finally:
        try:
            db_batch_finish(conn, batch_id, attempts, successes)
        except Exception:
            pass
        conn.close()


# ════════════════════════════════════════════════════════════════════════════
# Per-experiment retry state
# ════════════════════════════════════════════════════════════════════════════

def build_exp_state(jobs: list) -> dict:
    state: dict = {}
    for job in jobs:
        eid = job["id"]
        if eid not in state:
            target = job["target_runs"]
            mult   = max(float(job.get("max_retry_multiplier", 1)), 1)
            state[eid] = {
                "cpw":            job["cpw"],
                "target":         target,
                "only_succ":      bool(job.get("only_successful", False)),
                "max_attempts":   mult * target,
                "successes":      0,
                "attempts":       0,
                "next_attempt":   target + 1,   # next run_idx for retry jobs
                "base_job":       job,
            }
    return state


def _is_success(r: dict) -> bool:
    return r.get("status") in ("ok", "ok_no_gold")


# ════════════════════════════════════════════════════════════════════════════
# Progress reporting
# ════════════════════════════════════════════════════════════════════════════

def _progress_line(exp_state: dict, done_attempts: int, total_attempts: int) -> str:
    """
    When any experiment uses only_successful mode, report progress as
    successes / total-target so the progress line is meaningful.
    Otherwise report attempts / total.
    """
    any_only_succ = any(s["only_succ"] for s in exp_state.values())
    if any_only_succ:
        total_succ = sum(s["successes"] for s in exp_state.values())
        total_tgt  = sum(s["target"]    for s in exp_state.values())
        return f"PROGRESS {total_succ}/{total_tgt} attempts={done_attempts}"
    return f"PROGRESS {done_attempts}/{total_attempts}"


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    args     = parse_args()
    batch_id = args.batch_id or uuid.uuid4().hex
    print(f"BATCH_ID {batch_id}", flush=True)

    max_workers = max(int(args.max_workers or 1), 1)
    jobs        = load_jobs(args)

    if not jobs:
        print("No jobs found — check your manifest.", flush=True)
        return

    total_target = sum(
        j["target_runs"]
        for j in {j["id"]: j for j in jobs}.values()  # unique per experiment
    )
    total = len(jobs)   # grows if retries are added

    exp_state = build_exp_state(jobs)

    print(
        f"Pipeline → initial_jobs={total}  max_workers={max_workers}  "
        f"db={args.results_db}  only_successful={args.only_successful}",
        flush=True,
    )

    # Start background DB writer
    q      = Queue()
    writer = Thread(
        target=db_writer_loop,
        args=(args.results_db, batch_id, total_target, q),
        daemon=True,
    )
    writer.start()

    done_attempts = 0
    wall_start    = time.time()

    # ── Helper: process a completed result ────────────────────────────────
    def handle_result(job: dict, r: dict) -> "dict | None":
        """
        Emit result to DB queue, update exp_state.
        Returns a retry job if one is needed, else None.
        """
        nonlocal done_attempts, total
        q.put({"result": r})
        done_attempts += 1

        eid   = job["id"]
        state = exp_state.get(eid)
        if not state:
            return None

        state["attempts"] += 1
        success = _is_success(r)
        if success:
            state["successes"] += 1

        print(_progress_line(exp_state, done_attempts, total), flush=True)

        # Should we retry?
        if (state["only_succ"]
                and not success
                and state["successes"] < state["target"]
                and state["attempts"]  <= state["max_attempts"]):

            attempt_idx = state["next_attempt"]
            state["next_attempt"] += 1
            retry = make_retry_job(state["base_job"], attempt_idx)
            total += 1
            print(
                f"RETRY {state['cpw']} "
                f"(successes={state['successes']}/{state['target']} "
                f" attempts={state['attempts']}/{state['max_attempts']} "
                f" queuing attempt {attempt_idx})",
                flush=True,
            )
            return retry
        return None

    # ── Serial execution ──────────────────────────────────────────────────
    if max_workers == 1:
        pending = deque(jobs)
        while pending:
            job   = pending.popleft()
            r     = run_one_job(job)
            retry = handle_result(job, r)
            if retry:
                pending.append(retry)

    # ── Parallel execution ────────────────────────────────────────────────
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures: dict = {ex.submit(run_one_job, job): job for job in jobs}

            while futures:
                # wait() supports dynamically-added futures correctly
                done_set, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)

                for done_fut in done_set:
                    job = futures.pop(done_fut)
                    try:
                        r = done_fut.result()
                    except Exception as exc:
                        # Wrap unexpected thread-level exceptions
                        r = {
                            "job_id":           job["job_id"],
                            "cpw_name":         job["cpw"],
                            "framework":        job["framework"],
                            "model":            job.get("model", ""),
                            "attempt_number":   job["run_idx"],
                            "status":           "failed_error",
                            "error":            str(exc),
                            "total_seconds":    -1,
                            "gen_seconds":      -1,
                            "map_seconds":      -1,
                            "fix_seconds":      -1,
                            "layout_seconds":   -1,
                            "struct_sim_seconds": -1,
                            "render_diff_seconds": -1,
                            "ltl_seconds":      -1,
                            "cnstr_seconds":    -1,
                            "svg_seconds":      -1,
                            "gold_bpmn":        None,
                            "ltl_cnstr":        None,
                            "cnstr_ontology":   None,
                            "generated_bpmn":   None,
                            "mapped_bpmn":      None,
                            "node_sim":         None,
                            "struct_sim":       None,
                            "ged":              None,
                            "ltl_score":        None,
                            "ltl_conf":         None,
                            "cnstr_score":      None,
                            "cnstr_conf":       None,
                            "cpw_succ_cnstrs": None,
                            "cpw_total_cnstrs":      None,
                            "cpw_score":        None,
                            "incl_model_guide": int(bool(job.get("incl_model_guide"))),
                            "incl_knowl":       int(bool(job.get("incl_knowl"))),
                            "optim_prmt":       0,
                        }

                    retry = handle_result(job, r)
                    if retry:
                        new_fut = ex.submit(run_one_job, retry)
                        futures[new_fut] = retry

    # ── Shutdown ──────────────────────────────────────────────────────────
    q.put(None)
    writer.join()

    elapsed       = time.time() - wall_start
    total_succ    = sum(s["successes"] for s in exp_state.values())
    total_tgt_all = sum(s["target"]    for s in exp_state.values())

    print(
        f"DONE successes={total_succ}/{total_tgt_all} "
        f"attempts={done_attempts} elapsed={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()