# LLM4CPW

Extracts a BPMN process from a clinical pathway (CPW) description using an LLM. 
It can inject the following into the prompt to improve quality:
- Modelling guidelines
- Narrations of knowledge constraints

It then evaluates the LLM output as follows:

- _Based on a gold-standard reference process._
For that, an LLM is used to align activities from the LLM output and reference process.
Then, structural similarity between the two is calculated, and a diff diagram of the 
reference vs. LLM output process is generated.

- _Based on the relevant knowledge constraints._
Conformance with the original relevant constraints is checked, re-using the aligned LLM output from above.
For that purpose, it relies on the LoLA model checker.

Files:
- `batch_runner.py` – command-line entry point; prepares prompts and launches the pipeline.
- `pipeline.py` – the actual pipeline (extraction, mapping, fixing, scoring, storage).
- `gpt_direct_responses_api.py` – calls the OpenAI Responses API to generate the BPMN.
- `libs/` – helpers (activity mapping, BPMN utilities, similarity, constraint checks).
- `data/` – CPW texts (`cpw/`), prompt templates, rules and knowledge statements (`prmt/`),
  reference processs (`ref/`).

## Installation

Requirements: Python 3.10+ (developed on 3.13), an OpenAI API key, and the
[LoLA](https://theo.informatik.uni-rostock.de/theo-forschung/tools/lola/) model checker.

1. **Create a virtual environment and install the Python packages**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install LoLA.** The `lola` executable must be on your `PATH`;
   it is used for the LTL_f constraint checks.

3. **Install Graphviz** (the `dot` executable, e.g. `brew install graphviz`), needed to render
   BPMN/Petri-net diagrams.

4. **Set your OpenAI key** (the batch runner exits if it is missing):

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

Run everything from this directory, with the virtual environment active (the batch
runner starts `python pipeline.py`, so it must resolve to the venv's Python). 

In case of a new terminal session, be sure to activate the environment again before running:

```bash
source .venv/bin/activate
```

## Running the batch runner

```bash
python batch_runner.py -c CAP -s GPTDirect -m gpt-5.6-low
```

| Option | Meaning |
|---|---|
| `-c`, `--cpw` | CPW to run; matches `data/cpw/<cpw>.txt`. Comma-separated for several, e.g. `-c "CAP,COPD,Stroke 1"`. Available: `CAP`, `COPD`, `Heart Failure`, `Hip Fracture`, `Stroke 1`–`Stroke 4`. |
| `-s`, `--system` | System to use. Currently only `GPTDirect`. |
| `-m`, `--model` | Model name suffixed by reasoning effort, e.g. `gpt-5.6-low`, `gpt-5.4-medium`. |
| `-mg`, `--incl_model_guide` | Include the modelling guidelines (`data/prmt/rules/`) in the prompt. |
| `-ks`, `--incl_knowl` | Include the knowledge statements (`data/prmt/knowledge/<cpw>.txt`) in the prompt. Only takes effect together with `-mg`. |
| `-r`, `--runs` | Number of successful runs per CPW (default 1). |
| `-v`, `--verbose` | Print detailed step output from the pipeline. |

Note that the prompt template is chosen from flags (`-mg`, `-ks`): neither flag → `instruct_tmpl-0.txt`,
`-mg` → `instruct_tmpl-1.txt`, `-mg -ks` → `instruct_tmpl-2.txt`.

Failed runs are retried until the requested number of successful runs is reached,
up to 3× the requested number of runs in total (`MAX_RETRY_MULTIPLIER`).
Defaults such as the mapping model and the retry mode (`repair`) are set in the
`MANIFESTS` dict at the top of `batch_runner.py`.

Examples:

```bash
# Run with modelling guidelines and knowledge statements, one run, verbose
python batch_runner.py -c CAP -s GPTDirect -m gpt-5.6-low -mg -ks -r 1 -v

# Run with several CPWs in sequence, only modelling guidelines, three runs
python batch_runner.py -c "CAP,COPD" -s GPTDirect -m gpt-5.6-low -mg -r 3
```

## Pipeline steps

Each run in [`pipeline.py`](pipeline.py) runs these steps; the first failing step ends the run
and records its status (e.g. `failed_mapping`).

1. **Generate BPMN** – the instruction and input prompts are written into the `run/` folder
   and sent to the OpenAI model, which returns a BPMN XML file (`process.bpmn`) as the LLM output process.
2. **Activity mapping** – an LLM aligns the labels of the LLM output process to those of
   the reference process (`process_mapped.bpmn`, `label_align.csv`), so the two can be compared.
3. **Fix BPMN XML** – adds the incoming/outgoing references that gateways need
   (`*_fixed.bpmn`) and adds diagram layout information (`*_fixed_layout.bpmn`).
4. **Similarity** – calculates structural similarity to the reference process (node similarity,
   structural similarity, graph edit distance) using [struct_simil.py](libs/cmp4proc/bpmn_comparison/struct_simil.py). 
   Diff diagrams of reference vs. LLM output
   process are written as `cmp_ref.svg` and `cmp_out.svg`.
5. **Constraint conformance** – checks the LLM output against (a) LTL_f constraints relevant to the CPW
   (`data/prmt/knowledge/cnstrs/<cpw>.json`) using [ltl_check.py](libs/cmp4proc/ltl/lola/ltl_check.py) 
   and (b) the other constraints in the ontology `constraints.owl` using [activ_cnstr_check.py](libs/cmp4proc/activ_cnstr/activ_cnstr_check.py). 
   Each yields a score and a list of failed constraints. 
6. **Convert BPMN to SVG** – renders the final BPMN as an SVG image.

## Output

Everything is written below `out/`:

```
out/
├── results_db.sqlite                      # all results, shared by every run
└── <cpw>-<system>_<model>[-mg][-ks]/      # one folder per experiment configuration
    ├── batch_manifest.json                # the job description that was passed to pipeline.py
    └── Run-<experiment>-attempt<N>-<YYYYMMDD_HHMMSS>/   # one folder per run
```

For example `out/cap-gptdirect_gpt_5.6_low-mg-ks/Run-cap-gptdirect_gpt_5.6_low-mg-ks-attempt1-20261020_134838/`.
The experiment folder name is the lower-cased CPW, system and model.

Each attempt folder contains the files of the steps above:

| File | Produced by |
|---|---|
| `instruct.txt`, `input.txt` | the prompts sent to the model (step 1) |
| `process.bpmn` | LLM output BPMN (step 1) |
| `process_mapped.bpmn`, `label_align.csv` | activity mapping (step 2) |
| `process_mapped_fixed.bpmn`, `process_mapped_fixed_layout.bpmn` | XML fix and layout (step 3) |
| `cmp_ref.svg`, `cmp_out.svg` | structural comparison diagrams (step 4) |
| `process_mapped_fixed_layout.pnml/.dot/.svg/.png`, `*_lola.lola/.svg`, `*-bp.csv` | Petri-net / LoLA artefacts of the process (step 5) |
| `process_mapped.svg` | rendered BPMN (step 6) |

### Results database (`out/results_db.sqlite`)

One SQLite file collects results for all experiments :

- **`runs`** – one row per attempt: CPW (`qbp_name`), framework, model, attempt number,
  `status` (`ok` or the failing step), error message, timings per step, file paths,
  `node_sim`, `struct_sim`, `ged`, `ltl_score`, `ltl_conf` (for temporal constraints), 
  `cnstr_score`, `cnstr_conf` (for non-temporal constraints),
  and whether guidelines/knowledge were in the prompt (`incl_model_guide`, `incl_knowl`).
- **`batches`** – one row per pipeline invocation: start/finish time, status, target
  number of runs, total attempts and successes.

The easiest way to inspect it is [DB Browser for SQLite](https://sqlitebrowser.org/): 
open `out/results_db.sqlite`, then use
the *Browse Data* tab to look through the `runs` and `batches` tables.
(Close the file in the browser (or open it read-only) before starting a
new run, so it doesn't hold a lock on the database.)

## Constraints ontology

We created an OWL2 DL knowledge constraint ontology to represent knowledge constraints 
and LTL_f formulas independent of any particular model checker notation.
See `data/prmt/knowledge/cnstrs/constraints.owl`.

It includes the following constraint classes: `DecisionalDependency`, `Conflict`, `EntryPoint`, `ExitPoint`,
`PrioritizedConditional`, `Alternative`, `PartOf`, or a raw `LtlConstraint`. 
Every class except `PartOf` has an attached LTL_f formula (`hasLtlFormula`).

To create knowledge constraints (e.g., decisional dependency) for a particular CPW, 
you can instantiate the above classes. 
Each constraint is attached to a `ProcessModel` class instance.
The ontology currently contains instances for 8 CPW.

### Converting constraint instances

[`convert_constraints.py`](data/prmt/knowledge/cnstrs/convert_constraints.py)
can convert constraint instances into:

- LTL_f formulas, fine-tuned for a particular model checker (currently, LoLA)
- Narrations, which can be used for prompt-time injection


```bash
cd data/prmt/knowledge/cnstrs
python convert_constraints.py constraints.owl [-m MODEL] [-t {syntax,narration}] [-s {default,lola}] [-f {json_list,string}] [-o OUT_PATH]
```

| Option | Meaning |
|---|---|
| `ontology` | Path of the ontology file (here `constraints.owl`). |
| `-m`, `--model` | Process model to convert, e.g. `CAP` or `"Stroke 1"` (its `rdfs:label`). Without it, all process models are converted. |
| `-t`, `--target` | `syntax` (default): LTL_f formulas. `narration`: human-readable sentences. |
| `-s`, `--out-syntax` | Formula syntax, for `-t syntax`: `default` (formulas as stored in the ontology) or `lola` (applies the rewrites LoLA needs). |
| `-f`, `--out-format` | `json_list` (default): a JSON array. `string`: one text with the constraints separated by a blank line. |
| `-o`, `--out-path` | Save the result to this path. With all models it is a directory receiving one file per model (`<model>.json` or `<model>.txt`), and nothing is printed. |

Identical constraints (same formula or narration) are only included once.

#### Generating LTL_f formulas

```bash
# Formulas of one model, printed as JSON
python convert_constraints.py constraints.owl -m CAP

# Formulas of all models, one JSON file per model, e.g. to regenerate <cpw>.json
python convert_constraints.py constraints.owl -o out_formulas
```

The `lola` syntax (`-s lola`) applies rewrites for LoLA's model checker.

The generated JSON has the the following structure:

```json
[
  {"name": "cap1", "type": "raw",
   "formula": "G (F \"Perform chest x-ray\" -> (NOT \"Give antibiotics\" U \"Perform chest x-ray\"))"},
  ...
]
```

For the [LLM4CPW pipeline](#pipeline-steps), LTL_f JSON files need to be stored in `data/prmt/knowledge/cnstrs`.


#### Generating narrations

```bash
# Narrations of one model, as text (e.g. for data/prmt/knowledge/<cpw>.txt)
python convert_constraints.py constraints.owl -m CAP -t narration -f string

# Narrations of all models, one .txt file per model
python convert_constraints.py constraints.owl -t narration -f string -o out_narrations
```

Each constraint class has its own sentence template, for example:

> "Give antibiotics" may only occur after "Blood cultures" potentially takes place.
>
> "Check for sepsis onset" must be the first task in the process.

`PartOf` constraints, which have no formula, appear only in the narrations ("A" is part of "Y").
Formulas that don't match a known pattern get a generic operator-by-operator paraphrase.

For the [LLM4CPW pipeline](#pipeline-steps), narrations need to be stored in `data/prmt/knowledge`.