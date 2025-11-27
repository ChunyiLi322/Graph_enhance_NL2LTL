# Graph-Enhanced Prompt for NL→LTL Translation with Multi-Trial Inquiry

This repository implements an end-to-end pipeline for translating **natural language (NL) specifications into Linear Temporal Logic (LTL)** using:

- A **Concept–Predicate–Operator (CPO) graph** built from raw text;
- **Random-Walk-with-Restart (RWR)** to generate graph-based “thought chains”;
- A **Large Language Model (LLM)** (via an OpenAI-compatible API) to produce LTL formulas;
- A **formal LTL parser** (Lark-based) to check syntactic validity;
- **CSV-based logging** for multi-model, multi-trial evaluation;
- A **batch runner (`batch_run.py`)** to process many NL requirements from a text file.

If you use this code in academic work, please cite the corresponding paper (placeholder):

```text
.
├── Graph_enhance_NL2LTL.py   # Main script: graph-enhanced NL→LTL + CSV logging
├── batch_run.py              # Batch script: apply the main script line-by-line over a file
├── SSCS.txt                  # Example NL specification file (one requirement per line)
├── X_results.csv             # Example output CSV (generated)
└── README.md
```

---

## Required Packages

Install the core dependencies:

```bash
pip install networkx numpy nltk sentence-transformers openai lark-parser spacy
```

Optional (for stricter temporal logic tooling, if you intend to extend the checker):

```bash
pip install spot
```

### 3.3 Language Resources

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

For NLTK tokenization and POS tagging:

```python
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
```

---

## Running the Batch Script

Ensure Graph_enhance_NL2LTL.py, batch_run.py, and SSCS.txt are in the same directory (or adjust paths accordingly).

Configure the following variables at the top of batch_run.py:

- script_name – main script name (default: "Graph_enhance_NL2LTL.py").
- n_paths – maximum path length.
- n_trials – number of random-walk trials per line.
- nl_file – path to the NL specification file (e.g., "SSCS.txt").
- save_csv_file – path to the output CSV (e.g., "X_results.csv").
- model – list of model names (Python list).
- api – list of corresponding API keys (same length as model).

Run:

```bash
复制代码
python batch_run.py
```

For each non-empty line in SSCS.txt, batch_run.py will:

- Call Graph_enhance_NL2LTL.py once,
- Generate random-walk thought chains,
- Invoke all configured models,
- Append the results to X_results.csv.

Tip:
If you are concerned about rate limits, you can add a time.sleep(...) between successive subprocess.run calls.

---

## Extensibility and Notes

LTL grammar:
If your target LTL dialect differs from the current grammar, update LTLChecker’s ltl_grammar accordingly.

Model routing:
Names like claude-X, deepseek-X, gemini-X assume an OpenAI-compatible proxy gateway. Ensure your backend supports these.

Error handling:
LLMCaller includes basic exception handling but you can extend it.

```bibtex
@article{nl2ltl_graph_prompt,
  title   = {Graph-Enhanced Prompt for NL to LTL Translation with Multi-Trial Inquiry},
  author  = {To be completed},
  journal = {To be completed},
  year    = {To be completed} 
}
```
