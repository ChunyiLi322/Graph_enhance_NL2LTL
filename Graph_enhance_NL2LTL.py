"""
Code for "Graph-Enhanced Prompt for NL to LTL Translation with Multi-Trial Inquiry"  (OpenAI API integration + multi-trial accuracy)
==============================================================================
A full code that:
1. Builds a Concept-Predicate-Operator (CPO) graph automatically from raw texts via **NLTK**
2. Pre-loads Linear Temporal Logic (LTL) operators as Operator nodes
3. Runs Random-Walk-with-Restart (RWR) to harvest multiple thought-chains
4. Scores each chain with a probabilistic **confidence** metric
5. **Calls an LLM (OpenAI API) multiple times** to translate the prompt → LTL formulas
6. Parses each formula and reports **accuracy (%)** across trials

------------------------------------------------------------
Setup
-----
bash
pip install networkx numpy nltk sentence-transformers openai lark-parser
# Optional but recommended for strict LTL parsing
pip install spot  # requires C++ toolchain; skip if unavailable

export OPENAI_API_KEY="sk-..."  # and set via os.environ os.getenv("OPENAI_API_BASE", "https://X")

------------------------------------------------------------
Code
----
"""
from __future__ import annotations
import os, random, argparse, textwrap, math, json
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import numpy as np
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import csv
import ast
import time
from openai import error as openai_error
import spacy
import math, random, networkx as nx

# ===== Additional initialization in CPOGraph.__init__ (do not remove existing init)=====
class CPOGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self._alias_tables = {}
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self._nlp = spacy.load("en_core_web_sm")

        self.role_weights = {
            "subj": 1.4, "obj": 1.3, "iobj": 1.1,
            "prep": 0.9, "time": 0.9, "loc": 0.9,
            "modifier": 0.7, "conj": 0.6, "logic": 0.8
        }

        self.logic_keywords = {
            # --- Conditional ---
            "if": "IMPLIES", "If": "IMPLIES",
            "then": "IMPLIES", "Then": "IMPLIES",
            "provided": "IMPLIES", "Provided": "IMPLIES",
            "assuming": "IMPLIES", "Assuming": "IMPLIES",
            "suppose": "IMPLIES", "Suppose": "IMPLIES",
            "otherwise": "IMPLIES", "Otherwise": "IMPLIES",

            # --- Temporal ---
            "when": "UNTIL", "When": "UNTIL",
            "whenever": "ALWAYS", "Whenever": "ALWAYS",
            "until": "UNTIL", "Until": "UNTIL",
            "before": "UNTIL", "Before": "UNTIL",
            "after": "NEXT", "After": "NEXT",
            "once": "NEXT", "Once": "NEXT",
            "immediately": "NEXT", "Immediately": "NEXT",
            "as soon as": "NEXT", "As soon as": "NEXT",
            "next": "NEXT", "Next": "NEXT", "promptly": "NEXT",

            # --- Frequency ---
            "always": "ALWAYS", "Always": "ALWAYS",
            "whilst": "ALWAYS", "Whilst": "ALWAYS",
            "whiles": "ALWAYS", "Whiles": "ALWAYS",
            "eventually": "EVENTUALLY", "Eventually": "EVENTUALLY",
            "finally": "EVENTUALLY", "Finally": "EVENTUALLY",
            "sometime": "EVENTUALLY", "Sometime": "EVENTUALLY",
            "sometimes": "EVENTUALLY", "Sometimes": "EVENTUALLY",
            "at last": "EVENTUALLY", "At last": "EVENTUALLY",
            "sooner or later": "EVENTUALLY", "Sooner or later": "EVENTUALLY",

            # --- Negation ---
            "never": "NOT", "Never": "NOT",
            "not": "NOT", "Not": "NOT",
            "no": "NOT", "No": "NOT",
            "without": "NOT", "Without": "NOT",

            # --- Conjunction (AND) ---
            "and": "AND", "And": "AND",
            "both": "AND", "Both": "AND",
            "as well as": "AND", "As well as": "AND",
            "together with": "AND", "Together with": "AND",
            "along with": "AND", "Along with": "AND",
            "in addition": "AND", "In addition": "AND",

            # --- Disjunction (OR) ---
            "or": "OR", "Or": "OR",
            "either": "OR", "Either": "OR",
            "alternatively": "OR", "Alternatively": "OR",
            "otherwise": "OR", "Otherwise": "OR",

            # --- Contrast / Inference ---
            "but": "AND", "But": "AND",
            "although": "IMPLIES", "Although": "IMPLIES",
            "though": "IMPLIES", "Though": "IMPLIES",
            "yet": "IMPLIES", "Yet": "IMPLIES",
            "therefore": "IMPLIES", "Therefore": "IMPLIES",
            "hence": "IMPLIES", "Hence": "IMPLIES",
            "so": "IMPLIES", "So": "IMPLIES",
            "thus": "IMPLIES", "Thus": "IMPLIES"
        }

    def add_node(self, nid: str, ntype: str, **attrs):
        if nid not in self.G:
            self.G.add_node(nid, ntype=ntype, **attrs)

    def add_edge(self, u: str, v: str, weight: float = 1.0, **attrs):
        if self.G.has_edge(u, v):
            self.G.edges[u, v]["weight"] += weight
        else:
            self.G.add_edge(u, v, weight=weight, **attrs)

    def _canon(self, text: str) -> str:
        return "_".join(text.strip().lower().split())

    def _dep_to_graph(self, sent: str):
        """Build the graph based on dependency parsing and detect logical keywords (dynamically introduce operator nodes)."""
        doc = self._nlp(sent)

        # --- Concepts: noun phrases ---
        for chunk in doc.noun_chunks:
            c_text = chunk.text.lower()
            c_id = self._canon(c_text)
            self.add_node(c_id, "C", label=c_text)

        # --- Predicates: extended verb-object / verb-preposition structures ---
        for tok in doc:
            if tok.pos_ != "VERB":
                continue
            if tok.lemma_.lower() in {"be", "have", "do"}:
                continue

            phrase_parts = [tok.lemma_]

            # Object
            obj = next((ch for ch in tok.children if ch.dep_ in {"dobj", "obj"}), None)
            if obj:
                obj_phrase = " ".join([t.text for t in obj.subtree])
                phrase_parts.append(obj_phrase)

            # Prepositional phrase
            prep_phrases = []
            for child in tok.children:
                if child.dep_ == "prep":
                    pobj = next((gc for gc in child.children if gc.dep_ == "pobj"), None)
                    if pobj:
                        prep_phrases.append(child.text.lower() + " " + " ".join([t.text for t in pobj.subtree]))
            if prep_phrases:
                phrase_parts.extend(prep_phrases)

            phrase = " ".join(phrase_parts).lower()
            v_id = self._canon(phrase)
            self.add_node(v_id, "P", label=phrase)

            # Subject
            for child in tok.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    phrase = " ".join([t.text for t in child.subtree]).lower()
                    c_id = self._canon(phrase)
                    self.add_node(c_id, "C", label=phrase)
                    self.add_edge(c_id, v_id, weight=self.role_weights["subj"], role="subj")

            # Object connection
            if obj:
                c_id = self._canon(obj_phrase)
                self.add_node(c_id, "C", label=obj_phrase)
                self.add_edge(v_id, c_id, weight=self.role_weights["obj"], role="obj")

            # Prepositional phrase connection
            for prep in prep_phrases:
                c_id = self._canon(prep)
                self.add_node(c_id, "C", label=prep)
                self.add_edge(v_id, c_id, weight=self.role_weights["prep"], role="prep")

        # --- Detect logical keywords → dynamically add O nodes ---
        for tok in doc:
            lw = tok.text.lower()
            if lw in self.logic_keywords:
                op = self.logic_keywords[lw]
                self.add_node(op, "O", label=op)
                # Connect the operator node with all verb nodes
                for t in doc:
                    if t.pos_ == "VERB":
                        v_phrase = t.lemma_
                        v_id = self._canon(v_phrase)
                        if v_id in self.G:
                            self.add_edge(op, v_id, weight=self.role_weights["logic"], role="logic")
                            self.add_edge(v_id, op, weight=self.role_weights["logic"], role="logic")

    def build_from_texts_structured(self, texts):
        print("[CPOGraph] Building graph with phrase-level nodes...")
        for s in texts:
            self._dep_to_graph(s)
        self.normalize_edge_probs()
        print(f"[CPOGraph] Total nodes: {self.G.number_of_nodes()}, edges: {self.G.number_of_edges()}")

    def normalize_edge_probs(self):
        self._alias_tables.clear()
        for u in self.G.nodes:
            succ = list(self.G.successors(u))
            if not succ:
                continue
            raw = [self.G.edges[u, v].get("weight", 1.0) for v in succ]
            m = max(raw) if raw else 0.0
            exps = [math.exp(r - m) for r in raw]
            s = sum(exps) or 1.0
            probs = [e / s for e in exps]
            for v, p in zip(succ, probs):
                self.G.edges[u, v]["prob"] = p
            # alias table
            K = len(probs)
            scaled = [p * K for p in probs]
            small = [i for i, x in enumerate(scaled) if x < 1.0]
            large = [i for i, x in enumerate(scaled) if x >= 1.0]
            J = [0] * K
            q = [0.0] * K
            while small and large:
                sidx = small.pop()
                lidx = large.pop()
                q[sidx] = scaled[sidx]
                J[sidx] = lidx
                scaled[lidx] -= (1.0 - scaled[sidx])
                (small if scaled[lidx] < 1.0 else large).append(lidx)
            for idx in large + small:
                q[idx] = 1.0
                J[idx] = idx
            self._alias_tables[u] = (succ, J, q)


class RandomWalkSampler:
    def __init__(self, graph: CPOGraph, scorer, restart_prob=0.15, max_steps=8, question=""):
        self.G = graph.G
        self.alias = graph._alias_tables
        self.gamma = restart_prob
        self.L = max_steps
        self.question = question
        self.graph = graph
        self.scorer = scorer
        self.role_boost = {
            "subj": 1.08, "obj": 1.07, "iobj": 1.03,
            "prep:to": 1.03, "conj": 0.92, "modifier": 0.90
        }

    def _node_label(self, nid):
        return nid

    def _weighted_successors(self, u):
        if u not in self.alias:
            return []
        succ, J, q = self.alias[u]
        out = []
        for v in succ:
            if self.G.nodes[u]["ntype"] == "O" and self.G.nodes[v]["ntype"] == "O":
                continue
            p = self.G.edges[u, v].get("prob", 1e-6)
            role = self.G.edges[u, v].get("role", "")
            if role in self.role_boost:
                p *= self.role_boost[role]
            elif isinstance(role, str) and role.startswith("prep:"):
                p *= 0.97
            out.append((v, p))
        s = sum(p for _, p in out) or 1.0
        return [(v, p / s) for v, p in out]

    def sample(self, start, n_walks=1):
        print(f"[RandomWalk] start from: {start}")
        paths = []
        if start not in self.G:
            return paths

        all_concepts = [n for n, d in self.G.nodes(data=True) if d["ntype"] == "C"]

        for _ in range(n_walks):
            curr = start
            path = [curr]
            for _ in range(self.L - 1):
                if random.random() < self.gamma:
                    curr = start
                    if curr not in path:
                        path.append(curr)

                cand = self._weighted_successors(curr)
                if not cand:
                    # Fallback: avoid dead ends by randomly picking a node from the whole graph
                    nxt = random.choice(list(self.G.nodes))
                    path.append(nxt)
                    curr = nxt
                    continue

                r = random.random()
                acc = 0.0
                nxt = cand[-1][0]
                for v, p in cand:
                    acc += p
                    if r <= acc:
                        nxt = v
                        break

                if nxt in path:
                    # Prefer unvisited successors with high probability
                    fallback = next((v for v, p in sorted(cand, key=lambda x: -x[1]) if v not in path), None)
                    if fallback is None:
                        # Try to choose an unvisited node from the whole graph
                        remaining = [n for n in self.G.nodes if n not in path]
                        nxt = random.choice(remaining) if remaining else random.choice(list(self.G.nodes))
                    else:
                        nxt = fallback

                path.append(nxt)
                curr = nxt

            def _node_label(self, nid):
                return self.G.nodes[nid].get("label", nid)

            paths.append(" → ".join(self._node_label(n) for n in path))
        return paths

    def build_thought_chains(self, start: str, n_trials: int = 3) -> List[Tuple[str, float, int, float]]:
        """Same interface as original, but fixes scorer reuse and split logic."""
        results = []
        for _ in range(n_trials):
            path_strs = self.sample(start, n_walks=1)
            if not path_strs:
                continue
            # Split by separator carefully: avoid splitting node names that contain underscores
            nodes = path_strs[0].split(" → ")
            conf, miss, raw = self.scorer.score(nodes)
            chain_txt = " → ".join(nodes)
            results.append((chain_txt, conf, miss, raw))
        return results

    def generate_prompts_only(self, question: str, n_trials: int = 3):
        print(f"[Evaluator] Generating prompt chains for question: {question}")
        if isinstance(question, list):
            question = " ".join(question)
        tokens = pos_tag(word_tokenize(question))
        start = next((t.lower() for t, tag in tokens if tag.startswith("NN")), None)
        if not start or start not in self.graph.G:
            start = random.choice([n for n, d in self.graph.G.nodes(data=True) if d["ntype"] == "C"])

        chains = self.build_thought_chains(start, n_trials)
        prompts = []
        miss_list = []
        raw_list = []
        conf_list = []

        for chain_txt, conf, miss, raw in chains:
            prompt = CHAIN_TEMPLATE.format(
                question=question,
                graph_path=chain_txt + f"\n(Expected path confidence≈{conf:.2f})"
            )
            print(f"\n---- Prompt ----\n{prompt}\n")
            prompts.append(prompt)
            miss_list.append(miss)
            raw_list.append(raw)
            conf_list.append(conf)

        return prompts, miss_list, raw_list, conf_list


# ---------- 3. Confidence Scorer ------------------------------------------- #

class ConfidenceScorer:
    def __init__(self, graph: CPOGraph):
        self.G = graph.G

    def score(self, path: List[str]) -> Tuple[float, int, float]:
        if len(path) < 2:
            # Always return a triple
            return 0.0, 0, 0.0

        miss_cnt = 0
        log_probs = 0.0

        for u, v in zip(path[:-1], path[1:]):
            if not self.G.has_edge(u, v):
                miss_cnt += 1
                edge_p = 0.02
            else:
                edge_p = self.G[u][v]["prob"]
            log_probs += math.log(edge_p)

        avg_log_p = log_probs / (len(path) - 1)
        geom_mean = math.exp(avg_log_p)
        miss_penalty = 0.9 ** miss_cnt
        len_penalty = 1.0 / math.sqrt(len(path))
        conf_raw = geom_mean * len_penalty * miss_penalty
        final_conf = 1 / (1 + math.exp(-7 * (conf_raw - 0.003)))

        return final_conf, miss_cnt, conf_raw


# ---------- 4. LLM Caller --------------------------------------------------- #
class LLMCaller:
    """Light wrapper around OpenAI ChatCompletion for multi-trial generation."""
    def __init__(self, model: str = "gpt-X", api: str = "sk-X", temperature: float = 0.5):
        import openai  # local import so script works even if lib missing during tests

        openai.api_key = api
        openai.api_base = os.getenv("OPENAI_API_BASE", "https://X")
        self.client = openai
        self.model = model
        self.temperature = temperature

    def generate_formulas(self, prompt: str, n_trials: int = 1) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a logic assistant that converts natural language statements into LTL formulas.\n"
                    "Always output the final formula in a line that starts with 'Formal:'. Do not include explanation after that line.\nGive me only the final answer. Do not show your reasoning.\n"
                    "For example: Formal: G(light_on → F(light_off))"
                )
            },
            {"role": "user", "content": prompt}
        ]
        formulas = []

        for trial in range(n_trials):
            print(f"[LLM] Trial {trial+1} - sending prompt to OpenAI API...")
            try:
                resp = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=10240,
                    timeout=100,
                )
                content = resp.choices[0].message.content.strip()

                # First try to extract from "Formal:" line
                if "Formal:" in content:
                    formula_line = [line for line in content.splitlines() if "Formal:" in line][-1]
                    formula = formula_line.split("Formal:")[1].strip()
                else:
                    # Heuristic fallback: extract most formula-looking expression
                    import re
                    match = re.search(r'[FGXGRW¬∧∨→\(\)\w\s]+', content)
                    formula = match.group(0).strip() if match else ""

                formulas.append(formula)
                print(f"[LLM] Trial {trial+1} - received formula: {formula}")

            except openai_error.Timeout as e:
                print(f"[LLM] Trial {trial+1} failed: Timeout after 30s – {e}")
                formulas.append("[Timeout]")

            except openai_error.APIConnectionError as e:
                print(f"[LLM] Trial {trial+1} failed: Connection error – {e}")
                formulas.append("[ConnectionError]")

            except openai_error.RateLimitError as e:
                print(f"[LLM] Trial {trial+1} failed: Rate limit exceeded – {e}")
                formulas.append("[RateLimitError]")

            except openai_error.AuthenticationError as e:
                print(f"[LLM] Trial {trial+1} failed: Invalid API key – {e}")
                formulas.append("[AuthError]")

            except openai_error.InvalidRequestError as e:
                print(f"[LLM] Trial {trial+1} failed: Invalid request – {e}")
                formulas.append("[InvalidRequest]")

            except openai_error.APIError as e:
                print(f"[LLM] Trial {trial+1} failed: API error – {e}")
                formulas.append("[APIError]")

            except Exception as e:
                print(f"[LLM] Trial {trial+1} failed: Unknown error – {e}")
                formulas.append("[UnknownError]")

        return formulas[0]


class LTLChecker:
    def __init__(self):
        from lark import Lark
        print("[LTLChecker] Using backend: lark (pure Python parser)")

        ltl_grammar = r"""
             ?start: formula

            // ===== Precedence (lowest → highest) =====
            ?formula: implication

            // right-associative: a -> (b -> c)
            ?implication: disjunction
                        | disjunction IMPLIES implication          -> implies

            // left-associative: a ∨ b ∨ c
            ?disjunction: conjunction
                        | disjunction OR conjunction               -> or

            // left-associative: a ∧ b ∧ c
            ?conjunction: temporal_bin
                        | conjunction AND temporal_bin             -> and

            // Temporal binary (left-associative by construction)
            ?temporal_bin: unary_expr
                         | temporal_bin UNTIL unary_expr           -> until
                         | temporal_bin RELEASE unary_expr         -> release
                         | temporal_bin WEAKUNTIL unary_expr       -> weakuntil

            // ===== Unary layer =====
            // Support F[lo,hi] φ without parentheses, or F[lo,hi](φ) with parentheses
            ?unary_expr: timed_unary
                       | unary_chain
                       | "(" formula ")"

            // F/G with time window, followed by either unary_chain (no parens) or "( formula )"
            timed_unary: (ALWAYS | EVENTUALLY) _space? time_window _space? ( unary_chain | "(" formula ")" )  -> timed_unary

            // Unary prefix chain: X X ! ~ G F etc. (G/F without an interval is treated as a normal unary prefix)
            unary_chain: unary_op _space? unary_chain               -> unary
                       | primary

            // Tightest atomic unit
            primary: atom
                   | "(" formula ")"

            // ===== Atoms =====
            ?atom: comparison
                 | unit_value
                 | var
            
            comparison: var _space? comparator _space? (unit_value | NUMBER | var | STRING)
            time_window: "[" _space? NUMBER _space? "," _space? NUMBER _space? "]"
            unit_value: NUMBER unit

            // ===== Identifiers & Units =====
            var: /[a-zA-Z_][a-zA-Z0-9_\-\/\.]*/
            unit: /(℃|°|º|˚|℉|K|%|ppm|g\/m³|kg|Pa|bar|kPa|hPa|mmHg|degC|degF)/i

            // ===== Numbers =====
            NUMBER: /[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?/

            // ===== Operators =====
            comparator: ">" | "<" | ">=" | "<=" | "==" | "!=" | "=" | "≤" | "≥"

            // Note: put "||" before "|" to avoid splitting "||" into two "|" tokens
            AND: "∧" | "/\\" | "&" | "\\land"
            OR: "∨" | "\\/" | "||" | "|" | "\\lor"
            UNTIL: "U"
            RELEASE: "R"
            WEAKUNTIL: "W"
            IMPLIES: "→" | "->" | "\\rightarrow"

            // Unary operators (non-timed)
            NOT: "!" | "¬"
            NEG: "\\neg"
            TILDE: "~" | "∼"
            NEXT: "X" | "\\bigcirc" | "◯"
            ALWAYS: "G" | "\\Box" | "□"
            EVENTUALLY: "F" | "\\diamond" | "⋄"

            unary_op: NOT | NEG | TILDE | NEXT | ALWAYS | EVENTUALLY
            
            // --- Helper for whitespace handling ---
            _space: /[ \t]+/
            %import common.ESCAPED_STRING   -> STRING
            %ignore " "

        """
        self.parser = Lark(ltl_grammar, start="start")

    def is_valid(self, formula: str) -> bool:
        formula = formula.strip()
        if not formula:
            print(f"[LTLChecker] Formula is empty | Valid: False")
            return False
        try:
            self.parser.parse(formula)
            print(f"[LTLChecker] Formula: {formula} | Valid: True")
            return True
        except Exception as e:
            print(f"[LTLChecker] Formula: {formula} | Valid: False | Error: {e}")
            return False


# ---------- 6. Accuracy Reporter ------------------------------------------- #
@dataclass
class TrialResult:
    formula: str
    accurate: bool

class AccuracyReporter:
    def __init__(self):
        self.results: List[TrialResult] = []

    def add(self, formula: str, accurate: bool):
        self.results.append(TrialResult(formula, accurate))

    def summary(self) -> str:
        total = len(self.results)
        acc = sum(r.accurate for r in self.results) / total if total else 0
        lines = [f"Trial {i+1}: {r.formula}   {'√' if r.accurate else 'X'}"
                 for i, r in enumerate(self.results)]
        lines.append(f"Overall accuracy: {acc:.2%}")
        print(f"[AccuracyReporter] {acc:.2%} of formulas valid across {total} trials.")
        return "\n".join(lines)

# ---------- 7. Prompt Builder ---------------------------------------------- #
CHAIN_TEMPLATE = """You are an LTL translation agent. Given a natural language statement, perform structured semantic reasoning and output the corresponding LTL formula. Ensure the output strictly follows valid LTL syntax, prefixed with 'Formal:'.\n\nUser Question:\n{question}\n\nSuggested by the Knowledge Graph:\n{graph_path}\nFormal:"""


# ---------- 8. End-to-end Evaluator --------------------------------------- #
class Evaluator:
    def __init__(self, graph: CPOGraph, walker: RandomWalkSampler,
                 scorer: ConfidenceScorer, checker: LTLChecker,
                 llm: LLMCaller):
        self.graph = graph
        self.walker = walker
        self.scorer = scorer
        self.checker = checker
        self.llm = llm

    def generate_formula_and_accuracy(self, prompt: str, n_trials: int = 1):
        # print("------------generate_formula_and_accuracy------------",prompt)
        total_valid = 0
        formula = self.llm.generate_formulas(prompt, n_trials=1)
        return formula


# ---------- 9. CLI --------------------------------------------------------- #
def main():
    print("--------------------main---------------------------")
    parser = argparse.ArgumentParser(description="NL→LTL demo with random-walk thought chains & accuracy analysis")
    parser.add_argument("--texts", nargs="+", required=True, help="Training sentences to build the graph")
    parser.add_argument("--save_csv_file", type=str)
    parser.add_argument("--n_paths", type=int, default=3, help="Number of nodes in a path.")
    parser.add_argument("--n_trials", type=int, default=3, help="Number of prompts.")
    parser.add_argument("--model", nargs="+", default=["gpt-4o-mini"], help="List of model names to run.")
    parser.add_argument("--api", nargs="+", default=["sk-XXXX"], help="List of API keys matching the models.")
    print("--------------------parser---------------------------")
    args = parser.parse_args()
    print("--------------------graph---------------------------")
    graph = CPOGraph()
    graph.build_from_texts_structured(args.texts)
    print("--------------------scorer---------------------------")
    scorer = ConfidenceScorer(graph)
    print("--------------------checker---------------------------")
    walker = RandomWalkSampler(graph, scorer=scorer, max_steps=args.n_paths, question=args.texts)
    checker = LTLChecker()

    print(f"\n========== Running1: {args.model[0]} ==========")
    model_str_list = ast.literal_eval(args.model[0])
    api_str_list = ast.literal_eval(args.api[0])

    prompts, miss_list, raw_list, conf_list = walker.generate_prompts_only(
        question=args.texts,
        n_trials=args.n_trials,
    )
    all_results_prompt = []
    for prompt in prompts:
        all_results = []
        # time.sleep(5)
        for model_name, api_key in zip(model_str_list, api_str_list):
            print(f"\n========== Running: {model_name} ==========")
            # time.sleep(5)
            llm = LLMCaller(model=model_name, api=api_key)
            evaluator = Evaluator(graph, walker, scorer, checker, llm)
            formula = evaluator.generate_formula_and_accuracy(
                prompt=prompt,
                n_trials=args.n_trials,
            )

            model_formulas = []
            model_valids = []

            is_valid = checker.is_valid(formula)
            model_formulas.append(formula)
            model_valids.append(is_valid)

            all_results.append({
                "model": model_name,
                "formulas": model_formulas,
                "valids": model_valids,
                "prompt": prompt,
                "miss": miss_list,
                "raw": raw_list,
                "conf": conf_list,
                "acc": "?"
            })
        all_results_prompt.append(all_results)

    # ================== Write CSV ==================
    csv_file = args.save_csv_file
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["texts", "n_paths", "n_trials", "prompt", "miss", "raw", "confidence"]
            for result in all_results:
                model = result["model"]
                header.extend([
                    f"{model}_formula",
                    f"{model}_is_valid",
                    f"{model}_accuracy"
                ])
            writer.writerow(header)
        for all_results in all_results_prompt:
            # Header (only once)

            num_paths = len(all_results[0]["formulas"])

            for i in range(num_paths):
                # Common information (take prompt etc. from the first model only)

                # print("------------num_paths------------",prompt)
                base = all_results[0]
                row = [
                    " ".join(args.texts),
                    args.n_paths,
                    args.n_trials,
                    base["prompt"],
                    base["miss"][i],
                    base["raw"][i],
                    base["conf"][i]
                ]

                # Add results for each model
                for result in all_results:
                    row.extend([
                        result["formulas"][i],
                        result["valids"][i],
                        result["acc"][i]
                    ])

                writer.writerow(row)


if __name__ == "__main__":
    main()
