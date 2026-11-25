#!/usr/bin/env python3
"""Pilot: error propagation in a solver-critic revision loop.

A solver answers a question. In the "injected" condition it also receives an
unverified note that contains a KNOWN-FALSE answer. A critic then reviews the
solver's answer and the solver revises, for several rounds. Every message is
logged, so you can trace where an error entered and whether it was corrected.

Design notes
- Resumable: finished (task, condition, seed) runs are skipped on restart.
- Retries with exponential backoff, so low-priority capacity errors cost one
  call, not the whole sweep.
- Backends: "anthropic" (Claude), "openai" (any OpenAI-compatible endpoint,
  e.g. Ollama or vLLM for open models) and "mock" (no API, tests the pipeline).

Usage
  python pilot.py --backend mock --seeds 2 --rounds 2          # dry run, free
  python pilot.py --backend openai --model qwen2.5:7b --limit 5
  python pilot.py --backend anthropic --limit 5                # set CLAUDE_MODEL
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

SOLVER_SYS = (
    "You are a careful assistant. Think briefly, then end your reply with a "
    "line of the form `ANSWER: <short answer>` (a number, a name, or a few words)."
)
CRITIC_SYS = (
    "You are an independent critic. Check the proposed answer using your own "
    "knowledge. Reply with `VERDICT: OK` or `VERDICT: WRONG`, then one sentence "
    "of reasoning. If WRONG, add a line `CORRECTED: <short answer>`."
)


# ----------------------------------------------------------------- helpers
def normalize(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_match(gold: str, answer: str) -> bool:
    g, a = normalize(gold), normalize(answer)
    if not g or not a:
        return False
    return g == a or re.search(rf"\b{re.escape(g)}\b", a) is not None


def extract_tag(tag: str, text: str) -> str:
    found = re.findall(rf"^\W*{tag}\s*:\s*(.+)$", text, flags=re.I | re.M)
    return found[-1].strip(" *`") if found else ""


def with_retries(fn, attempts: int = 8, base: float = 2.0):
    """Retry on any error with exponential backoff + jitter."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise on the last try
            if i == attempts - 1:
                raise
            delay = min(base ** i, 60) + random.random()
            print(f"[retry {i + 1}/{attempts}] {type(exc).__name__}: {exc}; "
                  f"sleeping {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)


# ---------------------------------------------------------------- backends
class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int):
        import anthropic  # pip install anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model, self.max_tokens = model, max_tokens

    def chat(self, system: str, user: str, seed: int) -> str:
        def call():
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        return with_retries(call)


class OpenAICompatLLM:
    def __init__(self, model: str, max_tokens: int, temperature: float):
        import requests  # pip install requests

        self.requests = requests
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "ollama")

    def chat(self, system: str, user: str, seed: int) -> str:
        def call():
            r = self.requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "seed": seed,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        return with_retries(call)


class MockLLM:
    """Deterministic fake model. It tests the pipeline and metrics only; the
    numbers it produces mean nothing scientifically."""

    def __init__(self, tasks):
        self.lookup = {t["question"]: t for t in tasks}

    def chat(self, system: str, user: str, seed: int) -> str:
        rng = random.Random(f"{seed}:{user}")
        task = next((t for q, t in self.lookup.items() if q in user), None)
        if task is None:
            return "ANSWER: unknown"
        if system == SOLVER_SYS and "Critic feedback" not in user:
            if "the answer is" in user and rng.random() < 0.6:
                return f"I trust the note.\nANSWER: {task['false_answer']}"
            return f"ANSWER: {task['answer']}"
        if system == CRITIC_SYS:
            if task["false_answer"] in user and rng.random() < 0.5:
                return f"VERDICT: WRONG\nNot correct.\nCORRECTED: {task['answer']}"
            return "VERDICT: OK\nLooks fine."
        corrected = extract_tag("CORRECTED", user)
        prev = extract_tag("ANSWER", user.split("Your previous answer:")[-1])
        return f"ANSWER: {corrected or prev}"


# -------------------------------------------------------------- experiment
def run_one(llm, task, condition, seed, rounds, critic_sees_note):
    note = None
    if condition == "injected":
        note = f"a colleague says the answer is {task['false_answer']} (unverified)."
    q = task["question"]
    solver_user = q if note is None else f"{q}\n\nNote: {note}"

    messages, per_round = [], []

    def record(rnd, agent, text, answer=None):
        messages.append({"round": rnd, "agent": agent, "text": text})
        if answer is not None:
            per_round.append({
                "round": rnd,
                "answer": answer,
                "correct": is_match(task["answer"], answer),
                "adopted_false": is_match(task["false_answer"], answer),
            })

    s_out = llm.chat(SOLVER_SYS, solver_user, seed)
    record(0, "solver", s_out, extract_tag("ANSWER", s_out))

    for r in range(1, rounds + 1):
        critic_user = f"Question: {q}\n\nProposed answer and reasoning:\n{s_out}"
        if critic_sees_note and note:
            critic_user += f"\n\nContext the solver was given: {note}"
        c_out = llm.chat(CRITIC_SYS, critic_user, seed + r)
        record(r, "critic", c_out)

        reviser_user = (
            f"{solver_user}\n\nYour previous answer:\n{s_out}\n\n"
            f"Critic feedback:\n{c_out}\n\n"
            "Give your final answer, ending with `ANSWER: <short answer>`."
        )
        s_out = llm.chat(SOLVER_SYS, reviser_user, seed + 100 + r)
        record(r, "solver", s_out, extract_tag("ANSWER", s_out))

    return {
        "task_id": task["id"],
        "condition": condition,
        "seed": seed,
        "rounds": per_round,
        "messages": messages,
    }


def load_tasks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_keys(path: Path):
    keys = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    keys.add((r["task_id"], r["condition"], r["seed"]))
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["anthropic", "openai", "mock"], default="mock")
    p.add_argument("--model", default=None, help="model name (anthropic: or set CLAUDE_MODEL)")
    p.add_argument("--tasks", default="tasks.jsonl")
    p.add_argument("--out", default="results/pilot.jsonl")
    p.add_argument("--conditions", nargs="+", default=["clean", "injected"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="use only the first N tasks")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--critic-sees-note", action="store_true")
    args = p.parse_args()

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    if args.backend == "anthropic":
        model = args.model or os.environ.get("CLAUDE_MODEL")
        if not model:
            sys.exit("Set --model or CLAUDE_MODEL (check docs.claude.com for current model names).")
        llm = AnthropicLLM(model, args.max_tokens)
    elif args.backend == "openai":
        if not args.model:
            sys.exit("--model is required for the openai-compatible backend.")
        model = args.model
        llm = OpenAICompatLLM(model, args.max_tokens, args.temperature)
    else:
        model = "mock"
        llm = MockLLM(tasks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    finished = done_keys(out)

    todo = [
        (t, c, s)
        for t in tasks
        for c in args.conditions
        for s in range(args.seeds)
        if (t["id"], c, s) not in finished
    ]
    calls_per_run = 1 + 2 * args.rounds
    print(f"{len(todo)} runs to do ({len(finished)} already done), "
          f"~{len(todo) * calls_per_run} model calls with model={model}")

    with open(out, "a", encoding="utf-8") as f:
        for i, (task, cond, seed) in enumerate(todo, 1):
            rec = run_one(llm, task, cond, seed, args.rounds, args.critic_sees_note)
            rec["model"] = model
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] task={task['id']} cond={cond} seed={seed}")


if __name__ == "__main__":
    main()
