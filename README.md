# Multi-Agent Error Propagation

How do hallucinations and errors spread through multi-agent LLM systems, and which quality-control mechanisms stop them?

**Status:** early pilot. The code runs end to end; results are not yet published.

**Author:** Asmat, MS Computer Science (AI specialization), CECOS University, Peshawar, Pakistan
**Supervisor:** Dr. Maryam, Head of the Computer Science Department, CECOS University

## Motivation

Multi-agent debate and critic-reviser loops are widely assumed to make LLM outputs more reliable. Few studies test this against ground truth by planting a known error and tracking what happens to it. Related work reports that iteration can also hurt: refinement rounds can trade away relevance, and judge models can flag or miss hallucinations unreliably. This project measures the propagation directly.

## Research questions

1. **Propagation:** when an agent receives a false claim, how often does it adopt it, and how often does a critic catch it?
2. **Iteration:** does adding critique-revision rounds help, plateau, or introduce new errors?
3. **Diversity:** do heterogeneous agent teams (different models) correct errors better than homogeneous ones?
4. **Verification:** which verifier design catches errors best: LLM judge, retrieval grounding, or programmatic checks?

## Design (pilot)

A solver answers a question. In the `injected` condition it also sees an unverified note containing a known-false answer. A critic reviews the solver's answer and the solver revises, for R rounds. Every message is logged.

| Condition  | Solver input                          |
|------------|---------------------------------------|
| `clean`    | question only (control)               |
| `injected` | question + unverified false-answer note |

### Metrics

- **accuracy@r:** share of runs correct after round r
- **adoption@r:** (injected) share whose answer matches the planted false answer
- **correction rate:** among runs that adopted the false answer at round 0, the share that no longer hold it at the final round
- **late adoption:** among runs that were clean at round 0, the share that hold the false answer at the end (the loop introduced the error)

## Quickstart

```bash
pip install -r requirements.txt

# 1. Dry run with a fake model (free, tests the pipeline only)
python pilot.py --backend mock --seeds 3 --rounds 2 --out results/mock.jsonl
python analyze.py results/mock.jsonl

# 2. Open model through any OpenAI-compatible server (e.g. Ollama)
python pilot.py --backend openai --model qwen2.5:7b --limit 5

# 3. Claude (set a current model name from docs.claude.com)
export ANTHROPIC_API_KEY=...
export CLAUDE_MODEL=...
python pilot.py --backend anthropic --limit 5
```

Runs are **resumable**: finished (task, condition, seed) runs are skipped when you restart, and API errors are retried with exponential backoff. The script prints the estimated number of model calls before it starts, so you can budget tokens first.

## Important limitations

- `tasks.jsonl` is a **10-question smoke test**. Strong models will rarely be fooled by easy facts, so real experiments need harder items (multi-hop QA, domain questions) and 100+ tasks.
- Answer matching is string-based and works for short answers only.
- The mock backend produces meaningless numbers; it only tests the code.
- Results from a single model and a few seeds are pilots, not conclusions.

## Roadmap

1. Harder benchmark with ground truth, 100+ tasks, 5 seeds, confidence intervals
2. Homogeneous vs. heterogeneous teams (open models + Claude as judge and as one agent)
3. Iteration depth 1 to 5 rounds, tracking late adoption
4. Verifier comparison: LLM judge vs. retrieval grounding vs. programmatic checks (precision and recall)
5. Human-facing study of trust calibration in AI-generated responses

## Repository layout

```
pilot.py          experiment runner (solver-critic loop, error injection, logging)
analyze.py        summary metrics
tasks.jsonl       sample tasks (replace with a real benchmark)
results/          run logs (JSONL, one record per run with full message trace)
```

## License

Add a license before publishing (MIT is a common choice for research code).
