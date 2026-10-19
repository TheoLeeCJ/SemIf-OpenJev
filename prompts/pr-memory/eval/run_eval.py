"""Run a pr-memory prompt over synthetic_conversations.jsonl with the OpenAI API.

    pip install openai jinja2
    export OPENAI_API_KEY=sk-...
    python run_eval.py --dry-run                         # check the dataset, print one prompt
    python run_eval.py --out runs/v5-1.jsonl             # v5 prompt, validated and scored
    python run_eval.py --prompt v4 --out runs/v4-1.jsonl # v4 prompt, raw outputs only
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import jinja2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v5"))
from extract_validator import OutputFormatError, period_end, user_turns, validate_output  # noqa: E402

DATASET = HERE / "synthetic_conversations.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def render_conversation(turns):
    return "\n".join(f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['text']}" for t in turns)


def render_prompt(version, record):
    template = jinja2.Template((HERE.parent / version / "pr-template.jinja").read_text())
    conversation = render_conversation(record["turns"])
    if version == "v4":
        facts = "\n".join(f"- {item['fact']}" for item in record["profile_items"])
        return template.render(profile_facts=facts, conversation=conversation)
    return template.render(session_date=record["session_date"], profile_items=record["profile_items"],
                           episodic_items=record["episodic_items"], intent_items=record["intent_items"],
                           conversation=conversation)


def check_dataset(records):
    """Sanity-check the reference labels; returns a list of problems."""
    problems = []
    for r in records:
        today = date.fromisoformat(r["session_date"])
        turns = [" ".join(t.split()) for t in user_turns(render_conversation(r["turns"]))]
        items = {i["id"]: kind for kind in ("profile", "episodic", "intent") for i in r[f"{kind}_items"]}
        for e in r["expected"]:
            if not any(" ".join(e["evidence"].split()) in t for t in turns):
                problems.append(f"{r['id']}: evidence not in a User turn: {e['evidence']!r}")
            if e["target_id"] is not None and items.get(e["target_id"]) != e["memory_type"]:
                problems.append(f"{r['id']}: bad target {e['target_id']}")
            if e.get("anchor_date") and date.fromisoformat(e["anchor_date"]) > today:
                problems.append(f"{r['id']}: anchor_date in the future")
            if e.get("due_date") and period_end(date.fromisoformat(e["due_date"]), e["due_granularity"]) < today:
                problems.append(f"{r['id']}: due_date in the past")
    return problems


def call_model(client, model, prompt, temperature):
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "response_format": {"type": "json_object"}}
    if temperature is not None:
        kwargs["temperature"] = temperature
    for attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as exc:  # network errors, rate limits
            if attempt == 2:
                return f"__API_ERROR__ {exc}"


def key(action):
    return action["action"], action["memory_type"], action.get("target_id")


def score(record, output_text):
    """Validate one v5 output and match its actions to the reference labels."""
    result = {"id": record["id"], "tests": record["tests"], "raw": output_text}
    try:
        validated = validate_output(output_text, session_date=record["session_date"],
                                    conversation=render_conversation(record["turns"]),
                                    profile_items=record["profile_items"],
                                    episodic_items=record["episodic_items"],
                                    intent_items=record["intent_items"])
    except OutputFormatError as exc:
        result.update(format_error=str(exc), matched=[], missed=record["expected"], extra=[], rejects=[])
        return result

    predicted = list(validated.valid)
    matched, missed = [], []
    for expected in record["expected"]:
        hit = next((p for p in predicted if key(p) == key(expected)), None)
        if hit is None:
            missed.append(expected)
        else:
            predicted.remove(hit)
            matched.append({"expected": expected, "predicted": hit})
    result.update(format_error=None, matched=matched, missed=missed, extra=predicted,
                  rejects=[{"action": r.action, "errors": r.errors} for r in validated.rejects])
    return result


def summarize(results):
    tp = sum(len(r["matched"]) for r in results)
    fn = sum(len(r["missed"]) for r in results)
    fp = sum(len(r["extra"]) for r in results)
    by_action = Counter()
    for r in results:
        for m in r["matched"]:
            by_action[m["expected"]["action"], "hit"] += 1
        for m in r["missed"]:
            by_action[m["action"], "miss"] += 1
        for m in r["extra"]:
            by_action[m["action"], "extra"] += 1

    high_total = high_hit = date_total = date_hit = 0
    for r in results:
        for m in r["matched"]:
            exp, pred = m["expected"], m["predicted"]
            if exp.get("privacy_level") == "HIGH":
                high_total += 1
                high_hit += pred.get("privacy_level") == "HIGH"
            for field, gran in (("anchor_date", "granularity"), ("due_date", "due_granularity")):
                if field in exp:
                    date_total += 1
                    date_hit += (pred.get(field), pred.get(gran)) == (exp[field], exp.get(gran))

    noop = [r for r in results if not r["matched"] and not r["missed"]]
    return {
        "records": len(results),
        "format_errors": sum(r["format_error"] is not None for r in results),
        "rejected_actions": sum(len(r["rejects"]) for r in results),
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "privacy_high_recall": f"{high_hit}/{high_total}",
        "date_accuracy": f"{date_hit}/{date_total}",
        "noop_records_with_extra_actions": f"{sum(bool(r['extra']) for r in noop)}/{len(noop)}",
        "by_action": {f"{a}:{k}": n for (a, k), n in sorted(by_action.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=["v4", "v5"], default="v5")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="pass -1 to omit (some reasoning models reject it)")
    parser.add_argument("--data", default=str(DATASET))
    parser.add_argument("--out", help="results JSONL; must not exist yet")
    parser.add_argument("--ids", nargs="*", help="only run these record ids")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load(args.data)
    if args.ids:
        records = [r for r in records if r["id"] in set(args.ids)]
    problems = check_dataset(records)
    for p in problems:
        print("DATASET:", p)

    if args.dry_run:
        print(render_prompt(args.prompt, records[0]))
        print(f"\n{len(records)} records, {len(problems)} dataset problems")
        return
    if not args.out:
        parser.error("--out is required unless --dry-run")
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} already exists; choose a new path")
    out.parent.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    temperature = None if args.temperature < 0 else args.temperature

    def run_one(record):
        text = call_model(client, args.model, render_prompt(args.prompt, record), temperature)
        print(f"done {record['id']}", flush=True)
        if args.prompt == "v5":
            return score(record, text)
        return {"id": record["id"], "tests": record["tests"], "raw": text}

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_one, records))

    with open(out, "x") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    if args.prompt == "v5":
        summary = summarize(results)
        summary.update(model=args.model, prompt=args.prompt)
        Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
