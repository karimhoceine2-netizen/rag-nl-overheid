"""Review draft questions one by one in the terminal.

    uv run python -m evals.review_questions evals/gate_questions.jsonl

Keys: j = approve (status "reviewed"), n = reject (moved to <file>.rejected.jsonl),
s = skip for now, q = stop. The file is saved after every answer, so you can stop any time.
"""

import sys
from collections.abc import Callable
from pathlib import Path

from evals.questions import Question, load_questions


def save(path: Path, questions: list[Question]) -> None:
    path.write_text("".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8")


def show(q: Question, position: str) -> str:
    lines = [f"\n[{position}] {q.id}", f"Vraag:  {q.question}"]
    if q.answerable:
        lines += [f"Bron:   {q.expected_url}", f"Citaat: {q.evidence}"]
    else:
        lines.append("Label:  ONBEANTWOORDBAAR met dit corpus")
    if q.notes:
        lines.append(f"Notitie: {q.notes}")
    return "\n".join(lines)


def review(path: Path, ask: Callable[[str], str] = input) -> dict[str, int]:
    questions = load_questions(path)
    rejected_path = path.with_suffix(".rejected.jsonl")
    drafts = [q.id for q in questions if q.status == "draft"]
    counts = {"approved": 0, "rejected": 0, "skipped": 0}
    for n, qid in enumerate(drafts, 1):
        q = next(x for x in questions if x.id == qid)
        print(show(q, f"{n}/{len(drafts)}"))
        answer = ""
        while answer not in {"j", "n", "s", "q"}:
            answer = ask("Klopt het label? [j]a / [n]ee / [s]kip / [q]uit: ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            counts["skipped"] += 1
            continue
        if answer == "j":
            q.status = "reviewed"
            counts["approved"] += 1
        else:
            questions.remove(q)
            with rejected_path.open("a", encoding="utf-8") as f:
                f.write(q.model_dump_json() + "\n")
            counts["rejected"] += 1
        save(path, questions)
    return counts


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "evals/gate_questions.jsonl")
    counts = review(path)
    left = sum(q.status == "draft" for q in load_questions(path))
    print(
        f"\nGoedgekeurd {counts['approved']}, afgekeurd {counts['rejected']}, "
        f"overgeslagen {counts['skipped']}. Nog {left} concepten over."
    )


if __name__ == "__main__":
    main()
