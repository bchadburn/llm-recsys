"""
Reads the 4 experiment log files and generates overnight_report.md.
Each experiment prints EXPERIMENT N SUMMARY and EXPERIMENT N COMPLETE markers
that this script uses to extract the relevant sections.
"""

from pathlib import Path
from datetime import datetime

LOGS = {
    1: Path('logs/exp1_enrichment.log'),
    2: Path('logs/exp2_narration.log'),
    3: Path('logs/exp3_reranker.log'),
    4: Path('logs/exp4_synthetic.log'),
}
REPORT = Path('overnight_report.md')

EXP_TITLES = {
    1: "LLM Item Enrichment",
    2: "Zero-shot User Narration",
    3: "Contextual LLM Reranker",
    4: "Synthetic Context Injection",
}
EXP_PATTERNS = {
    1: "Offline LLM-generated content → richer embeddings → better retrieval",
    2: "User features → text profile → item index query (no trained tower)",
    3: "FAISS candidates + session context → Claude reranking with reasoning",
    4: "Synthetic occasions injected into training → context-aware two-tower",
}


def read_log(path: Path) -> str:
    if not path.exists():
        return f"[LOG NOT FOUND: {path}]"
    return path.read_text()


def check_completed(log: str, exp_num: int) -> bool:
    return f"EXPERIMENT {exp_num} COMPLETE" in log


def extract_between(log: str, start_marker: str, end_markers: list) -> str:
    idx = log.find(start_marker)
    if idx == -1:
        return "[section not found in log]"
    end = len(log)
    for marker in end_markers:
        pos = log.find(marker, idx + len(start_marker))
        if pos != -1:
            end = min(end, pos)
    return log[idx:end].strip()


def extract_summary(log: str, exp_num: int) -> str:
    return extract_between(
        log,
        f"EXPERIMENT {exp_num} SUMMARY",
        [f"EXPERIMENT {exp_num} COMPLETE"],
    )


def extract_first_user_block(log: str) -> str:
    """For Exp 3: extract first user's results block."""
    idx = log.find("User ")
    if idx == -1:
        return "[no user block found]"
    # Find second User block as end marker
    end = log.find("User ", idx + 10)
    return log[idx:end if end != -1 else idx + 3000].strip()


def generate():
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    logs = {n: read_log(path) for n, path in LOGS.items()}

    statuses = {
        n: "✅ COMPLETE" if check_completed(logs[n], n) else "❌ FAILED/INCOMPLETE"
        for n in range(1, 5)
    }

    lines = [
        "# Overnight LLM Experiments — Results",
        f"**Generated:** {now}",
        "",
        "## Status",
        "",
        "| Experiment | Status |",
        "|---|---|",
    ]
    for n in range(1, 5):
        lines.append(f"| Exp {n}: {EXP_TITLES[n]} | {statuses[n]} |")

    lines += [
        "",
        "---",
        "",
    ]

    for n in range(1, 5):
        lines += [
            f"## Exp {n}: {EXP_TITLES[n]}",
            f"*Pattern: {EXP_PATTERNS[n]}*",
            "",
        ]
        if n == 3:
            lines += [
                "### Sample output (see logs/exp3_reranker.log for full results)",
                "",
                "```",
                extract_first_user_block(logs[3]),
                "```",
            ]
        else:
            lines += [
                "```",
                extract_summary(logs[n], n),
                "```",
            ]
        lines += ["", "---", ""]

    lines += [
        "## Full Logs",
        "",
        "| Log | Path |",
        "|---|---|",
        "| Exp 1 | `logs/exp1_enrichment.log` |",
        "| Exp 2 | `logs/exp2_narration.log` |",
        "| Exp 3 | `logs/exp3_reranker.log` |",
        "| Exp 4 | `logs/exp4_synthetic.log` |",
        "| Orchestration | `logs/orchestration.log` |",
    ]

    REPORT.write_text("\n".join(lines))
    print(f"Report written to {REPORT}")
    for n, status in statuses.items():
        print(f"  Exp {n} ({EXP_TITLES[n]}): {status}")


if __name__ == '__main__':
    generate()
