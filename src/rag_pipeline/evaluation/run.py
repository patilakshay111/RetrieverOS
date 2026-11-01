# run.py: evaluates all chunkers against the golden set and writes a metric comparison report

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from rag_pipeline.config import Settings
from rag_pipeline.service import RAGService


# Load and validate the hand-authored question and expected-source data.
def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation dataset must be a non-empty JSON list.")
    required = {"id", "question", "reference_answer", "expected_sources", "answerable", "tags"}
    for case in cases:
        if not required.issubset(case):
            raise ValueError(f"Evaluation case is missing required fields: {case!r}")
    return cases


# Run the full golden suite against one isolated index built with a selected chunker.
def evaluate_strategy(
    strategy: str, cases: list[dict], corpus_dir: Path, settings: Settings
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"rag-eval-{strategy}-") as temporary:
        evaluation_settings = replace(settings, data_dir=Path(temporary))
        service = RAGService(evaluation_settings)
        for document in sorted(corpus_dir.glob("*")):
            if document.is_file():
                service.knowledge_base.ingest(document.name, document.read_bytes(), strategy)
        totals: dict[str, list[float]] = defaultdict(list)
        case_results: list[dict] = []
        for case in cases:
            result = service.ask(
                case["question"],
                "hybrid",
                f'eval-{strategy}-{case["id"]}',
            )
            actual_sources = {item.chunk.source for item in result.retrieved_chunks}
            expected_sources = set(case["expected_sources"])
            if expected_sources:
                retrieval_relevance = len(expected_sources & actual_sources) / len(expected_sources)
            else:
                retrieval_relevance = 1.0 if result.insufficient_context else 0.0
            context = "\n\n".join(item.chunk.text for item in result.retrieved_chunks)
            correctness = service.generator.evaluate_correctness(
                case["question"],
                case["reference_answer"] or "The correct behavior is to say that the indexed documents do not answer this question.",
                result.answer,
            )
            faithfulness = service.generator.evaluate_faithfulness(
                case["question"], context, result.answer
            )
            if result.citations:
                citation_accuracy = sum(item.verified for item in result.citations) / len(
                    result.citations
                )
            else:
                citation_accuracy = 1.0 if result.insufficient_context else 0.0
            totals["answer_correctness"].append(correctness.score)
            totals["faithfulness"].append(faithfulness.score)
            totals["retrieval_relevance"].append(retrieval_relevance)
            totals["citation_accuracy"].append(citation_accuracy)
            totals["citation_coverage"].append(
                result.confidence_breakdown["citation_coverage"]
            )
            case_results.append(
                {
                    "id": case["id"],
                    "tags": case["tags"],
                    "answerable": case["answerable"],
                    "answer_correctness": correctness.model_dump(),
                    "faithfulness": faithfulness.model_dump(),
                    "retrieval_relevance": retrieval_relevance,
                    "citation_accuracy": citation_accuracy,
                    "citation_coverage": result.confidence_breakdown["citation_coverage"],
                    "retrieved_sources": sorted(actual_sources),
                    "insufficient_context": result.insufficient_context,
                }
            )
        return {
            "strategy": strategy,
            "question_count": len(cases),
            "metrics": {
                name: sum(values) / len(values) if values else 0.0
                for name, values in totals.items()
            },
            "cases": case_results,
        }


# Compare all three chunking methods and write machine-readable and reviewer-friendly reports.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full hybrid-RAG golden evaluation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/results"),
        help="Directory for JSON results and the Markdown strategy comparison.",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    settings = Settings.from_env()
    cases = load_cases(root / "data" / "evaluation" / "golden_qa.json")
    corpus = root / "data" / "sample_docs"
    strategies = [
        evaluate_strategy(strategy, cases, corpus, settings)
        for strategy in ("fixed", "recursive", "semantic")
    ]
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "data/evaluation/golden_qa.json",
        "corpus": "data/sample_docs",
        "question_count": len(cases),
        "strategies": strategies,
    }
    (output / "chunking-comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metric_names = list(strategies[0]["metrics"])
    lines = [
        "# Chunking Strategy Evaluation",
        "",
        f"Golden questions: {len(cases)}",
        "",
        "| Strategy | " + " | ".join(metric_names) + " |",
        "|---|" + "|".join("---:" for _ in metric_names) + "|",
    ]
    for result in strategies:
        lines.append(
            "| "
            + result["strategy"]
            + " | "
            + " | ".join(f'{result["metrics"][name]:.3f}' for name in metric_names)
            + " |"
        )
    lines.extend(
        [
            "",
            "Metrics are means over the complete golden set. Correctness and faithfulness use "
            "the configured LLM judge; retrieval relevance is expected-source recall; citation "
            "accuracy is the proportion of generated citation checks marked supported. "
            "No scores are reported until this command is run with configured model credentials.",
        ]
    )
    (output / "chunking-comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)}-case comparison to {output}")


# Run the evaluation command when the module is invoked directly.
if __name__ == "__main__":
    main()
