import warnings
import time
import os
import sys
import asyncio
import traceback

import pandas as pd
from dotenv import load_dotenv
from transformers import logging as hf_logging

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

"""Script to run RAGAS-based evaluation for a single RFP query/response.

The script:
- loads environment variables and data
- runs the RAG chain to get an answer and sources
- truncates long texts to stay within model token limits
- evaluates multiple RAGAS metrics asynchronously
- reports scores and total evaluation time
"""

# ---------------------------------------------------------------------------
# Configuration and environment
# ---------------------------------------------------------------------------

# Load environment variables from .env file (project-local)
load_dotenv(os.path.join(os.path.dirname(__file__), "../..", "src", ".env"))

# Ensure project root is on sys.path so `src.*` imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Suppress only Hugging Face Hub unauthenticated request warnings
warnings.filterwarnings("ignore")

# Quiet transformer library logging (HF models etc.)
hf_logging.set_verbosity_error()

from src.core.rag_service import build_custom_rag_chain, setup_rag_chain
from src.core.prompt_templates import DEFAULT_CUSTOM_RAG_PROMPT_2
from src.core.vectorstore import get_db
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
    AnswerRelevancy,
    FactualCorrectness,
    SemanticSimilarity,
    BleuScore,
    RougeScore,
    ExactMatch,
)


# ---------------------------------------------------------------------------
# Input parameters (configure these as needed)
# ---------------------------------------------------------------------------

# Full CSV path passed directly to pandas.read_csv
INPUT_CSV_PATH: str = "src/eval/input/input2.csv"

# Maximum number of rows from the CSV to evaluate.
# Use None (or <= 0) to process all rows.
MAX_ROWS: int | None = None


# Names of metrics to actually run. Comment out or remove entries to
# skip expensive/less relevant metrics without changing the core code.
ENABLED_METRICS: list[str] = [
    "faithfulness",
    # "context_precision",
    "context_recall",
    # "context_entity_recall",
    # "noise_sensitivity",
    "answer_relevancy",
    "factual_correctness",
    "semantic_similarity",
    "bleu",
    # "rouge",
    # "exact_match",
]


def load_eval_rows() -> pd.DataFrame:
    """Load all rows from the evaluation CSV and return a dataframe.

    The dataframe contains at least the columns ``Prompt`` and ``Response``.
    """

    df = pd.read_csv(INPUT_CSV_PATH, header=0)
    df = df[["Prompt", "Response"]]
    if MAX_ROWS is not None and MAX_ROWS > 0:
        df = df.head(MAX_ROWS)
    return df


def run_rag(query: str, rag_chain) -> tuple[str, list]:
    """Run the RAG chain for a single query and return (answer, sources).

    The rag_chain is constructed once in main() and reused for all rows.
    """

    result = rag_chain.invoke(query)
    answer_val = result.get("result", "")
    sources_val = result.get("source_documents", [])
    return answer_val, sources_val


def build_llm_and_embeddings():
    """Initialise OpenAI chat model and embedding model for RAGAS metrics."""

    import openai

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = openai.AsyncOpenAI(api_key=openai_api_key)

    # Chat model and embedding model names (from .env)
    # Defaults are kept for convenience if env vars are not set.
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4-1106-preview")  # GPT-4.1
    emb_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # Increase max_tokens so structured outputs are less likely to be
    # truncated by the default 1024-token completion limit inside Ragas.
    llm_model = llm_factory(
        model=openai_model,
        provider="openai",
        client=openai_client,
        max_tokens=4096,
    )
    emb = embedding_factory("openai", model=emb_model, client=openai_client)

    return llm_model, emb


def build_metrics(llm_model, emb):
    """Construct all metric objects used in the evaluation."""

    faithfulness = Faithfulness(llm=llm_model)
    ctx_precision = ContextPrecision(llm=llm_model)
    ctx_recall = ContextRecall(llm=llm_model)
    ctx_entity_recall = ContextEntityRecall(llm=llm_model)
    noise_sens = NoiseSensitivity(llm=llm_model)
    ans_relevancy = AnswerRelevancy(llm=llm_model, embeddings=emb)
    factual = FactualCorrectness(llm=llm_model)
    semantic_sim = SemanticSimilarity(llm=llm_model, embeddings=emb)
    bleu = BleuScore()
    rouge = RougeScore()
    exact = ExactMatch()

    return (
        faithfulness,
        ctx_precision,
        ctx_recall,
        ctx_entity_recall,
        noise_sens,
        ans_relevancy,
        factual,
        semantic_sim,
        bleu,
        rouge,
        exact,
    )


async def run_metrics(
    query: str,
    exp_ans: str,
    answer: str,
    sources: list,
    metrics,
    enabled_metrics: list[str],
):
    """Run selected metrics asynchronously over the given query/answer/context.

    All metric objects are passed in via ``metrics``, but only those whose
    names are present in ``enabled_metrics`` are actually executed.
    """

    (
        faithfulness_metric,
        context_precision_metric,
        context_recall_metric,
        context_entity_recall_metric,
        noise_sensitivity_metric,
        answer_relevancy_metric,
        factual_correctness_metric,
        semantic_similarity_metric,
        bleu_metric,
        rouge_metric,
        exact_match_metric,
    ) = metrics

    # Use full texts/contexts without additional truncation for simplicity
    retrieved_texts = [doc.page_content for doc in sources]
    capped_answer = answer
    capped_reference = exp_ans

    enabled = set(enabled_metrics)

    scores: dict[str, float] = {}

    # Core RAGAS metrics
    if "faithfulness" in enabled:
        scores["faithfulness"] = await faithfulness_metric.ascore(
            user_input=query,
            retrieved_contexts=retrieved_texts,
            response=capped_answer,
        )

    if "context_precision" in enabled:
        scores["context_precision"] = await context_precision_metric.ascore(
            user_input=query,
            retrieved_contexts=retrieved_texts,
            reference=capped_reference,
        )

    if "context_recall" in enabled:
        scores["context_recall"] = await context_recall_metric.ascore(
            user_input=query,
            retrieved_contexts=retrieved_texts,
            reference=capped_reference,
        )

    if "context_entity_recall" in enabled:
        scores["context_entity_recall"] = await context_entity_recall_metric.ascore(
            retrieved_contexts=retrieved_texts,
            reference=capped_reference,
        )

    if "noise_sensitivity" in enabled:
        scores["noise_sensitivity"] = await noise_sensitivity_metric.ascore(
            user_input=query,
            retrieved_contexts=retrieved_texts,
            response=capped_answer,
            reference=capped_reference,
        )

    if "answer_relevancy" in enabled:
        scores["answer_relevancy"] = await answer_relevancy_metric.ascore(
            user_input=query,
            response=capped_answer,
        )

    if "factual_correctness" in enabled:
        scores["factual_correctness"] = await factual_correctness_metric.ascore(
            response=capped_answer,
            reference=capped_reference,
        )

    if "semantic_similarity" in enabled:
        scores["semantic_similarity"] = await semantic_similarity_metric.ascore(
            response=answer,
            reference=exp_ans,
        )

    # Text similarity / overlap metrics
    if "bleu" in enabled:
        scores["bleu"] = await bleu_metric.ascore(
            response=answer,
            reference=exp_ans,
        )

    if "rouge" in enabled:
        scores["rouge"] = await rouge_metric.ascore(
            response=answer,
            reference=exp_ans,
        )

    if "exact_match" in enabled:
        scores["exact_match"] = await exact_match_metric.ascore(
            response=answer,
            reference=exp_ans,
        )

    return scores


def main() -> None:
    """Entry point for running the evaluation on all rows in input.csv.

    If evaluation fails for a specific row (for example due to token limits),
    the error is recorded for that row and processing continues. Results for
    all successfully processed rows are always written to the CSV.
    """

    # Load all evaluation rows
    df = load_eval_rows()

    # Build RAG components once (reused across rows), using custom prompt
    db = get_db("RFP_Ryapte")
    rag_chain = build_custom_rag_chain(
        db,
        5,
        prompt_template=DEFAULT_CUSTOM_RAG_PROMPT_2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # Build LLM, embeddings, and metrics once (reused across rows)
    llm_model, emb = build_llm_and_embeddings()
    metrics = build_metrics(llm_model, emb)

    # For aggregating per-row results to save as CSV
    results: list[dict] = []

    overall_start = time.time()

    # Mapping from metric keys to human-readable column labels
    name_to_label = {
        "faithfulness": "Faithfulness",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "context_entity_recall": "Context Entity Recall",
        "noise_sensitivity": "Noise Sensitivity",
        "answer_relevancy": "Answer Relevancy",
        "factual_correctness": "Factual Correctness",
        "semantic_similarity": "Semantic Similarity",
        "bleu": "BLEU",
        "rouge": "ROUGE",
        "exact_match": "Exact Match",
    }

    for idx, row in df.iterrows():
        query = row["Prompt"]
        exp_ans = row["Response"]

        print(f"\n===== Row {idx} =====")
        print("Query:", query)

        # Default values in case something fails before they are set
        answer = ""
        sources: list = []
        row_start = time.time()

        try:
            # Run RAG for this row using shared chain
            answer, sources = run_rag(query, rag_chain)

            print("Expected Answer:", exp_ans[:100])
            print("Answer:", answer[:100])
            print("Sources Sizes:", [len(doc.page_content) for doc in sources])
            print("-" * 50)

            # Run selected metrics
            scores = asyncio.run(
                run_metrics(query, exp_ans, answer, sources, metrics, ENABLED_METRICS)
            )

            row_elapsed = time.time() - row_start

            print(f"Row {idx} evaluation time: {row_elapsed:.2f} sec")

            # Build result record for this row
            record: dict = {
                "row_index": idx,
                "Prompt": query,
                "Response": exp_ans,
                "rag_answer": answer,
                "eval_time_sec": round(row_elapsed, 2),
            }

            # Store only the numeric score value (unwrap MetricResult objects)
            for key in ENABLED_METRICS:
                if key in scores:
                    value = scores[key]
                    if hasattr(value, "value"):
                        value = value.value
                    record[name_to_label[key]] = value
                    print(f"{name_to_label[key]} score:", value)

            results.append(record)

        except Exception as e_row:  # noqa: BLE001 - per-row error reporting
            # Log the error but continue with remaining rows
            print(f"Error evaluating row {idx}:", repr(e_row))
            traceback.print_exc()

            row_elapsed = time.time() - row_start
            record = {
                "row_index": idx,
                "Prompt": query,
                "Response": exp_ans,
                "rag_answer": answer,
                "eval_time_sec": round(row_elapsed, 2),
                "error": repr(e_row),
            }
            results.append(record)

    # After all rows (successful or with errors), save results
    overall_elapsed = time.time() - overall_start
    minutes = int(overall_elapsed // 60)
    seconds = overall_elapsed % 60

    output_dir = os.path.join(os.path.dirname(__file__), "..", "eval", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "eval_results.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print(f"\nSaved evaluation results to: {output_csv}")
    print(f"Total metrics evaluation time (all rows): {minutes} min {seconds:.1f} sec")


if __name__ == "__main__":
    main()
