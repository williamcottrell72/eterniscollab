"""
Market Pipeline - End-to-End Question Generation and Capital Allocation

This module provides a complete pipeline that:
1. Generates prediction market questions from recent news
2. Estimates daily trading volume for each question
3. Computes probability estimates for each question
4. Allocates capital proportional to estimated volume
5. Saves results with full metadata for reproducibility

Usage:
    from market_pipeline import run_market_pipeline

    results = run_market_pipeline(
        n_questions=10,
        total_capital=10000,
        model="openai/gpt-4o-mini"
    )
"""

# Standard library
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Third-party
import pandas as pd
import numpy as np

# Local imports
from topic_generator import generate_topics_and_questions
from buzz import estimate_daily_volume
from probability_estimator import get_probability_distribution


def generate_questions(
    n_topics: int,
    k_questions_per_topic: int,
    model: str = "openai/gpt-4o-mini:online",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate prediction market questions from recent news.

    Args:
        n_topics: Number of topics to generate
        k_questions_per_topic: Max questions per topic
        model: OpenRouter model with :online for web search
        api_key: OpenRouter API key (if None, uses environment variable)

    Returns:
        List of dicts with 'topic' and 'question' keys
    """
    print(f"\n{'='*80}")
    print("STEP 1: GENERATING QUESTIONS FROM RECENT NEWS")
    print(f"{'='*80}")
    print(f"Topics: {n_topics}, Questions per topic: {k_questions_per_topic}")
    print(f"Model: {model}\n")

    results = generate_topics_and_questions(
        n_topics=n_topics,
        k_questions=k_questions_per_topic,
        model=model,
        api_key=api_key,
    )

    # Results is a dict mapping topic -> list of question strings
    # Flatten to list of question dicts
    all_questions = []
    for topic, questions_list in results.items():
        for question_text in questions_list:
            all_questions.append(
                {
                    "topic": topic,
                    "question": question_text,
                    # Note: generate_topics_and_questions doesn't return resolution dates
                    # These would need to be extracted from the question or estimated
                    "resolution_date": None,
                    "resolution_criteria": None,
                }
            )

    print(f"✓ Generated {len(all_questions)} questions across {len(results)} topics\n")
    return all_questions


def estimate_volumes(
    questions: List[Dict[str, Any]],
    n_examples: int = 20,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Estimate daily trading volume for each question.

    Args:
        questions: List of question dicts from generate_questions()
        n_examples: Number of historical examples to use
        model: OpenRouter model identifier
        api_key: OpenRouter API key (if None, uses environment variable)

    Returns:
        List of question dicts with 'estimated_daily_volume' added
    """
    print(f"\n{'='*80}")
    print("STEP 2: ESTIMATING DAILY VOLUMES")
    print(f"{'='*80}")
    print(f"Questions: {len(questions)}, Examples per estimate: {n_examples}")
    print(f"Model: {model}\n")

    enriched_questions = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Estimating volume for: {q['question'][:70]}...")

        try:
            result = estimate_daily_volume(
                query=q["question"],
                n_examples=n_examples,
                model=model,
                api_key=api_key,
                print_prompt=False,
            )

            q_enriched = q.copy()
            q_enriched["estimated_daily_volume"] = result["estimated_daily_volume"]
            q_enriched["volume_n_examples"] = result["n_examples"]
            enriched_questions.append(q_enriched)

            print(
                f"  → Estimated daily volume: ${result['estimated_daily_volume']:.2f}"
            )

        except Exception as e:
            print(f"  ✗ Error estimating volume: {e}")
            # Keep question but mark volume as unknown
            q_enriched = q.copy()
            q_enriched["estimated_daily_volume"] = None
            q_enriched["volume_error"] = str(e)
            enriched_questions.append(q_enriched)

    print(f"\n✓ Completed volume estimation for {len(enriched_questions)} questions\n")
    return enriched_questions


async def estimate_probabilities(
    questions: List[Dict[str, Any]],
    n_samples: int = 10,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    reword_temperature: float = 0.7,
    prompt_temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Estimate probability of YES resolution for each question.

    Args:
        questions: List of question dicts from estimate_volumes()
        n_samples: Number of probability samples to collect
        model: OpenRouter model identifier
        api_key: OpenRouter API key (if None, uses environment variable)
        reword_temperature: Temperature for prompt rewording
        prompt_temperature: Temperature for LLM sampling

    Returns:
        List of question dicts with probability estimates added
    """
    print(f"\n{'='*80}")
    print("STEP 3: ESTIMATING PROBABILITIES")
    print(f"{'='*80}")
    print(f"Questions: {len(questions)}, Samples per estimate: {n_samples}")
    print(f"Model: {model}, Reword temperature: {reword_temperature}\n")

    enriched_questions = []
    for i, q in enumerate(questions, 1):
        print(
            f"[{i}/{len(questions)}] Estimating probability for: {q['question'][:70]}..."
        )

        try:
            result = await get_probability_distribution(
                prompt=q["question"],
                n_samples=n_samples,
                reword_temperature=reword_temperature,
                prompt_temperature=prompt_temperature,
                model=model,
                api_key=api_key,
            )

            # Extract probabilities and compute statistics
            probabilities = result["probabilities"]

            q_enriched = q.copy()
            q_enriched["probability_mean"] = np.mean(probabilities)
            q_enriched["probability_median"] = np.median(probabilities)
            q_enriched["probability_std"] = np.std(probabilities)
            q_enriched["probability_samples"] = probabilities
            q_enriched["probability_n_samples"] = len(probabilities)
            enriched_questions.append(q_enriched)

            print(
                f"  → Probability: {q_enriched['probability_mean']:.3f} (std: {q_enriched['probability_std']:.3f})"
            )

        except Exception as e:
            print(f"  ✗ Error estimating probability: {e}")
            # Keep question but mark probability as unknown
            q_enriched = q.copy()
            q_enriched["probability_mean"] = None
            q_enriched["probability_error"] = str(e)
            enriched_questions.append(q_enriched)

    print(
        f"\n✓ Completed probability estimation for {len(enriched_questions)} questions\n"
    )
    return enriched_questions


def allocate_capital(
    questions: List[Dict[str, Any]],
    total_capital: float,
) -> List[Dict[str, Any]]:
    """
    Allocate capital proportional to estimated daily volume.

    Args:
        questions: List of question dicts from estimate_probabilities()
        total_capital: Total capital to allocate across all markets

    Returns:
        List of question dicts with 'allocated_capital' added
    """
    print(f"\n{'='*80}")
    print("STEP 4: ALLOCATING CAPITAL")
    print(f"{'='*80}")
    print(f"Total capital: ${total_capital:,.2f}")
    print(f"Questions: {len(questions)}\n")

    # Filter to questions with valid volume estimates
    questions_with_volume = [
        q
        for q in questions
        if q.get("estimated_daily_volume") is not None
        and q["estimated_daily_volume"] > 0
    ]

    if len(questions_with_volume) == 0:
        print("✗ No questions with valid volume estimates!")
        # Allocate equally if no volume data
        for q in questions:
            q["allocated_capital"] = total_capital / len(questions)
            q["allocation_method"] = "equal (no volume data)"
        return questions

    # Calculate total volume
    total_volume = sum(q["estimated_daily_volume"] for q in questions_with_volume)

    print(f"Questions with volume estimates: {len(questions_with_volume)}")
    print(f"Total estimated daily volume: ${total_volume:,.2f}\n")

    # Allocate proportionally
    enriched_questions = []
    for q in questions:
        q_enriched = q.copy()

        if q.get("estimated_daily_volume") and q["estimated_daily_volume"] > 0:
            # Allocate proportional to volume
            allocation = (q["estimated_daily_volume"] / total_volume) * total_capital
            q_enriched["allocated_capital"] = allocation
            q_enriched["allocation_method"] = "proportional to volume"
            q_enriched["volume_fraction"] = q["estimated_daily_volume"] / total_volume
        else:
            # No allocation if volume estimate failed
            q_enriched["allocated_capital"] = 0.0
            q_enriched["allocation_method"] = "none (no volume estimate)"
            q_enriched["volume_fraction"] = 0.0

        enriched_questions.append(q_enriched)

    # Verify allocation sums to total
    actual_total = sum(q["allocated_capital"] for q in enriched_questions)
    print(f"Total allocated: ${actual_total:,.2f}")
    print(f"✓ Capital allocation complete\n")

    return enriched_questions


def save_results(
    questions: List[Dict[str, Any]],
    pipeline_config: Dict[str, Any],
    output_dir: str = "data/pipelines",
) -> Path:
    """
    Save pipeline results with full metadata for reproducibility.

    Args:
        questions: List of enriched question dicts with all estimates
        pipeline_config: Configuration dict with all pipeline parameters
        output_dir: Base directory for pipeline outputs

    Returns:
        Path to the run directory
    """
    print(f"\n{'='*80}")
    print("STEP 5: SAVING RESULTS")
    print(f"{'='*80}")

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}\n")

    # Save main results as CSV
    df = pd.DataFrame(questions)

    # Separate out list columns for cleaner CSV
    list_columns = ["probability_samples"]
    df_display = df.drop(columns=[col for col in list_columns if col in df.columns])

    csv_path = run_dir / "results.csv"
    df_display.to_csv(csv_path, index=False)
    print(f"✓ Saved results to: {csv_path}")

    # Save full results as JSON (includes all data structures)
    json_path = run_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(questions, f, indent=2, default=str)
    print(f"✓ Saved full results to: {json_path}")

    # Save pipeline configuration
    config_path = run_dir / "pipeline_config.json"
    with open(config_path, "w") as f:
        json.dump(pipeline_config, f, indent=2, default=str)
    print(f"✓ Saved configuration to: {config_path}")

    # Save summary statistics
    summary = {
        "timestamp": timestamp,
        "n_questions": len(questions),
        "n_questions_with_volume": sum(
            1 for q in questions if q.get("estimated_daily_volume") is not None
        ),
        "n_questions_with_probability": sum(
            1 for q in questions if q.get("probability_mean") is not None
        ),
        "total_capital": pipeline_config["total_capital"],
        "total_allocated": sum(q.get("allocated_capital", 0) for q in questions),
        "avg_probability": (
            np.mean(
                [q["probability_mean"] for q in questions if q.get("probability_mean")]
            )
            if any(q.get("probability_mean") for q in questions)
            else None
        ),
        "avg_volume": (
            np.mean(
                [
                    q["estimated_daily_volume"]
                    for q in questions
                    if q.get("estimated_daily_volume")
                ]
            )
            if any(q.get("estimated_daily_volume") for q in questions)
            else None
        ),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✓ Saved summary to: {summary_path}")

    print(f"\n✓ All results saved to: {run_dir}\n")
    return run_dir


def run_market_pipeline(
    n_topics: int = 5,
    k_questions_per_topic: int = 2,
    total_capital: float = 10000.0,
    n_volume_examples: int = 20,
    n_probability_samples: int = 10,
    question_model: str = "openai/gpt-4o-mini:online",
    volume_model: str = "openai/gpt-4o-mini",
    probability_model: str = "openai/gpt-4o-mini",
    reword_temperature: float = 0.7,
    api_key: Optional[str] = None,
    output_dir: str = "data/pipelines",
) -> Dict[str, Any]:
    """
    Run the complete market pipeline from question generation to capital allocation.

    Args:
        n_topics: Number of topics to generate
        k_questions_per_topic: Max questions per topic
        total_capital: Total capital to allocate (in USD)
        n_volume_examples: Number of historical examples for volume estimation
        n_probability_samples: Number of samples for probability estimation
        question_model: Model for question generation (should include :online)
        volume_model: Model for volume estimation
        probability_model: Model for probability estimation
        reword_temperature: Temperature for prompt rewording in probability estimation
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY env var)
        output_dir: Base directory for saving results

    Returns:
        dict: Contains:
            - 'questions': List of enriched question dicts
            - 'config': Pipeline configuration
            - 'output_dir': Path to saved results
            - 'summary': Summary statistics

    Example:
        >>> results = run_market_pipeline(
        ...     n_topics=5,
        ...     k_questions_per_topic=2,
        ...     total_capital=10000
        ... )
        >>> print(f"Generated {len(results['questions'])} questions")
        >>> print(f"Results saved to: {results['output_dir']}")
    """
    # Get API key
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set"
            )

    # Store pipeline configuration
    pipeline_config = {
        "timestamp": datetime.now().isoformat(),
        "n_topics": n_topics,
        "k_questions_per_topic": k_questions_per_topic,
        "total_capital": total_capital,
        "n_volume_examples": n_volume_examples,
        "n_probability_samples": n_probability_samples,
        "question_model": question_model,
        "volume_model": volume_model,
        "probability_model": probability_model,
        "reword_temperature": reword_temperature,
        "output_dir": output_dir,
    }

    print(f"\n{'#'*80}")
    print("MARKET PIPELINE - END-TO-END EXECUTION")
    print(f"{'#'*80}")
    print(f"Configuration:")
    print(f"  Topics: {n_topics}, Questions/topic: {k_questions_per_topic}")
    print(f"  Total capital: ${total_capital:,.2f}")
    print(f"  Question model: {question_model}")
    print(f"  Volume model: {volume_model}")
    print(f"  Probability model: {probability_model}")

    try:
        # Step 1: Generate questions
        questions = generate_questions(
            n_topics=n_topics,
            k_questions_per_topic=k_questions_per_topic,
            model=question_model,
            api_key=api_key,
        )

        # Step 2: Estimate volumes
        questions = estimate_volumes(
            questions=questions,
            n_examples=n_volume_examples,
            model=volume_model,
            api_key=api_key,
        )

        # Step 3: Estimate probabilities
        questions = asyncio.run(
            estimate_probabilities(
                questions=questions,
                n_samples=n_probability_samples,
                model=probability_model,
                api_key=api_key,
                reword_temperature=reword_temperature,
            )
        )

        # Step 4: Allocate capital
        questions = allocate_capital(
            questions=questions,
            total_capital=total_capital,
        )

        # Step 5: Save results
        output_path = save_results(
            questions=questions,
            pipeline_config=pipeline_config,
            output_dir=output_dir,
        )

        # Create summary
        summary = {
            "n_questions_generated": len(questions),
            "n_with_volume_estimates": sum(
                1 for q in questions if q.get("estimated_daily_volume") is not None
            ),
            "n_with_probability_estimates": sum(
                1 for q in questions if q.get("probability_mean") is not None
            ),
            "total_capital_allocated": sum(
                q.get("allocated_capital", 0) for q in questions
            ),
        }

        print(f"\n{'#'*80}")
        print("PIPELINE COMPLETE")
        print(f"{'#'*80}")
        print(f"✓ Generated {summary['n_questions_generated']} questions")
        print(f"✓ Estimated volumes for {summary['n_with_volume_estimates']} questions")
        print(
            f"✓ Estimated probabilities for {summary['n_with_probability_estimates']} questions"
        )
        print(f"✓ Allocated ${summary['total_capital_allocated']:,.2f}")
        print(f"✓ Results saved to: {output_path}")
        print(f"{'#'*80}\n")

        return {
            "questions": questions,
            "config": pipeline_config,
            "output_dir": str(output_path),
            "summary": summary,
        }

    except Exception as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        # Allow command-line override of total capital
        total_capital = float(sys.argv[1])
    else:
        total_capital = 10000.0

    # Run pipeline with default settings
    results = run_market_pipeline(
        n_topics=3,
        k_questions_per_topic=2,
        total_capital=total_capital,
        n_volume_examples=15,
        n_probability_samples=10,
    )

    print("\nTo view results:")
    print(f"  cd {results['output_dir']}")
    print("  cat summary.json")
    print("  cat results.csv")
