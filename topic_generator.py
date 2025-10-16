"""
Topic Generator using OpenRouter API

This module searches for trending topics in news and social media, then generates
forecasting questions (yes/no questions with future objective answers) for each topic.
"""

import os
from typing import Dict, List, Optional, Tuple
from utils import query_llm_for_text_openrouter
import json
import re
from datetime import datetime


def validate_questions(
    topics: Dict[str, List[str]],
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.3,
) -> Dict[str, List[str]]:
    """
    Validate questions to ensure they are meaningful and not obviously answered.

    Filters out questions that:
    - Reference outdated contexts (e.g., Biden administration when he's no longer president)
    - Have obvious answers based on current events
    - Are meaningless in today's context

    Args:
        topics: Dictionary mapping topics to questions
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY env var)
        temperature: Sampling temperature (default: 0.3 for more consistent validation)

    Returns:
        dict: Filtered topics with only valid questions
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set"
            )

    current_date = datetime.now().strftime("%B %d, %Y")

    # Collect all questions for batch validation
    all_questions = []
    question_to_topic = {}
    for topic, questions in topics.items():
        for question in questions:
            all_questions.append(question)
            question_to_topic[question] = topic

    if not all_questions:
        return topics

    # Create validation prompt
    system_prompt = f"""You are a validation expert checking forecasting questions for logical consistency with current reality.

Today's date is {current_date}.

Your task is to identify questions that are:
1. NONSENSICAL - Reference outdated contexts (e.g., "Biden administration" policies when Biden is no longer president)
2. OBVIOUSLY ANSWERED - Have answers that are already clearly determined by current events
3. MEANINGLESS - Make no sense in today's context

Return ONLY questions that SHOULD BE REMOVED."""

    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(all_questions)])

    user_prompt = f"""Review these forecasting questions and identify which ones should be REMOVED because they are nonsensical, obviously answered, or meaningless in today's context ({current_date}).

QUESTIONS TO REVIEW:
{questions_text}

EXAMPLES OF QUESTIONS TO REMOVE:
- "Will the Biden administration achieve X by 2030?" (if Biden is no longer president)
- "Will candidate X win the 2024 election by November 2024?" (if 2024 election already happened)
- "Will company X release product Y in 2023?" (if we're past 2023)
- Questions about events that have already clearly occurred or been decided

Return ONLY a JSON object with this structure:
{{
    "remove": [
        1,
        3,
        5
    ]
}}

Where the numbers correspond to questions that should be removed.
If ALL questions are valid, return: {{"remove": []}}

Return ONLY the JSON object, no other text."""

    try:
        response_text = query_llm_for_text_openrouter(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=1000,
        )

        # Parse JSON response
        json_match = re.search(r"\{[\s\S]*\}", response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(
                r"```json\s*(\{[\s\S]*?\})\s*```", response_text, re.DOTALL
            )
            if json_match:
                json_text = json_match.group(1)
            else:
                print(
                    "Warning: Could not parse validation response, keeping all questions"
                )
                return topics
        else:
            json_text = json_match.group(0)

        validation_result = json.loads(json_text)
        indices_to_remove = set(validation_result.get("remove", []))

        # Filter out invalid questions
        questions_to_remove = set()
        for idx in indices_to_remove:
            if 1 <= idx <= len(all_questions):
                questions_to_remove.add(all_questions[idx - 1])

        # Build filtered result
        filtered_topics = {}
        removed_count = 0
        for topic, questions in topics.items():
            valid_questions = [q for q in questions if q not in questions_to_remove]
            if valid_questions:  # Only include topics that still have questions
                filtered_topics[topic] = valid_questions
            removed_count += len(questions) - len(valid_questions)

        if removed_count > 0:
            print(
                f"Validation removed {removed_count} nonsensical or obviously-answered questions"
            )

        return filtered_topics

    except Exception as e:
        print(f"Warning: Validation failed ({str(e)}), keeping all questions")
        return topics


def generate_topics_and_questions(
    n_topics: int = 5,
    k_questions: int = 3,
    model: str = "openai/gpt-4o-mini:online",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    enable_web_search: bool = True,
    num_days: int = 7,
    validate_questions_flag: bool = True,
) -> Dict[str, List[str]]:
    """
    Generate trending topics and forecasting questions using OpenRouter.

    This function searches for the top N most discussed and impactful stories
    in news and social media, then generates up to K yes/no forecasting questions
    for each topic. Questions must have objective future answers.

    Args:
        n_topics: Number of trending topics to generate (default: 5)
        k_questions: Maximum questions per topic (default: 3)
        model: OpenRouter model identifier (default: "openai/gpt-4o-mini:online")
               Use ":online" suffix for web search capability (e.g., "openai/gpt-4o-mini:online")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY env var)
        temperature: Sampling temperature (default: 0.7)
        enable_web_search: Automatically add ":online" suffix if not present (default: True)
        num_days: Number of days to look back for recent news (default: 7)
        validate_questions_flag: Validate and filter out nonsensical questions (default: True)

    Returns:
        dict: Mapping of topic keywords to list of forecasting questions
              Example: {
                  "AI Regulation": [
                      "Will the EU AI Act be fully implemented by December 2025?",
                      "Will the US pass federal AI regulation by January 2027?"
                  ],
                  "Climate Summit": [
                      "Will global emissions decrease by 5% before January 2026?"
                  ]
              }

    Raises:
        ValueError: If API key not found or invalid parameters
        Exception: If API call fails

    Example:
        >>> topics = generate_topics_and_questions(n_topics=3, k_questions=2)
        >>> for topic, questions in topics.items():
        ...     print(f"{topic}: {len(questions)} questions")

    Note:
        Web search is enabled by default using OpenRouter's :online feature.
        This allows the model to access recent news and current events.
        Web search costs $4 per 1,000 web results on OpenRouter.
    """
    # Validate parameters
    if n_topics < 1:
        raise ValueError("n_topics must be at least 1")
    if k_questions < 1:
        raise ValueError("k_questions must be at least 1")

    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set"
            )

    # Enable web search if requested and not already enabled
    if enable_web_search and not model.endswith(":online"):
        model = f"{model}:online"
        print(f"Web search enabled: using {model}")

    # Get current date for the prompt
    from datetime import datetime

    current_date = datetime.now().strftime("%B %d, %Y")

    # Generate the prompt for the LLM
    system_prompt = f"""You are an expert analyst who tracks trending topics in news and social media.
You identify the most impactful and discussed stories from RECENT NEWS (last {num_days} days), then generate forecasting questions about them.

Today's date is {current_date}.

CRITICAL REQUIREMENTS:
1. Topics must be DIVERSE - from different domains/subject areas
2. Do NOT create multiple topics about the same subject (e.g., don't have separate topics for "EU AI Act" and "AI Governance" - combine them into ONE topic)
3. Group ALL related stories into a SINGLE topic
4. Generate multiple questions for that one topic if needed
5. Prefer topics which are divisive, uncertain, or have generated significant debate.
6. Pay attention to platforms like Twitter / X, Reddit, YouTube, Facebook and news comment sections to identify hot topics.
7. Pay attention to upcoming events, deadlines, or decisions that could impact the topic.
8. Be sure to include alternative or niche news outlets like "The Epoch Times", "Heartland Institute", "The Daily Wire", "Zero Hedge", "Breitbart", "The Intercept", "Democracy Now", and "Wall Street Bets" etc. to get a full spectrum of perspectives.

Your forecasting questions must:
1. Be yes/no questions
2. Have answers that will be determined in the FUTURE (after today's date)
3. Have objective, verifiable answers (not subjective opinions)
4. Be specific and well-defined with concrete resolution dates
5. Be relevant to the topic
6. Prefer near-term questions (within 1-3 years) over far-future questions when possible

CRITICAL: Never use dates in the past or present. All resolution dates must be in the future.

You return results as valid JSON only."""

    user_prompt = f"""Search the web and identify the top {n_topics} most discussed and impactful stories from RECENT NEWS (within the last 7 days as of {current_date}).

CRITICAL: Each topic must be SUBSTANTIALLY DIFFERENT from the others. Do not create multiple topics about the same subject area or event.

For each topic:
1. Group ALL similar stories about the same general idea together into ONE topic
2. Label the topic with a few keywords (2-5 words)
3. Generate up to {k_questions} forecasting questions about that topic
4. Ensure topics are diverse - they should cover different subject areas, events, or domains
5. Avoid using vague words or ideas - be specific.  For example "Will X have a major impact on Y by end of 2025?" is too vague since "major impact" is subjective. Instead, ask specific questions with measurable outcomes.
6. Avoid references that are unclear without context. For example, "Will the candidate win by November 2024?" is too vague since "the candidate" is unclear. Instead, specify the candidate's name and position.

TOPIC DIVERSITY REQUIREMENTS:
- Topics must be from DIFFERENT domains/subject areas
- Do NOT create separate topics for the same event, policy, or subject area
- Combine all related stories into a SINGLE topic with multiple questions

EXAMPLES OF DUPLICATE TOPICS (DON'T DO THIS):
❌ BAD - These should be ONE topic:
  Topic 1: "EU AI Act Implementation"
  Topic 2: "AI Governance and Regulation"
  (Both are about EU AI regulation - should be combined into one topic)

❌ BAD - These should be ONE topic:
  Topic 1: "Climate Summit COP29"
  Topic 2: "International Climate Agreements"
  (Both are about international climate policy - should be combined)

✅ GOOD - These are distinct topics:
  Topic 1: "EU AI Act Implementation"
  Topic 2: "US-China Trade Relations"
  (Different domains: tech regulation vs. international trade)

✅ GOOD - These are distinct topics:
  Topic 1: "SpaceX Mars Mission"
  Topic 2: "Climate Change Legislation"
  (Different domains: space exploration vs. environmental policy)

Requirements for questions:
- Must be yes/no questions
- Must have FUTURE resolution dates (after {current_date})
- Must be specific and verifiable
- Include specific dates for resolution (e.g., "by December 31, 2025", "before January 1, 2027")
- If articles mention specific future dates/deadlines, use those dates
- If no specific date is mentioned, choose a reasonable near-term date (typically 1-3 years from now)
- Prefer near-term resolutions when possible
- If you cannot generate {k_questions} distinct valid questions, generate fewer
- If no valid questions can be generated, skip that topic and find another

EXAMPLES OF GOOD QUESTIONS:
- "Will the EU implement the AI Act by December 2025?" (if article mentions 2025 deadline)
- "Will China reduce emissions by 5% before January 1, 2027?" (if article mentions goal without date)
- "Will SpaceX land humans on Mars by 2030?" (if article discusses Mars mission plans)

EXAMPLES OF BAD QUESTIONS:
- "Will X happen by end of 2024?" (2024 is in the past relative to {current_date})
- "Will Y occur by March 2025?" (might be in the past depending on current date)
- "Is climate change real?" (subjective, not future-dated)
- "Will any major candidate drop out of the race for the 2028 election before September 1, 2028?" (too vague, "major candidate" is subjective)
- "Will the new solar energy technology be commercially available by January 1, 2027?" (too vague, "commercially available" is too subjective and it is unclear which 'new solar technology' is being referenced)

Return ONLY a JSON object with this exact structure:
{{
    "Topic Keywords 1": [
        "Question 1 with specific future date?",
        "Question 2 with specific future date?"
    ],
    "Topic Keywords 2": [
        "Question 1 with specific future date?"
    ]
}}

Do not include any explanation or text outside the JSON object.
Ensure you have exactly {n_topics} topics with at least one question each.
Remember: Today is {current_date}. All dates must be AFTER this date."""

    print(
        f"Querying {model} to generate {n_topics} topics with up to {k_questions} questions each..."
    )

    # Query the LLM
    # Use higher max_tokens for models with reasoning (like gpt-5)
    # Reasoning tokens count toward the limit but aren't in the final output
    max_tokens_to_use = (
        8000 if "gpt-5" in model or "o1" in model or "o3" in model else 4000
    )

    response_text = query_llm_for_text_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens_to_use,
    )

    # Debug: Check response
    if response_text is None:
        raise ValueError("Received None response from API")

    if not response_text.strip():
        raise ValueError("Received empty response from API")

    # Parse the JSON response
    try:
        # Extract JSON from response (in case there's extra text)
        # Try multiple JSON extraction patterns
        json_match = re.search(r"\{[\s\S]*\}", response_text, re.DOTALL)
        if not json_match:
            # Try with json code blocks
            json_match = re.search(
                r"```json\s*(\{[\s\S]*?\})\s*```", response_text, re.DOTALL
            )
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try plain code blocks
                json_match = re.search(
                    r"```\s*(\{[\s\S]*?\})\s*```", response_text, re.DOTALL
                )
                if json_match:
                    json_text = json_match.group(1)
                else:
                    # No JSON found
                    print(
                        f"DEBUG: Response text (first 1000 chars):\n{response_text[:1000]}"
                    )
                    raise ValueError(
                        f"No JSON object found in response. Response length: {len(response_text)}"
                    )
        else:
            json_text = json_match.group(0)

        result = json.loads(json_text)

        # Validate the result
        if not isinstance(result, dict):
            raise ValueError("Response is not a dictionary")

        # Ensure all values are lists of strings
        validated_result = {}
        for topic, questions in result.items():
            if not isinstance(topic, str):
                continue
            if not isinstance(questions, list):
                continue
            # Filter to only string questions and limit to k_questions
            valid_questions = [q for q in questions if isinstance(q, str)][:k_questions]
            if valid_questions:  # Only include topics with at least one question
                validated_result[topic] = valid_questions

        # Check if we got enough topics
        if len(validated_result) < n_topics:
            print(
                f"Warning: Only generated {len(validated_result)} topics out of {n_topics} requested"
            )

        print(
            f"Successfully generated {len(validated_result)} topics with {sum(len(q) for q in validated_result.values())} total questions"
        )

        # Validate questions to filter out nonsensical or obviously-answered ones
        if validate_questions_flag:
            print("Validating questions for logical consistency...")
            # Extract model base for validation (remove :online suffix)
            validation_model = model.replace(":online", "")
            validated_result = validate_questions(
                validated_result,
                model=validation_model,
                api_key=api_key,
                temperature=0.3,
            )

        return validated_result

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON response: {str(e)}\nResponse: {response_text[:500]}"
        )
    except Exception as e:
        raise Exception(f"Error processing response: {str(e)}")


def print_topics_and_questions(topics: Dict[str, List[str]]) -> None:
    """
    Pretty print topics and questions.

    Args:
        topics: Dictionary mapping topics to questions
    """
    print("\n" + "=" * 80)
    print("GENERATED TOPICS AND FORECASTING QUESTIONS")
    print("=" * 80 + "\n")

    for i, (topic, questions) in enumerate(topics.items(), 1):
        print(f"{i}. {topic}")
        print("-" * 80)
        for j, question in enumerate(questions, 1):
            print(f"   {j}) {question}")
        print()


if __name__ == "__main__":
    import sys

    # Default parameters
    n_topics = 5
    k_questions = 3

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            n_topics = int(sys.argv[1])
        except ValueError:
            print(f"Invalid n_topics: {sys.argv[1]}")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            k_questions = int(sys.argv[2])
        except ValueError:
            print(f"Invalid k_questions: {sys.argv[2]}")
            sys.exit(1)

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    print(f"Generating {n_topics} topics with up to {k_questions} questions each...\n")

    try:
        # Generate topics and questions
        result = generate_topics_and_questions(
            n_topics=n_topics, k_questions=k_questions
        )

        # Print results
        print_topics_and_questions(result)

        # Print summary
        total_questions = sum(len(questions) for questions in result.values())
        print(f"Summary: {len(result)} topics, {total_questions} questions total")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
