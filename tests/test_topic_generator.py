"""
Unit tests for topic_generator module.

Run with: pytest tests/test_topic_generator.py -v
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path to import topic_generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topic_generator import generate_topics_and_questions, print_topics_and_questions


class TestGenerateTopicsAndQuestions:
    """Test suite for generate_topics_and_questions function."""

    @pytest.fixture
    def mock_api_key(self):
        """Fixture to set a mock API key."""
        original_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key-12345"
        yield "test-key-12345"
        # Cleanup
        if original_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = original_key

    @pytest.fixture
    def sample_llm_response(self):
        """Fixture providing a sample LLM response."""
        return """{
            "AI Regulation": [
                "Will the EU AI Act be fully implemented by December 2025?"
            ],
            "Climate Policy": [
                "Will global CO2 emissions decrease in 2025?"
            ]
        }"""

    def test_basic_functionality_n2_k1(self, mock_api_key, sample_llm_response):
        """Test basic functionality with N=2 topics and k=1 question (as requested)."""
        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = sample_llm_response

            result = generate_topics_and_questions(n_topics=2, k_questions=1)

            # Verify the function was called with correct parameters
            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs['api_key'] == mock_api_key

            # Verify result structure
            assert isinstance(result, dict)
            assert len(result) == 2
            assert "AI Regulation" in result
            assert "Climate Policy" in result

            # Verify each topic has exactly 1 question
            for topic, questions in result.items():
                assert isinstance(questions, list)
                assert len(questions) == 1
                assert isinstance(questions[0], str)
                assert questions[0].endswith("?")

    def test_parameter_validation_n_topics(self, mock_api_key):
        """Test that n_topics must be at least 1."""
        with pytest.raises(ValueError, match="n_topics must be at least 1"):
            generate_topics_and_questions(n_topics=0, k_questions=1)

        with pytest.raises(ValueError, match="n_topics must be at least 1"):
            generate_topics_and_questions(n_topics=-1, k_questions=1)

    def test_parameter_validation_k_questions(self, mock_api_key):
        """Test that k_questions must be at least 1."""
        with pytest.raises(ValueError, match="k_questions must be at least 1"):
            generate_topics_and_questions(n_topics=1, k_questions=0)

        with pytest.raises(ValueError, match="k_questions must be at least 1"):
            generate_topics_and_questions(n_topics=1, k_questions=-1)

    def test_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        original_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ.pop("OPENROUTER_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="OpenRouter API key not provided"):
                generate_topics_and_questions(n_topics=1, k_questions=1)
        finally:
            if original_key is not None:
                os.environ["OPENROUTER_API_KEY"] = original_key

    def test_multiple_questions_per_topic(self, mock_api_key):
        """Test generating multiple questions per topic."""
        response = """{
            "Tech Industry": [
                "Will Apple release a foldable iPhone by 2026?",
                "Will Microsoft acquire another major gaming company by 2025?",
                "Will quantum computers solve a practical problem by 2027?"
            ]
        }"""

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(n_topics=1, k_questions=3)

            assert len(result) == 1
            assert "Tech Industry" in result
            assert len(result["Tech Industry"]) == 3

    def test_k_questions_limit(self, mock_api_key):
        """Test that k_questions parameter limits the number of questions returned."""
        response = """{
            "Space Exploration": [
                "Will humans land on Mars by 2030?",
                "Will SpaceX complete Starship orbital test by 2025?",
                "Will the James Webb telescope discover signs of life by 2026?",
                "Will China establish a moon base by 2028?",
                "Will commercial space tourism become affordable by 2027?"
            ]
        }"""

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            # Request only 2 questions per topic
            result = generate_topics_and_questions(n_topics=1, k_questions=2)

            assert len(result["Space Exploration"]) == 2

    def test_json_parsing_with_extra_text(self, mock_api_key):
        """Test that JSON can be extracted even with surrounding text."""
        response = """Here are the topics and questions:

{
    "Economic Policy": [
        "Will inflation fall below 2% in the US by end of 2025?"
    ]
}

I hope this helps!"""

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(n_topics=1, k_questions=1)

            assert len(result) == 1
            assert "Economic Policy" in result

    def test_invalid_json_response(self, mock_api_key):
        """Test that invalid JSON raises appropriate error."""
        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = "This is not JSON at all"

            with pytest.raises(Exception, match="Error processing response"):
                generate_topics_and_questions(n_topics=1, k_questions=1)

    def test_malformed_json_response(self, mock_api_key):
        """Test that malformed JSON raises appropriate error."""
        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = '{"topic": ["question",]}'  # Trailing comma

            with pytest.raises(ValueError, match="Failed to parse JSON response"):
                generate_topics_and_questions(n_topics=1, k_questions=1)

    def test_empty_response(self, mock_api_key):
        """Test handling of empty JSON response."""
        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = "{}"

            result = generate_topics_and_questions(n_topics=1, k_questions=1)

            assert isinstance(result, dict)
            assert len(result) == 0

    def test_topics_without_questions_filtered_out(self, mock_api_key):
        """Test that topics with no valid questions are filtered out."""
        response = """{
            "Topic With Questions": [
                "Will this happen by 2025?"
            ],
            "Topic Without Questions": []
        }"""

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(n_topics=2, k_questions=1)

            assert len(result) == 1
            assert "Topic With Questions" in result
            assert "Topic Without Questions" not in result

    def test_custom_model_parameter(self, mock_api_key):
        """Test using a custom model parameter."""
        response = '{"Topic": ["Question?"]}'

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(
                n_topics=1,
                k_questions=1,
                model="anthropic/claude-sonnet-4"
            )

            # Verify the custom model was used (with :online suffix added automatically)
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs['model'] == "anthropic/claude-sonnet-4:online"

    def test_custom_temperature_parameter(self, mock_api_key):
        """Test using a custom temperature parameter."""
        response = '{"Topic": ["Question?"]}'

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(
                n_topics=1,
                k_questions=1,
                temperature=0.9
            )

            # Verify the custom temperature was used
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs['temperature'] == 0.9

    def test_web_search_enabled_by_default(self, mock_api_key):
        """Test that web search (:online) is enabled by default."""
        response = '{"Topic": ["Question?"]}'

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(
                n_topics=1,
                k_questions=1,
                model="openai/gpt-4o-mini"  # Without :online suffix
            )

            # Verify :online was automatically added
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs['model'].endswith(":online")
            assert call_kwargs['model'] == "openai/gpt-4o-mini:online"

    def test_web_search_can_be_disabled(self, mock_api_key):
        """Test that web search can be disabled."""
        response = '{"Topic": ["Question?"]}'

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(
                n_topics=1,
                k_questions=1,
                model="openai/gpt-4o-mini",
                enable_web_search=False
            )

            # Verify :online was NOT added
            call_kwargs = mock_query.call_args[1]
            assert not call_kwargs['model'].endswith(":online")
            assert call_kwargs['model'] == "openai/gpt-4o-mini"

    def test_web_search_not_duplicated(self, mock_api_key):
        """Test that :online suffix is not duplicated if already present."""
        response = '{"Topic": ["Question?"]}'

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(
                n_topics=1,
                k_questions=1,
                model="openai/gpt-4o-mini:online"  # Already has :online
            )

            # Verify :online was not duplicated
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs['model'] == "openai/gpt-4o-mini:online"
            assert call_kwargs['model'].count(":online") == 1

    def test_explicit_api_key(self):
        """Test providing API key explicitly."""
        response = '{"Topic": ["Question?"]}'

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(
                n_topics=1,
                k_questions=1,
                api_key="explicit-key-123"
            )

            # Verify the explicit key was used
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs['api_key'] == "explicit-key-123"

    def test_non_string_questions_filtered(self, mock_api_key):
        """Test that non-string questions are filtered out."""
        response = """{
            "Mixed Topic": [
                "Valid question?",
                123,
                null,
                "Another valid question?"
            ]
        }"""

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(n_topics=1, k_questions=5)

            # Only the two string questions should remain
            assert len(result["Mixed Topic"]) == 2
            assert all(isinstance(q, str) for q in result["Mixed Topic"])

    def test_print_topics_and_questions(self, capsys):
        """Test the print_topics_and_questions helper function."""
        topics = {
            "AI Safety": [
                "Will AGI be developed by 2030?",
                "Will AI alignment be solved by 2028?"
            ],
            "Climate": [
                "Will net-zero be achieved by 2050?"
            ]
        }

        print_topics_and_questions(topics)

        captured = capsys.readouterr()
        assert "AI Safety" in captured.out
        assert "Climate" in captured.out
        assert "Will AGI be developed by 2030?" in captured.out
        assert "Will net-zero be achieved by 2050?" in captured.out

    def test_prompt_includes_diversity_requirements(self, mock_api_key):
        """Test that the prompt includes topic diversity requirements."""
        response = '{"Topic": ["Question?"]}'

        with patch('topic_generator.query_llm_for_text_openrouter') as mock_query:
            mock_query.return_value = response

            result = generate_topics_and_questions(n_topics=2, k_questions=1)

            # Verify the prompt includes diversity requirements
            call_args = mock_query.call_args
            system_prompt = call_args[1]['system_prompt']
            user_prompt = call_args[1]['user_prompt']

            # Check system prompt mentions diversity
            assert "DIVERSE" in system_prompt or "different domains" in system_prompt

            # Check user prompt has examples of duplicate topics
            assert "EU AI Act" in user_prompt
            assert "AI Governance" in user_prompt
            assert "DUPLICATE" in user_prompt or "BAD" in user_prompt


class TestIntegration:
    """Integration tests that may call the actual API."""

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set - skipping integration test"
    )
    def test_real_api_call_n2_k1(self):
        """Integration test with real API call (N=2, k=1 as requested)."""
        result = generate_topics_and_questions(n_topics=2, k_questions=1)

        # Basic validation
        assert isinstance(result, dict)
        assert len(result) >= 1  # At least 1 topic (may be less than 2 if LLM struggles)

        for topic, questions in result.items():
            assert isinstance(topic, str)
            assert len(topic) > 0
            assert isinstance(questions, list)
            assert len(questions) >= 1
            assert len(questions) <= 1  # Should have at most 1 question

            for question in questions:
                assert isinstance(question, str)
                assert len(question) > 0
                assert question.endswith("?")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
