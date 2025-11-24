"""Tests for Azure OpenAI verbosity parameter support."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from livekit.agents.types import NOT_GIVEN
from livekit.plugins import openai


class TestAzureOpenAIVerbosity:
    """Test verbosity parameter support in Azure OpenAI integration."""

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "test-key",
            "OPENAI_API_VERSION": "2024-10-01-preview",
        },
    )
    def test_with_azure_accepts_verbosity_parameter(self):
        """Test that with_azure() accepts verbosity parameter."""
        llm = openai.LLM.with_azure(
            model="gpt-4o",
            verbosity="high",
        )
        assert llm is not None
        # Verify verbosity is stored in options
        assert llm._opts.verbosity == "high"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "test-key",
            "OPENAI_API_VERSION": "2024-10-01-preview",
        },
    )
    def test_with_azure_verbosity_all_values(self):
        """Test all valid verbosity values."""
        for verbosity in ["low", "medium", "high"]:
            llm = openai.LLM.with_azure(
                model="gpt-4o",
                verbosity=verbosity,
            )
            assert llm._opts.verbosity == verbosity

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "test-key",
            "OPENAI_API_VERSION": "2024-10-01-preview",
        },
    )
    def test_with_azure_verbosity_default_not_given(self):
        """Test that verbosity defaults to NOT_GIVEN when not specified."""
        llm = openai.LLM.with_azure(
            model="gpt-4o",
        )
        assert llm._opts.verbosity is NOT_GIVEN

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "test-key",
            "OPENAI_API_VERSION": "2024-10-01-preview",
        },
    )
    def test_with_azure_verbosity_with_other_parameters(self):
        """Test verbosity works alongside other parameters."""
        llm = openai.LLM.with_azure(
            model="gpt-4o",
            verbosity="medium",
            temperature=0.7,
            top_p=0.9,
            reasoning_effort="medium",
        )
        assert llm._opts.verbosity == "medium"
        assert llm._opts.temperature == 0.7
        assert llm._opts.top_p == 0.9
        assert llm._opts.reasoning_effort == "medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
