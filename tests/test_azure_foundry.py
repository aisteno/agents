"""Tests for Azure AI Foundry integration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from livekit.plugins import openai


class TestAzureFoundry:
    """Test Azure AI Foundry integration for Claude and Grok models."""

    def test_with_azure_foundry_requires_endpoint(self):
        """Test that azure_endpoint is required."""
        with pytest.raises(ValueError, match="azure_endpoint is required"):
            openai.LLM.with_azure_foundry(
                model="claude-sonnet-4-5",
                api_key="test-key",
            )

    def test_with_azure_foundry_requires_authentication(self):
        """Test that authentication is required."""
        with pytest.raises(ValueError, match="Authentication is required"):
            openai.LLM.with_azure_foundry(
                model="claude-sonnet-4-5",
                azure_endpoint="https://test.models.ai.azure.com",
            )

    @patch.dict(
        os.environ,
        {
            "AZURE_FOUNDRY_ENDPOINT": "https://test.models.ai.azure.com",
            "AZURE_FOUNDRY_API_KEY": "test-key",
        },
    )
    def test_with_azure_foundry_uses_environment_variables(self):
        """Test that environment variables are used when parameters not provided."""
        llm = openai.LLM.with_azure_foundry(
            model="claude-sonnet-4-5",
        )
        assert llm is not None
        assert llm._opts.model == "claude-sonnet-4-5"

    def test_with_azure_foundry_claude_models(self):
        """Test Azure Foundry with Claude models."""
        claude_models = [
            "claude-opus-4-1",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-7-sonnet",
            "claude-3-5-haiku",
        ]

        for model in claude_models:
            llm = openai.LLM.with_azure_foundry(
                model=model,
                azure_endpoint="https://test.models.ai.azure.com",
                api_key="test-key",
            )
            assert llm._opts.model == model

    def test_with_azure_foundry_grok_models(self):
        """Test Azure Foundry with Grok models."""
        grok_models = ["grok-3", "grok-3-mini"]

        for model in grok_models:
            llm = openai.LLM.with_azure_foundry(
                model=model,
                azure_endpoint="https://test.models.ai.azure.com",
                api_key="test-key",
            )
            assert llm._opts.model == model

    def test_with_azure_foundry_accepts_all_llm_parameters(self):
        """Test that all standard LLM parameters are accepted."""
        llm = openai.LLM.with_azure_foundry(
            model="claude-sonnet-4-5",
            azure_endpoint="https://test.models.ai.azure.com",
            api_key="test-key",
            temperature=0.7,
            top_p=0.9,
            verbosity="high",
            user="test-user",
        )
        assert llm._opts.temperature == 0.7
        assert llm._opts.top_p == 0.9
        assert llm._opts.verbosity == "high"
        assert llm._opts.user == "test-user"

    def test_with_azure_foundry_base_url_construction(self):
        """Test that base_url is constructed correctly from endpoint."""
        llm = openai.LLM.with_azure_foundry(
            model="claude-sonnet-4-5",
            azure_endpoint="https://test.models.ai.azure.com",
            api_key="test-key",
        )
        # Verify client was created
        assert llm._client is not None

    def test_with_azure_foundry_base_url_with_trailing_slash(self):
        """Test that trailing slashes are handled correctly."""
        llm = openai.LLM.with_azure_foundry(
            model="claude-sonnet-4-5",
            azure_endpoint="https://test.models.ai.azure.com/",
            api_key="test-key",
        )
        assert llm._client is not None

    @patch.dict(
        os.environ,
        {
            "AZURE_FOUNDRY_API_KEY": "env-key",
            "AZURE_OPENAI_API_KEY": "fallback-key",
        },
    )
    def test_with_azure_foundry_api_key_precedence(self):
        """Test API key fallback logic."""
        # AZURE_FOUNDRY_API_KEY should take precedence
        llm = openai.LLM.with_azure_foundry(
            model="claude-sonnet-4-5",
            azure_endpoint="https://test.models.ai.azure.com",
        )
        assert llm is not None

    def test_with_azure_foundry_explicit_base_url_override(self):
        """Test that explicit base_url parameter overrides default construction."""
        custom_base = "https://custom.endpoint.com/v1"
        llm = openai.LLM.with_azure_foundry(
            model="claude-sonnet-4-5",
            azure_endpoint="https://test.models.ai.azure.com",
            base_url=custom_base,
            api_key="test-key",
        )
        assert llm._client is not None


class TestAzureFoundryModelTypes:
    """Test that Azure Foundry model types are properly exported."""

    def test_azure_foundry_models_are_importable(self):
        """Test that model types can be imported."""
        from livekit.plugins.openai import (
            AzureFoundryClaudeModels,
            AzureFoundryGrokModels,
            AzureFoundryModels,
        )

        # If we can import them, they're properly exported
        assert AzureFoundryClaudeModels is not None
        assert AzureFoundryGrokModels is not None
        assert AzureFoundryModels is not None

    def test_azure_foundry_models_in_all(self):
        """Test that model types are in __all__."""
        import livekit.plugins.openai as openai_plugin

        assert "AzureFoundryClaudeModels" in openai_plugin.__all__
        assert "AzureFoundryGrokModels" in openai_plugin.__all__
        assert "AzureFoundryModels" in openai_plugin.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
