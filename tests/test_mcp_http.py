from __future__ import annotations

from contextlib import asynccontextmanager

from livekit.agents.llm import mcp


@asynccontextmanager
async def _dummy_streams():
    yield None


def test_streamable_http_uses_post_only_transport_by_default(monkeypatch):
    post_only_calls: list[dict] = []

    def fake_post_only(**kwargs):
        post_only_calls.append(kwargs)
        return _dummy_streams()

    def fail_streamable(**kwargs):
        raise AssertionError("expected post-only streamable HTTP client")

    monkeypatch.setattr(mcp, "_post_only_streamablehttp_client", fake_post_only)
    monkeypatch.setattr(mcp, "streamablehttp_client", fail_streamable)

    server = mcp.MCPServerHTTP("https://example.com/mcp", headers={"x-test": "1"})
    streams = server.client_streams()

    assert streams is not None
    assert len(post_only_calls) == 1
    assert post_only_calls[0]["url"] == "https://example.com/mcp"
    assert post_only_calls[0]["headers"] == {"x-test": "1"}


def test_streamable_http_can_opt_into_server_push(monkeypatch):
    streamable_calls: list[dict] = []

    def fail_post_only(**kwargs):
        raise AssertionError("expected regular streamable HTTP client")

    def fake_streamable(**kwargs):
        streamable_calls.append(kwargs)
        return _dummy_streams()

    monkeypatch.setattr(mcp, "_post_only_streamablehttp_client", fail_post_only)
    monkeypatch.setattr(mcp, "streamablehttp_client", fake_streamable)

    server = mcp.MCPServerHTTP(
        "https://example.com/mcp",
        headers={"x-test": "1"},
        enable_server_push=True,
    )
    streams = server.client_streams()

    assert streams is not None
    assert len(streamable_calls) == 1
    assert streamable_calls[0]["url"] == "https://example.com/mcp"
    assert streamable_calls[0]["headers"] == {"x-test": "1"}


def test_sse_transport_still_uses_sse_client(monkeypatch):
    sse_calls: list[dict] = []

    def fake_sse(**kwargs):
        sse_calls.append(kwargs)
        return _dummy_streams()

    monkeypatch.setattr(mcp, "sse_client", fake_sse)

    server = mcp.MCPServerHTTP("https://example.com/sse")
    streams = server.client_streams()

    assert streams is not None
    assert len(sse_calls) == 1
    assert sse_calls[0]["url"] == "https://example.com/sse"
