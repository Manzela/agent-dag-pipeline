"""Tests for GateAgent ABC and Gate-Agent pattern."""

from __future__ import annotations

import pytest

from agent_dag.adk.gate_agent import GateAgent, GateDecision, GateResult


class StubPassGate(GateAgent):
    """Test stub: always passes gate."""

    name: str = "stub_pass"
    description: str = "Test stub"

    def gate(self, state):
        return GateResult(GateDecision.PASS)

    async def agent(self, state, ctx):
        return {"output": "success"}


class StubRejectGate(GateAgent):
    """Test stub: always rejects at gate."""

    name: str = "stub_reject"
    description: str = "Test stub"

    def gate(self, state):
        return GateResult(GateDecision.REJECT, "test rejection")

    async def agent(self, state, ctx):
        raise AssertionError("Agent should not be called on REJECT")


class StubDeferGate(GateAgent):
    """Test stub: always defers at gate."""

    name: str = "stub_defer"
    description: str = "Test stub"

    def gate(self, state):
        return GateResult(GateDecision.DEFER, "quota exceeded")

    async def agent(self, state, ctx):
        raise AssertionError("Agent should not be called on DEFER")


class StubErrorAgent(GateAgent):
    """Test stub: gate passes but agent raises."""

    name: str = "stub_error"
    description: str = "Test stub"

    def gate(self, state):
        return GateResult(GateDecision.PASS)

    async def agent(self, state, ctx):
        raise RuntimeError("Intentional test error")


class TestGateDecision:
    """Test GateDecision enum."""

    def test_values(self):
        assert GateDecision.PASS.value == "PASS"
        assert GateDecision.REJECT.value == "REJECT"
        assert GateDecision.DEFER.value == "DEFER"


class TestGateResult:
    """Test GateResult dataclass."""

    def test_default(self):
        result = GateResult(GateDecision.PASS)
        assert result.decision == GateDecision.PASS
        assert result.reason == ""

    def test_with_reason(self):
        result = GateResult(GateDecision.REJECT, "invalid locale")
        assert result.reason == "invalid locale"

    def test_frozen(self):
        result = GateResult(GateDecision.PASS)
        with pytest.raises(AttributeError):
            result.decision = GateDecision.REJECT  # type: ignore


class TestGateAgentInstantiation:
    """Test GateAgent subclass instantiation."""

    def test_pass_agent(self):
        agent = StubPassGate()
        assert agent.name == "stub_pass"

    def test_reject_agent(self):
        agent = StubRejectGate()
        result = agent.gate({})
        assert result.decision == GateDecision.REJECT

    def test_defer_agent(self):
        agent = StubDeferGate()
        result = agent.gate({})
        assert result.decision == GateDecision.DEFER
        assert result.reason == "quota exceeded"

    def test_gate_determines_execution(self):
        """Gate REJECT should prevent agent from being called."""
        reject = StubRejectGate()
        result = reject.gate({})
        assert result.decision == GateDecision.REJECT

        # Verify agent would fail if called
        with pytest.raises(AssertionError):
            import asyncio
            asyncio.run(reject.agent({}, None))
