from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass
class BudgetLedger:
    """
    Shared token budget across a top-level turn trace and all downstream IPC fan-out calls,
    with atomic pre-call reservation so parallel `asyncio.gather` fan-outs cannot overshoot.
    """

    token_budget: int
    tokens_used: int = 0
    tokens_reserved: int = 0

    def remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used - self.tokens_reserved)

    def can_spend(self, min_reserve: int = 256) -> bool:
        return self.remaining() >= min_reserve

    def try_reserve(self, estimated_tokens: int, *, min_reserve: int = 64) -> int | None:
        """
        Atomically reserve `estimated_tokens` before starting an LLM call.
        Returns the reserved amount if admitted, or `None` if budget would be exceeded.
        """
        req = max(min_reserve, int(estimated_tokens))
        if self.remaining() < req:
            return None
        self.tokens_reserved += req
        return req

    def release(self, reserved: int) -> None:
        self.tokens_reserved = max(0, self.tokens_reserved - max(0, reserved))

    def settle(
        self,
        reserved: int,
        tokens_in: int,
        tokens_out: int,
        *,
        cache_read_tokens: int = 0,
        cache_read_weight: float = 0.1,
    ) -> None:
        self.release(reserved)
        self.record(
            tokens_in,
            tokens_out,
            cache_read_tokens=cache_read_tokens,
            cache_read_weight=cache_read_weight,
        )

    def record(
        self,
        tokens_in: int,
        tokens_out: int,
        *,
        cache_read_tokens: int = 0,
        cache_read_weight: float = 0.1,
    ) -> None:
        t_in = max(0, tokens_in)
        cached = min(max(0, cache_read_tokens), t_in)
        uncached = t_in - cached
        eff_in = uncached + int(round(cached * max(0.0, min(1.0, cache_read_weight))))
        self.tokens_used += eff_in + max(0, tokens_out)


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    status: str  # "ok" | "rejected_cycle" | "rejected_hop" | "rejected_budget"
    error_code: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class TraceContext:
    """Immutable execution trace enforcing cycle detection, hop depth, and token budgets."""

    trace_id: str
    call_stack: tuple[str, ...]
    hop_depth: int
    max_hop_depth: int
    ledger: BudgetLedger
    min_turn_reserve: int = 256

    @classmethod
    def start(
        cls,
        root_node_id: str,
        max_hop_depth: int = 3,
        token_budget: int = 32_000,
        min_turn_reserve: int = 256,
    ) -> TraceContext:
        return cls(
            trace_id=f"tr_{uuid.uuid4().hex[:10]}",
            call_stack=(root_node_id,),
            hop_depth=0,
            max_hop_depth=max_hop_depth,
            ledger=BudgetLedger(token_budget=token_budget),
            min_turn_reserve=min_turn_reserve,
        )

    def check_ipc_call(
        self,
        target_node_id: str,
        *,
        costs_hop: bool = True,
        costs_tokens: bool = True,
    ) -> GuardrailDecision:
        if costs_hop and target_node_id in self.call_stack:
            chain = " -> ".join([*self.call_stack, target_node_id])
            return GuardrailDecision(
                allowed=False,
                status="rejected_cycle",
                error_code="CYCLE_DETECTED",
                detail=f"Mutual recursion blocked: {chain}",
            )

        if costs_hop and self.hop_depth >= self.max_hop_depth:
            return GuardrailDecision(
                allowed=False,
                status="rejected_hop",
                error_code="MAX_HOP_DEPTH_EXCEEDED",
                detail=f"Hop depth {self.hop_depth} reached max_hop_depth={self.max_hop_depth}",
            )

        if costs_tokens and not self.ledger.can_spend(self.min_turn_reserve):
            return GuardrailDecision(
                allowed=False,
                status="rejected_budget",
                error_code="TOKEN_BUDGET_EXCEEDED",
                detail=(
                    f"Trace token budget exhausted ({self.ledger.tokens_used}/"
                    f"{self.ledger.token_budget} tokens used)"
                ),
            )

        return GuardrailDecision(allowed=True, status="ok")

    def descend(self, target_node_id: str) -> TraceContext:
        return TraceContext(
            trace_id=self.trace_id,
            call_stack=(*self.call_stack, target_node_id),
            hop_depth=self.hop_depth + 1,
            max_hop_depth=self.max_hop_depth,
            ledger=self.ledger,
            min_turn_reserve=self.min_turn_reserve,
        )
