from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ToolPermission = Literal["read_only", "read_write"]
SpawnPolicy = Literal["auto", "suggest", "manual"]
LLMBackendKind = Literal["auto", "litellm", "openai_compat", "mock"]

DEFAULT_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek/": 128_000,
    "deepseek-": 128_000,
    "openai/gpt-5": 128_000,
    "openai/gpt-4.1": 128_000,
    "openai/gpt-4o": 128_000,
    "openai/o3": 200_000,
    "openai/o1": 200_000,
    "gpt-5": 128_000,
    "gpt-4.1": 128_000,
    "gpt-4o": 128_000,
    "anthropic/claude": 200_000,
    "claude-": 200_000,
    "gemini/": 1_000_000,
    "gemini-": 1_000_000,
    "mistral/": 128_000,
    "xai/": 128_000,
    "groq/": 128_000,
    "together_ai/": 32_768,
    "ollama/": 8_192,
    "vllm/": 8_192,
    "sglang/": 8_192,
    "lm_studio/": 8_192,
    "llamacpp/": 8_192,
    "mock/": 120_000,
}


def resolve_model_context_window(
    model: str | None,
    *,
    overrides: dict[str, int] | None = None,
    configured_max: int | None = None,
) -> int:
    """
    Resolve the effective input context window (in tokens) for `model`, combining:
      1) explicit per-model overrides (`GuardrailPolicy.model_context_windows`),
      2) conservative built-in provider/model window table (`DEFAULT_MODEL_CONTEXT_WINDOWS`),
      3) optional upper cap (`configured_max`, e.g. `GuardrailPolicy.max_branch_context_tokens`).
    """
    low = (model or "").strip().lower()
    window: int | None = None
    if low and overrides:
        # Exact match first, then longest prefix/substring match
        for key, val in overrides.items():
            if key.strip().lower() == low and val > 0:
                window = int(val)
                break
        if window is None:
            for key, val in sorted(overrides.items(), key=lambda kv: len(kv[0]), reverse=True):
                k = key.strip().lower()
                if k and (low.startswith(k) or k in low) and val > 0:
                    window = int(val)
                    break
    if window is None and low:
        for prefix, val in sorted(
            DEFAULT_MODEL_CONTEXT_WINDOWS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if low.startswith(prefix) or prefix in low:
                window = val
                break
    if window is None:
        window = configured_max if (configured_max is not None and configured_max > 0) else 120_000
    if configured_max is not None and configured_max > 0:
        return min(window, int(configured_max))
    return int(window)


class GuardrailPolicy(BaseModel):
    """Deterministic safety & cost boundaries enforced by the orchestrator before LLM calls."""

    max_hop_depth: int = Field(default=3, ge=0, le=16)
    max_staleness_hops: int = Field(default=3, ge=1, le=8)
    token_budget: int = Field(default=96_000, ge=500, le=2_000_000)
    cache_read_budget_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    min_turn_reserve: int = Field(default=256, ge=0)
    max_tool_rounds: int = Field(default=6, ge=1, le=25)
    allow_mutating_requests: bool = True
    allow_autonomous_spawns: bool = True
    default_child_permission: ToolPermission = "read_only"
    spawn_policy: SpawnPolicy = "auto"
    max_active_children: int | None = Field(default=None, ge=1, le=100_000)
    max_active_subagents: int = Field(default=12, ge=1, le=1_000)
    materiality_cosine_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    short_branch_shadow_threshold_tokens: int = Field(default=4_000, ge=0)
    warm_cache_ttl_seconds: int = Field(default=300, ge=0)
    max_branch_context_tokens: int = Field(default=120_000, ge=256)
    reply_reserve_tokens: int = Field(default=1_024, ge=0, le=65_536)
    model_context_windows: dict[str, int] = Field(default_factory=dict)
    max_stale_notices: int = Field(default=8, ge=1, le=100)
    max_mailbox_notices: int = Field(default=6, ge=1, le=100)
    auto_compact_enabled: bool = True
    auto_compact_threshold_tokens: int = Field(default=24_000, ge=256, le=1_000_000)
    auto_compact_trigger_ratio: float = Field(default=0.75, ge=0.1, le=0.98)
    auto_compact_keep_recent_turns: int = Field(default=4, ge=0, le=24)
    auto_compact_min_messages: int = Field(default=6, ge=2, le=100)
    compaction_enforce_savings: bool = True
    compaction_handoff_max_chars: int = Field(default=3_200, ge=400, le=32_000)
    shadow_max_tokens: int | None = Field(default=512, ge=64, le=16_384)
    max_concurrent_background_jobs: int = Field(default=4, ge=1, le=64)
    max_shadow_cache_entries: int = Field(default=500, ge=10, le=50_000)
    semantic_shadow_cache_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    max_pinned_invariants: int = Field(default=8, ge=1, le=32)
    max_pinned_value_chars: int = Field(default=120, ge=20, le=500)
    preserve_kv_cache_tool_schemas: bool = True
    enable_compound_ask_tool: bool = True
    auto_title_enabled: bool = True
    compact_steady_state_envelope: bool = True
    bg_summary_min_dirty_messages: int = Field(default=2, ge=1, le=50)


class RagConfig(BaseModel):
    """Configuration for 4-tier hybrid Tree RAG (chunks + lazy summaries + Tier-2.5 facts + subtree rollups)."""

    enabled: bool = True
    chunk_target_tokens: int = Field(default=320, ge=50, le=2000)
    chunk_overlap_tokens: int = Field(default=60, ge=0, le=500)
    bm25_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    recency_boost: float = Field(default=0.08, ge=0.0, le=1.0)
    embedding_model: str = "hash/local-128"
    embedding_dim: int = Field(default=128, ge=16, le=4096)
    lazy_summary_turn_interval: int = Field(default=8, ge=1)
    speculative_pointer_top_k: int = Field(default=3, ge=0, le=10)
    speculative_pointer_min_score: float = Field(default=0.012, ge=0.0, le=1.0)
    enable_atomic_facts: bool = True
    fact_hint_top_k: int = Field(default=4, ge=0, le=12)
    max_rollup_depth: int = Field(default=8, ge=1, le=64)


class AblationConfig(BaseModel):
    """Feature switches for controlled research ablations and benchmarking."""

    enable_shadow_turns: bool = True
    enable_shadow_cache: bool = True
    enable_staleness_notices: bool = True
    enable_transitive_staleness: bool = True
    enable_materiality_gating: bool = True
    enable_ephemeral_cues: bool = True
    enable_speculative_pointers: bool = True
    enable_hybrid_rag: bool = True
    enable_subtree_rollups: bool = True
    enable_atomic_facts: bool = True
    enable_parallel_ipc: bool = True
    enable_provider_cache_control: bool = True


class ChatreeConfig(BaseSettings):
    """Top-level configuration for the Chatree framework and optional server."""

    model_config = SettingsConfigDict(
        env_prefix="CHATREE_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_path: str = "./data/chatree.db"
    llm_provider: LLMBackendKind = "auto"
    default_model: str = "mock/chatree-demo"
    shadow_model: str = "inherit"
    summary_model: str = "inherit"
    api_base: str | None = None
    api_key: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    api_token: str | None = None
    persist_api_keys_in_db: bool = False
    auto_commit_handoffs: bool = False
    auto_background_summarize_on_switch: bool = False
    max_hop_depth: int = 3
    token_budget: int = 96_000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    guardrails: GuardrailPolicy = Field(default_factory=GuardrailPolicy)
    rag: RagConfig = Field(default_factory=RagConfig)
    ablations: AblationConfig = Field(default_factory=AblationConfig)

    def model_post_init(self, __context: object) -> None:
        # Synchronize flat env overrides with nested guardrail defaults when explicitly set
        if self.max_hop_depth != 3 and self.guardrails.max_hop_depth == 3:
            self.guardrails.max_hop_depth = self.max_hop_depth
        if self.token_budget != 96_000 and self.guardrails.token_budget == 96_000:
            self.guardrails.token_budget = self.token_budget

    def resolved_db_path(self) -> Path | str:
        if self.db_path == ":memory:":
            return ":memory:"
        p = Path(self.db_path)
        if not p.is_absolute():
            p = p.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
