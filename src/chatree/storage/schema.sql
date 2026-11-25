PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS trees (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    root_node_id TEXT,
    head_node_id TEXT,
    default_model TEXT NOT NULL,
    shadow_model TEXT NOT NULL DEFAULT 'inherit',
    summary_model TEXT NOT NULL DEFAULT 'inherit',
    auto_commit_handoffs INTEGER NOT NULL DEFAULT 0,
    reflog_cursor INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES nodes(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    depth INTEGER NOT NULL DEFAULT 0,
    title TEXT NOT NULL,
    goal TEXT NOT NULL DEFAULT '',
    summary_brief TEXT NOT NULL DEFAULT '',
    context_mode TEXT NOT NULL CHECK(context_mode IN ('root', 'fork', 'slice', 'clean')),
    spawn_msg_seq INTEGER NOT NULL DEFAULT 0,
    slice_turn_count INTEGER NOT NULL DEFAULT 6,
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'COMPLETED', 'ARCHIVED')),
    node_kind TEXT NOT NULL DEFAULT 'chat' CHECK(node_kind IN ('chat', 'subagent')),
    tool_permission TEXT NOT NULL DEFAULT 'read_only' CHECK(tool_permission IN ('read_only', 'read_write')),
    model_override TEXT,
    summary TEXT,
    summary_version INTEGER NOT NULL DEFAULT 0,
    summary_minor_version INTEGER NOT NULL DEFAULT 0,
    last_change_severity TEXT NOT NULL DEFAULT 'material',
    last_change_delta TEXT NOT NULL DEFAULT '',
    rollup_summary TEXT,
    is_dirty INTEGER NOT NULL DEFAULT 0,
    last_msg_seq INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_tree_path ON nodes(tree_id, path);
CREATE INDEX IF NOT EXISTS idx_nodes_tree_parent ON nodes(tree_id, parent_id);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    model_used TEXT,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(node_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_messages_node_seq ON messages(node_id, seq);

CREATE TABLE IF NOT EXISTS ipc_events (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    trace_id TEXT NOT NULL,
    hop_depth INTEGER NOT NULL DEFAULT 0,
    from_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    to_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL CHECK(
        event_type IN (
            'spawn',
            'send',
            'send_finish',
            'request_summary',
            'request_shadow',
            'request_mutate',
            'search',
            'reflog_jump',
            'custom_tool'
        )
    ),
    payload_json TEXT NOT NULL DEFAULT '{}',
    response_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'ok',
    target_summary_version INTEGER,
    target_msg_seq INTEGER,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ipc_tree_created ON ipc_events(tree_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ipc_trace ON ipc_events(trace_id);

CREATE TABLE IF NOT EXISTS node_dependencies (
    consumer_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    source_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    seen_summary_version INTEGER NOT NULL DEFAULT 0,
    seen_msg_seq INTEGER NOT NULL DEFAULT 0,
    artifact_slug TEXT,
    last_ipc_event_id TEXT REFERENCES ipc_events(id) ON DELETE SET NULL,
    dismissed_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (consumer_node_id, source_node_id)
);

CREATE INDEX IF NOT EXISTS idx_deps_source ON node_dependencies(source_node_id);

CREATE TABLE IF NOT EXISTS shadow_cache (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    target_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_msg_seq INTEGER NOT NULL,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    summary_version INTEGER NOT NULL DEFAULT 0,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(target_node_id, target_msg_seq, model, prompt_hash)
);

CREATE INDEX IF NOT EXISTS idx_shadow_cache_lookup
    ON shadow_cache(target_node_id, target_msg_seq, model, prompt_hash);

CREATE TABLE IF NOT EXISTS tree_sessions (
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT 'Main Session',
    kind TEXT NOT NULL DEFAULT 'human' CHECK(kind IN ('human', 'agent')),
    head_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
    reflog_cursor INTEGER NOT NULL DEFAULT 0,
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tree_id, id)
);

CREATE INDEX IF NOT EXISTS idx_tree_sessions_tree ON tree_sessions(tree_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS reflog (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL DEFAULT 'default',
    seq INTEGER NOT NULL,
    from_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
    to_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    action TEXT NOT NULL CHECK(
        action IN ('init', 'spawn', 'jump', 'return', 'close', 'reopen', 'fork_edit', 'extract', 'promote')
    ),
    note TEXT NOT NULL DEFAULT '',
    orientation_cue TEXT NOT NULL DEFAULT '',
    cue_consumed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(tree_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_reflog_tree_seq ON reflog(tree_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_reflog_tree_session_seq ON reflog(tree_id, session_id, seq DESC);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'markdown',
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    created_by_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    UNIQUE(tree_id, slug, version)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_tree_slug ON artifacts(tree_id, slug, version DESC);

CREATE TABLE IF NOT EXISTS mailbox (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    to_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    from_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    ipc_event_id TEXT REFERENCES ipc_events(id) ON DELETE SET NULL,
    kind TEXT NOT NULL DEFAULT 'handoff' CHECK(kind IN ('handoff', 'message')),
    status TEXT NOT NULL DEFAULT 'pending_review' CHECK(status IN ('pending_review', 'unread', 'delivered', 'dismissed')),
    summary TEXT NOT NULL DEFAULT '',
    artifact_refs_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    delivered_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_mailbox_to_status ON mailbox(to_node_id, status);
CREATE INDEX IF NOT EXISTS idx_mailbox_tree_status ON mailbox(tree_id, status);

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    granularity TEXT NOT NULL CHECK(granularity IN ('chunk', 'summary', 'rollup', 'fact')),
    chunk_index INTEGER NOT NULL DEFAULT 0,
    seq_start INTEGER,
    seq_end INTEGER,
    content TEXT NOT NULL,
    vector_blob BLOB NOT NULL,
    vector_dim INTEGER NOT NULL,
    summary_version INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_tree_node ON embeddings(tree_id, node_id, granularity);
CREATE INDEX IF NOT EXISTS idx_embeddings_tree_path ON embeddings(tree_id, path);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS background_jobs (
    job_id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    parent_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    child_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    goal TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
    turns_executed INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    error TEXT,
    mailbox_id TEXT REFERENCES mailbox(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_bg_jobs_tree_created ON background_jobs(tree_id, created_at DESC);

CREATE TABLE IF NOT EXISTS sandbox_workspaces (
    node_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    base_ref TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS branch_facts (
    id TEXT PRIMARY KEY,
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    fact_text TEXT NOT NULL,
    entities_json TEXT NOT NULL DEFAULT '[]',
    resolved_date TEXT NOT NULL DEFAULT '',
    source_msg_seq INTEGER,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'superseded')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_branch_facts_tree_node ON branch_facts(tree_id, node_id, status);

CREATE TABLE IF NOT EXISTS tree_invariants (
    tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
    source_summary_version INTEGER NOT NULL DEFAULT 0,
    source_msg_seq INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tree_id, key)
);

-- FTS5 lexical indexes for hybrid BM25 + vector tree search
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    message_id UNINDEXED,
    tree_id UNINDEXED,
    node_id UNINDEXED,
    seq UNINDEXED,
    content,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    node_id UNINDEXED,
    tree_id UNINDEXED,
    title,
    goal,
    summary,
    rollup_summary,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts USING fts5(
    embedding_id UNINDEXED,
    tree_id UNINDEXED,
    node_id UNINDEXED,
    granularity UNINDEXED,
    content,
    tokenize='porter unicode61'
);
