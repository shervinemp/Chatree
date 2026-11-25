export type ContextMode = 'root' | 'fork' | 'slice' | 'clean'
export type ChildContextMode = 'fork' | 'slice' | 'clean'
export type NodeStatus = 'ACTIVE' | 'COMPLETED' | 'ARCHIVED'
export type ToolPermission = 'read_only' | 'read_write'
export type LLMBackendKind = 'auto' | 'litellm' | 'openai_compat' | 'mock'
export type ReflogAction = 'init' | 'spawn' | 'jump' | 'return' | 'close' | 'reopen' | 'fork_edit'
export type IpcEventType =
  | 'spawn'
  | 'send'
  | 'send_finish'
  | 'request_summary'
  | 'request_shadow'
  | 'request_mutate'
  | 'search'
  | 'reflog_jump'
  | 'custom_tool'
export type SearchScope = 'all' | 'ancestors' | 'subtree' | 'siblings'
export type RagGranularity = 'chunk' | 'summary' | 'rollup' | 'fact' | 'node' | 'message'

export interface ArtifactRef {
  id: string
  slug: string
  version: number
  kind: string
  title: string
  summary: string
}

export interface ArtifactModel {
  id: string
  tree_id: string
  slug: string
  version: number
  kind: string
  title: string
  content: string
  summary: string
  created_by_node_id: string | null
  created_at: string
}

export interface StaleDependencyInfo {
  source_node_id: string
  source_title: string
  seen_summary_version: number
  current_summary_version: number
  seen_msg_seq: number
  current_msg_seq: number
  source_is_dirty: boolean
  reason: string
}

export interface TreeModel {
  id: string
  title: string
  root_node_id: string | null
  head_node_id: string | null
  default_model: string
  shadow_model: string
  summary_model: string
  auto_commit_handoffs: boolean
  reflog_cursor: number
  created_at: string
  updated_at: string
}

export interface NodeModel {
  id: string
  tree_id: string
  parent_id: string | null
  path: string
  depth: number
  title: string
  goal: string
  summary_brief: string
  context_mode: ContextMode
  spawn_msg_seq: number
  slice_turn_count: number
  status: NodeStatus
  model_override: string | null
  tool_permission?: ToolPermission
  summary: string | null
  summary_version: number
  is_dirty: boolean
  last_msg_seq: number
  tokens_in: number
  tokens_out: number
  cache_read_tokens: number
  created_at: string
  updated_at: string
  stale_dependencies: StaleDependencyInfo[]
  child_count: number
  pending_handoffs_count: number
}

export interface MessageModel {
  id: string
  node_id: string
  tree_id: string
  seq: number
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  tool_calls: Array<Record<string, unknown>> | null
  tool_call_id: string | null
  tool_name: string | null
  model_used: string | null
  tokens_in: number
  tokens_out: number
  cache_read_tokens: number
  latency_ms: number
  metadata: Record<string, unknown>
  created_at: string
}

export interface IpcEventModel {
  id: string
  tree_id: string
  trace_id: string
  hop_depth: number
  from_node_id: string
  to_node_id: string | null
  event_type: IpcEventType
  payload: Record<string, unknown>
  response: Record<string, unknown>
  status: string
  target_summary_version: number | null
  target_msg_seq: number | null
  tokens_in: number
  tokens_out: number
  latency_ms: number
  created_at: string
}

export interface ReflogEntryModel {
  id: string
  tree_id: string
  seq: number
  head_ref: string
  from_node_id: string | null
  to_node_id: string
  action: ReflogAction
  note: string
  orientation_cue: string
  cue_consumed: boolean
  created_at: string
}

export interface MailboxItemModel {
  id: string
  tree_id: string
  to_node_id: string
  from_node_id: string
  ipc_event_id: string | null
  kind: 'handoff' | 'message'
  status: 'pending_review' | 'unread' | 'delivered' | 'dismissed'
  summary: string
  artifact_refs: ArtifactRef[]
  payload: Record<string, unknown>
  created_at: string
  delivered_at: string | null
}

export interface BranchFactModel {
  id: string
  tree_id: string
  node_id: string
  fact_text: string
  entities: string[]
  resolved_date: string
  source_msg_seq: number | null
  status: 'active' | 'superseded'
  created_at: string
}

export interface TreeInvariantModel {
  tree_id: string
  key: string
  value: string
  source_node_id: string | null
  source_summary_version: number
  source_msg_seq: number
  updated_at: string
}

export interface SearchHit {
  node_id: string
  node_title: string
  node_status: NodeStatus
  path: string
  summary_version: number
  summary: string | null
  snippet: string
  score: number
  granularity?: RagGranularity
  matched_seq: number | null
  source: 'hybrid' | 'vector' | 'bm25' | 'node' | 'message'
}

export interface CompactBranchResponse {
  ok: boolean
  compacted: boolean
  from_node_id: string
  from_node_title: string
  to_node_id: string
  to_node_title: string
  context_mode: ChildContextMode
  keep_recent_turns: number
  tokens_before: number
  tokens_after: number
  compression_ratio: number
  source_summary_version: number
  handoff_preview: string
  switched_head: boolean
}

export interface ArchiveBranchResponse {
  ok: boolean
  archived_node_id: string
  archived_node_title: string
  status: NodeStatus
  quarantined_from_search: boolean
  reason: string
  switched_head_to: string | null
}

export interface AskTreeResponse {
  result: {
    query?: string
    scope?: SearchScope
    deep?: boolean
    answer?: string
    targets_queried?: string[]
    answers?: Array<Record<string, unknown>>
    found_answers?: Array<Record<string, unknown>>
    facts_used?: number
    synth_tokens_in?: number
    [key: string]: unknown
  }
  ipc_event: IpcEventModel | null
}

export interface TreeSnapshotResponse {
  tree: TreeModel
  nodes: NodeModel[]
  reflog: ReflogEntryModel[]
  pending_handoffs: MailboxItemModel[]
  artifacts: ArtifactModel[]
  recent_ipc: IpcEventModel[]
}

export interface NodeDetailResponse {
  node: NodeModel
  ancestors: NodeModel[]
  inherited_messages: MessageModel[]
  messages: MessageModel[]
  pending_handoffs: MailboxItemModel[]
  estimated_context_tokens: number
}

export interface RuntimeSettingsModel {
  llm_provider: LLMBackendKind
  default_model: string
  shadow_model: string
  summary_model: string
  api_base: string | null
  max_hop_depth: number
  token_budget: number
  auto_commit_handoffs: boolean
  configured_providers: Record<string, boolean>
}

export interface StreamCallbacks {
  onTurnStart?: (data: {
    trace_id: string
    node_id: string
    user_message: MessageModel
    ephemeral_notices: string[]
    reopened: boolean
  }) => void
  onToken?: (delta: string) => void
  onToolCall?: (data: { tool_call_id: string; name: string; arguments: Record<string, unknown> }) => void
  onIpcActivity?: (ev: IpcEventModel) => void
  onHandoffPending?: (mb: MailboxItemModel) => void
  onHeadChanged?: (data: { from_node_id: string; to_node_id: string; reflog_entry: ReflogEntryModel }) => void
  onStalenessUpdated?: (data: { source_node_id: string; stale_node_ids: string[] }) => void
  onTurnComplete?: (data: {
    trace_id: string
    assistant_message: MessageModel | null
    node: NodeModel | null
    head_node_id: string
    trace_summary: { tokens_used: number; token_budget: number; ipc_count: number }
  }) => void
  onError?: (err: { code: string; message: string }) => void
}
