import type {
  ArchiveBranchResponse,
  ArtifactModel,
  AskTreeResponse,
  BranchFactModel,
  ChildContextMode,
  CompactBranchResponse,
  IpcEventModel,
  LLMBackendKind,
  MailboxItemModel,
  NodeDetailResponse,
  NodeModel,
  RuntimeSettingsModel,
  SearchHit,
  SearchScope,
  StreamCallbacks,
  ToolPermission,
  TreeInvariantModel,
  TreeModel,
  TreeSnapshotResponse,
} from '../types'

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text()
    let detail = text
    try {
      const parsed = JSON.parse(text)
      detail = parsed.detail || text
    } catch {
      // keep raw text
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listTrees: () => jsonFetch<TreeModel[]>('/api/trees'),

  createTree: (payload: {
    title: string
    goal?: string
    default_model?: string
    auto_commit_handoffs?: boolean
  }) =>
    jsonFetch<TreeSnapshotResponse>('/api/trees', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getTreeSnapshot: (treeId: string) =>
    jsonFetch<TreeSnapshotResponse>(`/api/trees/${treeId}`),

  updateTree: (
    treeId: string,
    payload: Partial<{
      title: string
      default_model: string
      shadow_model: string
      summary_model: string
      auto_commit_handoffs: boolean
    }>
  ) =>
    jsonFetch<TreeModel>(`/api/trees/${treeId}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),

  deleteTree: (treeId: string) =>
    jsonFetch<{ deleted: boolean }>(`/api/trees/${treeId}`, { method: 'DELETE' }),

  getNodeDetail: (nodeId: string) =>
    jsonFetch<NodeDetailResponse>(`/api/nodes/${nodeId}`),

  updateNode: (
    nodeId: string,
    payload: Partial<{
      title: string
      goal: string
      model_override: string | null
      summary: string
      status: string
    }>
  ) =>
    jsonFetch<NodeModel>(`/api/nodes/${nodeId}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),

  spawnChild: (
    parentId: string,
    payload: {
      title?: string
      goal: string
      context_mode: ChildContextMode
      summary_brief?: string
      slice_turn_count?: number
      model_override?: string | null
      tool_permission?: ToolPermission | null
      switch_head?: boolean
    }
  ) =>
    jsonFetch<{
      child: NodeModel | null
      ipc_event: IpcEventModel | null
      head_changed: { from_node_id: string; to_node_id: string } | null
    }>(`/api/nodes/${parentId}/spawn`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  finishNode: (
    nodeId: string,
    payload: {
      summary?: string
      artifacts?: Array<{ slug: string; title: string; kind: string; content: string; summary?: string }>
      auto_commit?: boolean
      return_to_parent?: boolean
    }
  ) =>
    jsonFetch<{
      node: NodeModel | null
      result: Record<string, unknown>
      handoff_pending: MailboxItemModel | null
      ipc_event: IpcEventModel | null
      head_changed: { from_node_id: string; to_node_id: string } | null
    }>(`/api/nodes/${nodeId}/finish`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  directRequest: (
    callerNodeId: string,
    payload: { target_id: string; prompt?: string | null; mutate?: boolean }
  ) =>
    jsonFetch<{
      result: Record<string, unknown>
      ipc_event: IpcEventModel | null
      stale_node_ids: string[]
    }>(`/api/nodes/${callerNodeId}/request`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  askTree: (
    callerNodeId: string,
    payload: { query: string; scope?: SearchScope; max_targets?: number; deep?: boolean }
  ) =>
    jsonFetch<AskTreeResponse>(`/api/nodes/${callerNodeId}/ask`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  compactNode: (
    nodeId: string,
    payload: { keep_recent_turns?: number | null; title?: string | null; reason?: string; switch_head?: boolean }
  ) =>
    jsonFetch<CompactBranchResponse>(`/api/nodes/${nodeId}/compact`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  archiveNode: (
    nodeId: string,
    payload: { reason?: string; switch_to_parent?: boolean }
  ) =>
    jsonFetch<ArchiveBranchResponse>(`/api/nodes/${nodeId}/archive`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listInvariants: (treeId: string) =>
    jsonFetch<TreeInvariantModel[]>(`/api/trees/${treeId}/invariants`),

  pinInvariant: (
    treeId: string,
    payload: { key: string; value: string; source_node_id?: string | null }
  ) =>
    jsonFetch<TreeInvariantModel>(`/api/trees/${treeId}/invariants`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  unpinInvariant: (treeId: string, key: string) =>
    jsonFetch<{ key: string; removed: boolean }>(
      `/api/trees/${treeId}/invariants/${encodeURIComponent(key)}`,
      { method: 'DELETE' }
    ),

  listFacts: (treeId: string, opts?: { nodeId?: string; includeArchived?: boolean }) => {
    const params = new URLSearchParams()
    if (opts?.nodeId) params.set('node_id', opts.nodeId)
    if (opts?.includeArchived) params.set('include_archived', 'true')
    const qs = params.toString()
    return jsonFetch<BranchFactModel[]>(`/api/trees/${treeId}/facts${qs ? `?${qs}` : ''}`)
  },

  addFacts: (
    treeId: string,
    payload: {
      node_id: string
      facts: Array<{ fact: string; resolved_date?: string; entities?: string[] }>
      source_msg_seq?: number | null
    }
  ) =>
    jsonFetch<BranchFactModel[]>(`/api/trees/${treeId}/facts`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  dismissStale: (nodeId: string, sourceNodeId: string) =>
    jsonFetch<{ ok: boolean; node: NodeModel | null }>(
      `/api/nodes/${nodeId}/dismiss-stale/${sourceNodeId}`,
      { method: 'POST' }
    ),

  commitHandoff: (
    mailboxId: string,
    payload: {
      summary?: string
      return_head_to_target?: boolean
    }
  ) =>
    jsonFetch<{
      mailbox_item: MailboxItemModel
      target_node: NodeModel
    }>(`/api/mailbox/${mailboxId}/commit`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  dismissHandoff: (mailboxId: string) =>
    jsonFetch<{ mailbox_item: MailboxItemModel }>(`/api/mailbox/${mailboxId}/dismiss`, {
      method: 'POST',
    }),

  reflogJump: (treeId: string, target: string, note = '') =>
    jsonFetch<{
      head_node: NodeModel
      reflog_cursor: number
    }>(`/api/trees/${treeId}/reflog/jump`, {
      method: 'POST',
      body: JSON.stringify({ target, note }),
    }),

  searchTree: (treeId: string, q: string, scope: SearchScope = 'all', nodeId?: string) => {
    const params = new URLSearchParams({ q, scope })
    if (nodeId) params.set('node_id', nodeId)
    return jsonFetch<SearchHit[]>(`/api/trees/${treeId}/search?${params.toString()}`)
  },

  createArtifact: (
    treeId: string,
    payload: {
      slug: string
      title: string
      kind: string
      content: string
      summary?: string
      created_by_node_id?: string | null
    }
  ) =>
    jsonFetch<ArtifactModel>(`/api/trees/${treeId}/artifacts`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getSettings: () => jsonFetch<RuntimeSettingsModel>('/api/settings'),

  updateSettings: (payload: {
    llm_provider?: LLMBackendKind
    default_model?: string
    shadow_model?: string
    summary_model?: string
    api_base?: string
    max_hop_depth?: number
    token_budget?: number
    auto_commit_handoffs?: boolean
    api_keys?: Record<string, string>
  }) =>
    jsonFetch<RuntimeSettingsModel>('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  async streamTurn(
    nodeId: string,
    content: string,
    callbacks: StreamCallbacks,
    signal?: AbortSignal
  ): Promise<void> {
    const res = await fetch(`/api/nodes/${nodeId}/turns/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
      signal,
    })

    if (!res.ok || !res.body) {
      const text = await res.text()
      callbacks.onError?.({ code: `HTTP_${res.status}`, message: text || 'Stream failed' })
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    const dispatchBlock = (block: string) => {
      if (!block.trim()) return
      let evName = 'message'
      const dataLines: string[] = []
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith('event:')) {
          evName = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trim())
        }
      }
      if (dataLines.length === 0) return
      let parsed: any
      try {
        parsed = JSON.parse(dataLines.join('\n'))
      } catch {
        return
      }
      switch (evName) {
        case 'turn_start':
          callbacks.onTurnStart?.(parsed)
          break
        case 'token':
          callbacks.onToken?.(parsed.delta || '')
          break
        case 'tool_call':
          callbacks.onToolCall?.(parsed)
          break
        case 'ipc_activity':
          callbacks.onIpcActivity?.(parsed)
          break
        case 'handoff_pending':
          callbacks.onHandoffPending?.(parsed)
          break
        case 'head_changed':
          callbacks.onHeadChanged?.(parsed)
          break
        case 'staleness_updated':
          callbacks.onStalenessUpdated?.(parsed)
          break
        case 'turn_complete':
          callbacks.onTurnComplete?.(parsed)
          break
        case 'error':
          callbacks.onError?.(parsed)
          break
      }
    }

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split(/\r?\n\r?\n/)
      buffer = parts.pop() ?? ''
      for (const part of parts) {
        dispatchBlock(part)
      }
    }
    if (buffer.trim()) {
      dispatchBlock(buffer)
    }
  },
}
