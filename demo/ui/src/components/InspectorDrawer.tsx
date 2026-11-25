import { useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Archive,
  Calendar,
  ChevronDown,
  ChevronRight,
  Copy,
  Database,
  FileCode2,
  History,
  Pin,
  Plus,
  ShieldAlert,
  Trash2,
} from 'lucide-react'
import type {
  ArtifactModel,
  BranchFactModel,
  IpcEventModel,
  NodeModel,
  ReflogEntryModel,
  TreeInvariantModel,
} from '../types'

interface InspectorDrawerProps {
  nodes: NodeModel[]
  reflog: ReflogEntryModel[]
  reflogCursor: number
  ipcEvents: IpcEventModel[]
  artifacts: ArtifactModel[]
  invariants: TreeInvariantModel[]
  facts: BranchFactModel[]
  activeNodeId: string | null
  includeArchivedFacts: boolean
  onToggleIncludeArchivedFacts: (include: boolean) => void
  onJumpReflog: (target: string, note?: string) => void
  onJumpToNode: (nodeId: string) => void
  onCreateArtifact: (payload: {
    slug: string
    title: string
    kind: string
    content: string
    summary: string
  }) => Promise<void>
  onPinInvariant: (payload: {
    key: string
    value: string
    source_node_id?: string | null
  }) => Promise<void>
  onUnpinInvariant: (key: string) => Promise<void>
  onAddFact: (payload: {
    node_id: string
    fact: string
    resolved_date?: string
    entities: string[]
  }) => Promise<void>
}

export function InspectorDrawer({
  nodes,
  reflog,
  reflogCursor,
  ipcEvents,
  artifacts,
  invariants,
  facts,
  activeNodeId,
  includeArchivedFacts,
  onToggleIncludeArchivedFacts,
  onJumpReflog,
  onJumpToNode,
  onCreateArtifact,
  onPinInvariant,
  onUnpinInvariant,
  onAddFact,
}: InspectorDrawerProps) {
  const [tab, setTab] = useState<'reflog' | 'ipc' | 'artifacts' | 'memory'>('reflog')
  const [expandedIpc, setExpandedIpc] = useState<Record<string, boolean>>({})
  const [expandedArt, setExpandedArt] = useState<Record<string, boolean>>({})

  // New artifact state
  const [showNewArt, setShowNewArt] = useState(false)
  const [artSlug, setArtSlug] = useState('')
  const [artTitle, setArtTitle] = useState('')
  const [artKind, setArtKind] = useState('markdown')
  const [artContent, setArtContent] = useState('')
  const [savingArt, setSavingArt] = useState(false)

  // New PINNED invariant state
  const [showNewPin, setShowNewPin] = useState(false)
  const [pinKey, setPinKey] = useState('')
  const [pinVal, setPinVal] = useState('')
  const [pinLinkActive, setPinLinkActive] = useState(true)
  const [savingPin, setSavingPin] = useState(false)

  // Tier-2.5 facts state
  const [factsScope, setFactsScope] = useState<'all' | 'active'>('all')
  const [showNewFact, setShowNewFact] = useState(false)
  const [factText, setFactText] = useState('')
  const [factDate, setFactDate] = useState('')
  const [factEntities, setFactEntities] = useState('')
  const [savingFact, setSavingFact] = useState(false)

  const nodeNames: Record<string, string> = {}
  const nodesById: Record<string, NodeModel> = {}
  for (const n of nodes) {
    nodeNames[n.id] = n.title
    nodesById[n.id] = n
  }

  const getInvariantStaleReason = (inv: TreeInvariantModel): string | null => {
    if (!inv.source_node_id) return null
    const src = nodesById[inv.source_node_id]
    if (!src) return null
    if (src.status === 'ARCHIVED') return `Source branch ${src.id} was archived`
    if (src.summary_version > inv.source_summary_version) {
      return `Source branch ${src.id} revised to v${src.summary_version}`
    }
    if (src.is_dirty && src.last_msg_seq > inv.source_msg_seq) {
      return `Source branch ${src.id} has new uncommitted edits (#${src.last_msg_seq} > #${inv.source_msg_seq})`
    }
    return null
  }

  const visibleFacts =
    factsScope === 'active' && activeNodeId
      ? facts.filter((f) => f.node_id === activeNodeId)
      : facts

  const memoryCount = invariants.length + facts.length
  const stalePinCount = invariants.filter((inv) => getInvariantStaleReason(inv) !== null).length

  const handleSaveArtifact = async () => {
    if (!artSlug.trim() || !artTitle.trim() || !artContent.trim()) return
    setSavingArt(true)
    try {
      await onCreateArtifact({
        slug: artSlug.trim(),
        title: artTitle.trim(),
        kind: artKind,
        content: artContent,
        summary: '',
      })
      setArtSlug('')
      setArtTitle('')
      setArtContent('')
      setShowNewArt(false)
    } finally {
      setSavingArt(false)
    }
  }

  const handleSavePin = async () => {
    if (!pinKey.trim() || !pinVal.trim()) return
    setSavingPin(true)
    try {
      await onPinInvariant({
        key: pinKey.trim(),
        value: pinVal.trim(),
        source_node_id: pinLinkActive ? activeNodeId : null,
      })
      setPinKey('')
      setPinVal('')
      setShowNewPin(false)
    } finally {
      setSavingPin(false)
    }
  }

  const handleSaveFact = async () => {
    if (!activeNodeId || !factText.trim()) return
    setSavingFact(true)
    try {
      const ents = factEntities
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
      await onAddFact({
        node_id: activeNodeId,
        fact: factText.trim(),
        resolved_date: factDate.trim() || undefined,
        entities: ents,
      })
      setFactText('')
      setFactDate('')
      setFactEntities('')
      setShowNewFact(false)
    } finally {
      setSavingFact(false)
    }
  }

  return (
    <aside className="w-72 xl:w-84 shrink-0 border-l border-zinc-800/90 bg-zinc-950/95 flex flex-col h-full min-h-0">
      {/* Tabs */}
      <div className="h-12 px-1.5 border-b border-zinc-800/80 flex items-center gap-0.5">
        <button
          type="button"
          onClick={() => setTab('reflog')}
          className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
            tab === 'reflog'
              ? 'bg-zinc-800 text-zinc-100'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
          }`}
        >
          <History className="w-3 h-3 shrink-0" />
          <span>Reflog</span>
          <span className="font-mono text-[9px] text-zinc-500">{reflog.length}</span>
        </button>
        <button
          type="button"
          onClick={() => setTab('ipc')}
          className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
            tab === 'ipc'
              ? 'bg-zinc-800 text-zinc-100'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
          }`}
        >
          <Activity className="w-3 h-3 shrink-0" />
          <span>IPC</span>
          <span className="font-mono text-[9px] text-zinc-500">{ipcEvents.length}</span>
        </button>
        <button
          type="button"
          onClick={() => setTab('artifacts')}
          className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
            tab === 'artifacts'
              ? 'bg-zinc-800 text-zinc-100'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
          }`}
        >
          <FileCode2 className="w-3 h-3 shrink-0" />
          <span>Board</span>
          <span className="font-mono text-[9px] text-zinc-500">{artifacts.length}</span>
        </button>
        <button
          type="button"
          onClick={() => setTab('memory')}
          title="PINNED Invariants & Tier-2.5 Atomic Temporal Facts"
          className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
            tab === 'memory'
              ? 'bg-zinc-800 text-zinc-100'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
          }`}
        >
          <Pin className="w-3 h-3 shrink-0 text-indigo-400" />
          <span>Memory</span>
          <span
            className={`font-mono text-[9px] ${
              stalePinCount > 0 ? 'text-amber-400 font-semibold' : 'text-zinc-500'
            }`}
          >
            {memoryCount}
          </span>
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {tab === 'reflog' && (
          <>
            <div className="text-[11px] text-zinc-500 pb-1">
              Click any entry to jump <code className="font-mono text-zinc-400">HEAD</code>. Use{' '}
              <kbd className="px-1 rounded bg-zinc-900 border border-zinc-800 font-mono text-[10px]">
                Alt+←/→
              </kbd>{' '}
              for back/forward.
            </div>
            {reflog.map((entry, idx) => {
              const isCursor = idx === reflogCursor
              return (
                <button
                  key={entry.id}
                  type="button"
                  onClick={() => onJumpReflog(entry.head_ref, `Jump via ${entry.head_ref}`)}
                  className={`w-full text-left rounded-lg border p-2.5 text-xs transition-colors ${
                    isCursor
                      ? 'border-indigo-500/50 bg-indigo-500/10'
                      : 'border-zinc-800/80 bg-zinc-900/40 hover:bg-zinc-900'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-[11px] text-indigo-300 font-semibold">
                      {entry.head_ref}
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono text-[10px]">
                      {entry.action}
                    </span>
                  </div>
                  <div className="mt-1 font-medium text-zinc-200 truncate">
                    → {nodeNames[entry.to_node_id] || entry.to_node_id}
                  </div>
                  {entry.note && (
                    <div className="mt-0.5 text-[11px] text-zinc-400 line-clamp-2">
                      {entry.note}
                    </div>
                  )}
                </button>
              )
            })}
          </>
        )}

        {tab === 'ipc' && (
          <>
            {ipcEvents.length === 0 && (
              <div className="text-xs text-zinc-500 text-center py-8">
                No cross-node IPC events recorded yet. Spawn a branch, run a shadow query, or finish
                a child branch to inspect live traces.
              </div>
            )}
            {ipcEvents.map((ev) => {
              const open = !!expandedIpc[ev.id]
              const blocked = ev.status.startsWith('rejected_') || ev.status === 'error'
              return (
                <div
                  key={ev.id}
                  className={`rounded-lg border text-xs ${
                    blocked
                      ? 'border-rose-500/40 bg-rose-950/20'
                      : 'border-zinc-800/80 bg-zinc-900/40'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() =>
                      setExpandedIpc((prev) => ({ ...prev, [ev.id]: !prev[ev.id] }))
                    }
                    className="w-full p-2.5 text-left space-y-1"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-[11px] font-semibold text-indigo-300 flex items-center gap-1 truncate">
                        {blocked && <ShieldAlert className="w-3 h-3 text-rose-400 shrink-0" />}
                        {ev.event_type === 'custom_tool' && typeof ev.payload.tool === 'string'
                          ? ev.payload.tool
                          : ev.event_type}
                      </span>
                      <span className="font-mono text-[10px] text-zinc-500 flex items-center gap-1 shrink-0">
                        {ev.latency_ms}ms
                        {open ? (
                          <ChevronDown className="w-3 h-3" />
                        ) : (
                          <ChevronRight className="w-3 h-3" />
                        )}
                      </span>
                    </div>
                    <div className="text-[11px] text-zinc-300 truncate">
                      {nodeNames[ev.from_node_id] || ev.from_node_id} →{' '}
                      {ev.to_node_id ? nodeNames[ev.to_node_id] || ev.to_node_id : 'tree'}
                    </div>
                    <div className="flex items-center justify-between font-mono text-[10px] text-zinc-500">
                      <span>
                        hop {ev.hop_depth} · {ev.status}
                      </span>
                      <span>{ev.tokens_in + ev.tokens_out} tok</span>
                    </div>
                  </button>
                  {open && (
                    <div className="px-2.5 pb-2.5 pt-1 border-t border-zinc-800/70">
                      <pre className="p-2 rounded bg-zinc-950 font-mono text-[10px] text-zinc-300 overflow-x-auto max-h-56">
                        {JSON.stringify({ payload: ev.payload, response: ev.response }, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )
            })}
          </>
        )}

        {tab === 'artifacts' && (
          <>
            <div className="flex items-center justify-between pb-1">
              <span className="text-[11px] text-zinc-500">Versioned Blackboard</span>
              <button
                type="button"
                onClick={() => setShowNewArt(!showNewArt)}
                className="flex items-center gap-1 px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-[11px]"
              >
                <Plus className="w-3 h-3" />
                Add
              </button>
            </div>

            {showNewArt && (
              <div className="rounded-lg border border-zinc-700 bg-zinc-900/90 p-2.5 space-y-2 text-xs">
                <input
                  value={artSlug}
                  onChange={(e) => setArtSlug(e.target.value)}
                  placeholder="slug (e.g. schema.sql)"
                  className="w-full rounded bg-zinc-950 border border-zinc-800 px-2 py-1 font-mono text-xs text-zinc-100"
                />
                <div className="flex gap-1.5">
                  <input
                    value={artTitle}
                    onChange={(e) => setArtTitle(e.target.value)}
                    placeholder="Title"
                    className="flex-1 rounded bg-zinc-950 border border-zinc-800 px-2 py-1 text-xs text-zinc-100"
                  />
                  <select
                    value={artKind}
                    onChange={(e) => setArtKind(e.target.value)}
                    className="rounded bg-zinc-950 border border-zinc-800 px-1.5 py-1 text-xs text-zinc-200"
                  >
                    <option value="markdown">md</option>
                    <option value="sql">sql</option>
                    <option value="json">json</option>
                    <option value="code">code</option>
                  </select>
                </div>
                <textarea
                  value={artContent}
                  onChange={(e) => setArtContent(e.target.value)}
                  rows={4}
                  placeholder="Exact content (code, schema, JSON, spec)..."
                  className="w-full rounded bg-zinc-950 border border-zinc-800 p-2 font-mono text-[11px] text-zinc-100"
                />
                <div className="flex justify-end gap-1.5">
                  <button
                    type="button"
                    onClick={() => setShowNewArt(false)}
                    className="px-2 py-1 rounded text-zinc-400 hover:text-zinc-200"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    disabled={savingArt}
                    onClick={handleSaveArtifact}
                    className="px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-medium"
                  >
                    Save
                  </button>
                </div>
              </div>
            )}

            {artifacts.length === 0 && !showNewArt && (
              <div className="text-xs text-zinc-500 text-center py-8">
                No versioned artifacts yet. Artifacts attached on branch handoffs or added manually
                appear here as <code className="font-mono">slug@vN</code>.
              </div>
            )}

            {artifacts.map((art) => {
              const open = !!expandedArt[art.id]
              return (
                <div
                  key={art.id}
                  className="rounded-lg border border-zinc-800/80 bg-zinc-900/40 text-xs"
                >
                  <button
                    type="button"
                    onClick={() =>
                      setExpandedArt((prev) => ({ ...prev, [art.id]: !prev[art.id] }))
                    }
                    className="w-full p-2.5 text-left space-y-1"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-indigo-300 font-semibold truncate">
                        {art.slug}@v{art.version}
                      </span>
                      <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 font-mono text-[10px]">
                        {art.kind}
                      </span>
                    </div>
                    <div className="text-zinc-200 font-medium truncate">{art.title}</div>
                    {art.created_by_node_id && (
                      <div className="text-[10px] text-zinc-500">
                        from {nodeNames[art.created_by_node_id] || art.created_by_node_id}
                      </div>
                    )}
                  </button>
                  {open && (
                    <div className="px-2.5 pb-2.5 pt-1 border-t border-zinc-800/70 space-y-1.5">
                      <div className="flex justify-end">
                        <button
                          type="button"
                          onClick={() => navigator.clipboard.writeText(art.content)}
                          className="flex items-center gap-1 text-[10px] text-zinc-400 hover:text-zinc-100"
                        >
                          <Copy className="w-3 h-3" /> Copy
                        </button>
                      </div>
                      <pre className="p-2 rounded bg-zinc-950 font-mono text-[10px] text-zinc-200 overflow-x-auto max-h-56 whitespace-pre-wrap">
                        {art.content}
                      </pre>
                    </div>
                  )}
                </div>
              )
            })}
          </>
        )}

        {tab === 'memory' && (
          <div className="space-y-4">
            {/* Section 1: PINNED Tree-Wide Invariants */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <Pin className="w-3.5 h-3.5 text-indigo-400" />
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-300">
                    PINNED Invariants
                  </span>
                  <span className="font-mono text-[10px] text-zinc-500">{invariants.length}/8</span>
                </div>
                <button
                  type="button"
                  onClick={() => setShowNewPin(!showNewPin)}
                  className="flex items-center gap-1 px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-[11px]"
                >
                  <Plus className="w-3 h-3" />
                  Pin
                </button>
              </div>

              {showNewPin && (
                <div className="rounded-lg border border-indigo-500/35 bg-zinc-900/90 p-2.5 space-y-2 text-xs">
                  <input
                    value={pinKey}
                    onChange={(e) => setPinKey(e.target.value)}
                    placeholder="key (e.g. auth_signing)"
                    className="w-full rounded bg-zinc-950 border border-zinc-800 px-2 py-1 font-mono text-xs text-zinc-100"
                  />
                  <input
                    value={pinVal}
                    onChange={(e) => setPinVal(e.target.value)}
                    placeholder="value (e.g. Ed25519 keys rotated daily)"
                    className="w-full rounded bg-zinc-950 border border-zinc-800 px-2 py-1 text-xs text-zinc-100"
                  />
                  {activeNodeId && (
                    <label className="flex items-center gap-1.5 text-[11px] text-zinc-400 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={pinLinkActive}
                        onChange={(e) => setPinLinkActive(e.target.checked)}
                        className="rounded border-zinc-700 bg-zinc-950 text-indigo-500"
                      />
                      Track staleness from active branch (
                      <span className="truncate font-mono text-zinc-300">
                        {nodeNames[activeNodeId] || activeNodeId}
                      </span>
                      )
                    </label>
                  )}
                  <div className="flex justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => setShowNewPin(false)}
                      className="px-2 py-1 rounded text-zinc-400 hover:text-zinc-200"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      disabled={savingPin || !pinKey.trim() || !pinVal.trim()}
                      onClick={handleSavePin}
                      className="px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-medium disabled:opacity-40"
                    >
                      Save Pin
                    </button>
                  </div>
                </div>
              )}

              {invariants.length === 0 && !showNewPin && (
                <div className="rounded-lg border border-dashed border-zinc-800 p-2.5 text-[11px] text-zinc-500">
                  No pinned invariants yet. Pin key constraints (e.g.{' '}
                  <code className="font-mono text-zinc-400">db_engine=Postgres 16</code>) so every
                  branch sees them in its ephemeral KV-safe tail.
                </div>
              )}

              {invariants.map((inv) => {
                const staleReason = getInvariantStaleReason(inv)
                const stale = staleReason !== null
                return (
                  <div
                    key={inv.key}
                    className={`rounded-lg border p-2.5 text-xs space-y-1 ${
                      stale
                        ? 'border-amber-500/45 bg-amber-950/20'
                        : 'border-zinc-800/80 bg-zinc-900/50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="font-mono text-[11px] font-semibold text-indigo-300 truncate">
                          {inv.key}
                        </span>
                        {stale && (
                          <span
                            title={staleReason || 'Source branch changed after pinning'}
                            className="flex items-center gap-0.5 px-1 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono text-[9px] font-semibold shrink-0"
                          >
                            <AlertTriangle className="w-2.5 h-2.5" />
                            STALE
                          </span>
                        )}
                      </div>
                      <button
                        type="button"
                        title={`Unpin ${inv.key}`}
                        onClick={() => onUnpinInvariant(inv.key)}
                        className="p-1 rounded text-zinc-500 hover:text-rose-400 hover:bg-zinc-800 shrink-0"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                    <div className="text-zinc-200 leading-snug break-words">{inv.value}</div>
                    {inv.source_node_id && (
                      <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500 pt-0.5">
                        <button
                          type="button"
                          onClick={() => onJumpToNode(inv.source_node_id!)}
                          className="text-indigo-400 hover:underline truncate"
                        >
                          src: {nodeNames[inv.source_node_id] || inv.source_node_id}
                        </button>
                        <span>
                          v{inv.source_summary_version} · #{inv.source_msg_seq}
                        </span>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>

            {/* Section 2: Tier-2.5 Atomic Temporal Facts */}
            <div className="space-y-2 pt-3 border-t border-zinc-800/80">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-300">
                    Tier-2.5 Atomic Facts
                  </span>
                  <span className="font-mono text-[10px] text-zinc-500">{visibleFacts.length}</span>
                </div>
                {activeNodeId && (
                  <button
                    type="button"
                    onClick={() => setShowNewFact(!showNewFact)}
                    className="flex items-center gap-1 px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-[11px]"
                  >
                    <Plus className="w-3 h-3" />
                    Fact
                  </button>
                )}
              </div>

              <div className="flex items-center justify-between gap-2 text-[10px]">
                <div className="flex items-center rounded bg-zinc-900 border border-zinc-800 p-0.5">
                  <button
                    type="button"
                    onClick={() => setFactsScope('all')}
                    className={`px-2 py-0.5 rounded ${
                      factsScope === 'all' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500'
                    }`}
                  >
                    All branches
                  </button>
                  <button
                    type="button"
                    onClick={() => setFactsScope('active')}
                    className={`px-2 py-0.5 rounded ${
                      factsScope === 'active' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500'
                    }`}
                  >
                    Current
                  </button>
                </div>
                <label className="flex items-center gap-1 text-zinc-400 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={includeArchivedFacts}
                    onChange={(e) => onToggleIncludeArchivedFacts(e.target.checked)}
                    className="rounded border-zinc-700 bg-zinc-900 text-indigo-500"
                  />
                  Show archived
                </label>
              </div>

              {showNewFact && activeNodeId && (
                <div className="rounded-lg border border-emerald-500/35 bg-zinc-900/90 p-2.5 space-y-2 text-xs">
                  <div className="text-[10px] font-mono text-zinc-400">
                    Scoped to: {nodeNames[activeNodeId] || activeNodeId}
                  </div>
                  <textarea
                    value={factText}
                    onChange={(e) => setFactText(e.target.value)}
                    rows={2}
                    placeholder="Atomic fact (e.g. Migrated cache from Redis to Valkey)"
                    className="w-full rounded bg-zinc-950 border border-zinc-800 p-2 text-xs text-zinc-100"
                  />
                  <div className="grid grid-cols-2 gap-1.5">
                    <input
                      value={factDate}
                      onChange={(e) => setFactDate(e.target.value)}
                      placeholder="YYYY-MM-DD (opt)"
                      className="rounded bg-zinc-950 border border-zinc-800 px-2 py-1 font-mono text-[11px] text-zinc-100"
                    />
                    <input
                      value={factEntities}
                      onChange={(e) => setFactEntities(e.target.value)}
                      placeholder="Entities, comma-sep"
                      className="rounded bg-zinc-950 border border-zinc-800 px-2 py-1 text-[11px] text-zinc-100"
                    />
                  </div>
                  <div className="flex justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => setShowNewFact(false)}
                      className="px-2 py-1 rounded text-zinc-400 hover:text-zinc-200"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      disabled={savingFact || !factText.trim()}
                      onClick={handleSaveFact}
                      className="px-2.5 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-medium disabled:opacity-40"
                    >
                      Save Fact
                    </button>
                  </div>
                </div>
              )}

              {visibleFacts.length === 0 && !showNewFact && (
                <div className="rounded-lg border border-dashed border-zinc-800 p-2.5 text-[11px] text-zinc-500">
                  No active Tier-2.5 atomic facts yet. Facts are piggybacked automatically on branch
                  summaries (0 extra LLM calls) or can be added manually above.
                </div>
              )}

              {visibleFacts.map((f) => {
                const srcNode = nodesById[f.node_id]
                const isFromArchived = srcNode?.status === 'ARCHIVED'
                return (
                  <div
                    key={f.id}
                    className={`rounded-lg border p-2.5 text-xs space-y-1.5 ${
                      isFromArchived
                        ? 'border-zinc-800 bg-zinc-900/30 opacity-75'
                        : 'border-zinc-800/80 bg-zinc-900/50'
                    }`}
                  >
                    <div className="text-zinc-200 leading-snug">{f.fact_text}</div>
                    <div className="flex flex-wrap items-center gap-1">
                      {f.resolved_date && (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 font-mono text-[10px]">
                          <Calendar className="w-2.5 h-2.5" />
                          {f.resolved_date}
                        </span>
                      )}
                      {isFromArchived && (
                        <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 font-mono text-[9px]">
                          <Archive className="w-2.5 h-2.5" />
                          ARCHIVED
                        </span>
                      )}
                      {f.entities.map((ent) => (
                        <span
                          key={ent}
                          className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono text-[10px]"
                        >
                          {ent}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500">
                      <button
                        type="button"
                        onClick={() => onJumpToNode(f.node_id)}
                        className="text-indigo-400 hover:underline truncate"
                      >
                        {nodeNames[f.node_id] || f.node_id}
                      </button>
                      {f.source_msg_seq != null && <span>#{f.source_msg_seq}</span>}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}
