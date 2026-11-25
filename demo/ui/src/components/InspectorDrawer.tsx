import { useState } from 'react'
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Copy,
  FileCode2,
  History,
  Plus,
  ShieldAlert,
} from 'lucide-react'
import type {
  ArtifactModel,
  IpcEventModel,
  NodeModel,
  ReflogEntryModel,
} from '../types'

interface InspectorDrawerProps {
  nodes: NodeModel[]
  reflog: ReflogEntryModel[]
  reflogCursor: number
  ipcEvents: IpcEventModel[]
  artifacts: ArtifactModel[]
  activeNodeId: string | null
  onJumpReflog: (target: string, note?: string) => void
  onCreateArtifact: (payload: {
    slug: string
    title: string
    kind: string
    content: string
    summary: string
  }) => Promise<void>
}

export function InspectorDrawer({
  nodes,
  reflog,
  reflogCursor,
  ipcEvents,
  artifacts,
  onJumpReflog,
  onCreateArtifact,
}: InspectorDrawerProps) {
  const [tab, setTab] = useState<'reflog' | 'ipc' | 'artifacts'>('reflog')
  const [expandedIpc, setExpandedIpc] = useState<Record<string, boolean>>({})
  const [expandedArt, setExpandedArt] = useState<Record<string, boolean>>({})
  const [showNewArt, setShowNewArt] = useState(false)
  const [artSlug, setArtSlug] = useState('')
  const [artTitle, setArtTitle] = useState('')
  const [artKind, setArtKind] = useState('markdown')
  const [artContent, setArtContent] = useState('')
  const [savingArt, setSavingArt] = useState(false)

  const nodeNames: Record<string, string> = {}
  for (const n of nodes) nodeNames[n.id] = n.title

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

  return (
    <aside className="w-72 xl:w-80 shrink-0 border-l border-zinc-800/90 bg-zinc-950/95 flex flex-col h-full min-h-0">
      {/* Tabs */}
      <div className="h-12 px-2 border-b border-zinc-800/80 flex items-center gap-1">
        <button
          type="button"
          onClick={() => setTab('reflog')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
            tab === 'reflog'
              ? 'bg-zinc-800 text-zinc-100'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
          }`}
        >
          <History className="w-3.5 h-3.5" />
          Reflog
          <span className="font-mono text-[10px] text-zinc-500">{reflog.length}</span>
        </button>
        <button
          type="button"
          onClick={() => setTab('ipc')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
            tab === 'ipc'
              ? 'bg-zinc-800 text-zinc-100'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
          }`}
        >
          <Activity className="w-3.5 h-3.5" />
          IPC
          <span className="font-mono text-[10px] text-zinc-500">{ipcEvents.length}</span>
        </button>
        <button
          type="button"
          onClick={() => setTab('artifacts')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
            tab === 'artifacts'
              ? 'bg-zinc-800 text-zinc-100'
              : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
          }`}
        >
          <FileCode2 className="w-3.5 h-3.5" />
          Board
          <span className="font-mono text-[10px] text-zinc-500">{artifacts.length}</span>
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
                      <span className="font-mono text-[11px] font-semibold text-indigo-300 flex items-center gap-1">
                        {blocked && <ShieldAlert className="w-3 h-3 text-rose-400" />}
                        {ev.event_type}
                      </span>
                      <span className="font-mono text-[10px] text-zinc-500 flex items-center gap-1">
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
                      <span>
                        {ev.tokens_in + ev.tokens_out} tok
                      </span>
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
      </div>
    </aside>
  )
}
