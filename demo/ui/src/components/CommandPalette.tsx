import { useEffect, useMemo, useState } from 'react'
import { FileCode2, GitBranch, History, Search, X } from 'lucide-react'
import { api } from '../api/client'
import type { ArtifactModel, NodeModel, ReflogEntryModel, SearchHit } from '../types'

interface CommandPaletteProps {
  treeId: string
  activeNodeId: string | null
  nodes: NodeModel[]
  reflog: ReflogEntryModel[]
  artifacts: ArtifactModel[]
  onClose: () => void
  onSelectNode: (nodeId: string) => void
  onJumpReflog: (headRef: string) => void
}

export function CommandPalette({
  treeId,
  activeNodeId,
  nodes,
  reflog,
  artifacts,
  onClose,
  onSelectNode,
  onJumpReflog,
}: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [ftsHits, setFtsHits] = useState<SearchHit[]>([])
  const [searching, setSearching] = useState(false)

  useEffect(() => {
    const q = query.trim()
    if (q.length < 2) {
      setFtsHits([])
      return
    }
    const timer = setTimeout(async () => {
      setSearching(true)
      try {
        const res = await api.searchTree(treeId, q, 'all', activeNodeId ?? undefined)
        setFtsHits(res)
      } catch {
        setFtsHits([])
      } finally {
        setSearching(false)
      }
    }, 180)
    return () => clearTimeout(timer)
  }, [query, treeId, activeNodeId])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const qLow = query.trim().toLowerCase()

  const matchedNodes = useMemo(
    () =>
      nodes.filter(
        (n) =>
          !qLow ||
          n.title.toLowerCase().includes(qLow) ||
          n.id.toLowerCase().includes(qLow) ||
          n.goal.toLowerCase().includes(qLow)
      ),
    [nodes, qLow]
  )

  const matchedReflog = useMemo(
    () =>
      reflog
        .filter(
          (r) =>
            !qLow ||
            r.head_ref.toLowerCase().includes(qLow) ||
            r.note.toLowerCase().includes(qLow) ||
            r.action.toLowerCase().includes(qLow)
        )
        .slice(0, 6),
    [reflog, qLow]
  )

  const matchedArtifacts = useMemo(
    () =>
      artifacts
        .filter(
          (a) =>
            !qLow ||
            a.slug.toLowerCase().includes(qLow) ||
            a.title.toLowerCase().includes(qLow)
        )
        .slice(0, 5),
    [artifacts, qLow]
  )

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-start justify-center pt-20 p-4"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-xl rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden"
      >
        <div className="px-4 py-3 border-b border-zinc-800 flex items-center gap-2.5">
          <Search className="w-4 h-4 text-indigo-400 shrink-0" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Jump to node, HEAD@{n}, artifact, or search full tree text…"
            className="flex-1 bg-transparent text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none"
          />
          <button type="button" onClick={onClose} className="text-zinc-500 hover:text-zinc-200">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="max-h-[60vh] overflow-y-auto p-2 space-y-3 text-xs">
          {/* FTS5 Search Hits */}
          {query.trim().length >= 2 && (
            <div>
              <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-indigo-400">
                Full-Text Tree Search {searching ? '(searching…)' : `(${ftsHits.length})`}
              </div>
              {ftsHits.map((hit, idx) => (
                <button
                  key={`${hit.node_id}-${hit.matched_seq ?? idx}`}
                  type="button"
                  onClick={() => {
                    onSelectNode(hit.node_id)
                    onClose()
                  }}
                  className="w-full text-left px-2.5 py-2 rounded-lg hover:bg-zinc-900 flex items-start justify-between gap-2"
                >
                  <div className="min-w-0">
                    <div className="font-medium text-zinc-200 flex items-center gap-1.5">
                      <span>{hit.node_title}</span>
                      <span className="font-mono text-[10px] text-zinc-500">{hit.node_id}</span>
                    </div>
                    <div className="text-[11px] text-zinc-400 truncate mt-0.5">{hit.snippet}</div>
                  </div>
                  <span className="font-mono text-[10px] text-indigo-300 shrink-0">
                    {hit.granularity ? `${hit.granularity} · ` : ''}
                    {hit.source}
                  </span>
                </button>
              ))}
            </div>
          )}

          {/* Nodes */}
          <div>
            <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
              Branches ({matchedNodes.length})
            </div>
            {matchedNodes.map((n) => (
              <button
                key={n.id}
                type="button"
                onClick={() => {
                  onSelectNode(n.id)
                  onClose()
                }}
                className="w-full text-left px-2.5 py-2 rounded-lg hover:bg-zinc-900 flex items-center justify-between gap-2"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <GitBranch className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                  <span className="font-medium text-zinc-200 truncate">{n.title}</span>
                  <span className="font-mono text-[10px] text-zinc-500">{n.id}</span>
                </div>
                <span className="font-mono text-[10px] text-zinc-400">{n.status}</span>
              </button>
            ))}
          </div>

          {/* Reflog */}
          {matchedReflog.length > 0 && (
            <div>
              <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                Recent Reflog
              </div>
              {matchedReflog.map((r) => (
                <button
                  key={r.id}
                  type="button"
                  onClick={() => {
                    onJumpReflog(r.head_ref)
                    onClose()
                  }}
                  className="w-full text-left px-2.5 py-2 rounded-lg hover:bg-zinc-900 flex items-center justify-between gap-2"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <History className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
                    <span className="font-mono text-indigo-300">{r.head_ref}</span>
                    <span className="text-zinc-300 truncate">{r.note || r.action}</span>
                  </div>
                  <span className="font-mono text-[10px] text-zinc-500">{r.action}</span>
                </button>
              ))}
            </div>
          )}

          {/* Artifacts */}
          {matchedArtifacts.length > 0 && (
            <div>
              <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                Artifacts
              </div>
              {matchedArtifacts.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  onClick={() => {
                    if (a.created_by_node_id) onSelectNode(a.created_by_node_id)
                    onClose()
                  }}
                  className="w-full text-left px-2.5 py-2 rounded-lg hover:bg-zinc-900 flex items-center justify-between gap-2"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <FileCode2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span className="font-mono text-indigo-300">
                      {a.slug}@v{a.version}
                    </span>
                    <span className="text-zinc-300 truncate">{a.title}</span>
                  </div>
                  <span className="font-mono text-[10px] text-zinc-500">{a.kind}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
