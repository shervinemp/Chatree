import { useMemo, useState } from 'react'
import {
  AlertTriangle,
  Archive,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleDot,
  GitBranch,
  Inbox,
  Minimize2,
  Plus,
  Search,
  Sparkles,
  Trash2,
} from 'lucide-react'
import type { NodeModel, TreeModel } from '../types'

interface TreeOutlineProps {
  trees: TreeModel[]
  activeTree: TreeModel | null
  nodes: NodeModel[]
  activeNodeId: string | null
  onSelectTree: (treeId: string) => void
  onCreateTree: () => void
  onDeleteTree: (treeId: string) => void
  onSelectNode: (nodeId: string) => void
  onSpawnFromNode: (node: NodeModel) => void
  onShadowQueryNode: (node: NodeModel) => void
  onCompactNode: (node: NodeModel) => void
  onArchiveNode: (node: NodeModel) => void
}

export function TreeOutline({
  trees,
  activeTree,
  nodes,
  activeNodeId,
  onSelectTree,
  onCreateTree,
  onDeleteTree,
  onSelectNode,
  onSpawnFromNode,
  onShadowQueryNode,
  onCompactNode,
  onArchiveNode,
}: TreeOutlineProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const [showArchived, setShowArchived] = useState(true)
  const archivedCount = useMemo(
    () => nodes.filter((n) => n.status === 'ARCHIVED').length,
    [nodes]
  )

  const { childrenByParent, roots, totals } = useMemo(() => {
    const map: Record<string, NodeModel[]> = {}
    const rootList: NodeModel[] = []
    let tokensIn = 0
    let tokensOut = 0
    let cacheRead = 0

    for (const n of nodes) {
      tokensIn += n.tokens_in
      tokensOut += n.tokens_out
      cacheRead += n.cache_read_tokens
      if (!n.parent_id) {
        rootList.push(n)
      } else {
        ;(map[n.parent_id] ||= []).push(n)
      }
    }
    const cacheRate = tokensIn > 0 ? Math.round((cacheRead / tokensIn) * 100) : 0
    return {
      childrenByParent: map,
      roots: rootList,
      totals: { tokensIn, tokensOut, cacheRead, cacheRate },
    }
  }, [nodes])

  const toggleCollapse = (id: string) =>
    setCollapsed((prev) => ({ ...prev, [id]: !prev[id] }))

  const renderNode = (node: NodeModel, level = 0) => {
    const isArchived = node.status === 'ARCHIVED'
    if (isArchived && !showArchived && node.id !== activeNodeId) return null
    const kids = childrenByParent[node.id] ?? []
    const isCollapsed = !!collapsed[node.id]
    const isHead = node.id === activeNodeId
    const isStale = node.stale_dependencies.length > 0
    const hasPending = node.pending_handoffs_count > 0

    return (
      <div key={node.id}>
        <div
          style={{ paddingLeft: `${level * 14 + 8}px` }}
          className={`relative group flex items-center gap-1.5 pr-2 py-1.5 rounded-md cursor-pointer text-xs transition-colors ${
            isHead
              ? 'bg-indigo-500/15 text-zinc-100 ring-1 ring-inset ring-indigo-500/35'
              : isArchived
                ? 'text-zinc-500 opacity-70 hover:opacity-100 hover:bg-zinc-900/70'
                : 'text-zinc-300 hover:bg-zinc-800/70'
          }`}
          onClick={() => onSelectNode(node.id)}
        >
          <button
            type="button"
            className="w-4 h-4 flex items-center justify-center text-zinc-500 hover:text-zinc-200 shrink-0"
            onClick={(e) => {
              e.stopPropagation()
              if (kids.length > 0) toggleCollapse(node.id)
            }}
            aria-label="Toggle children"
          >
            {kids.length > 0 ? (
              isCollapsed ? (
                <ChevronRight className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )
            ) : (
              <span className="w-1.5 h-1.5 rounded-full bg-zinc-700" />
            )}
          </button>

          {/* Status icon */}
          {isArchived ? (
            <Archive className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
          ) : node.status === 'COMPLETED' ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          ) : (
            <CircleDot className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          )}

          {/* Title */}
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <span className={`truncate font-medium ${isArchived ? 'line-through decoration-zinc-600' : ''}`}>
                {node.title}
              </span>
              {isHead && (
                <span className="px-1 py-0.2 rounded bg-indigo-500/25 text-indigo-300 font-mono text-[9px] uppercase tracking-wider shrink-0">
                  HEAD
                </span>
              )}
              {node.node_kind === 'subagent' && (
                <span
                  title="Isolated sub-agent worker node (queryable via shadow turn; auto-promotes if you chat inside it)"
                  className="inline-flex items-center gap-0.5 px-1 py-0.2 rounded bg-cyan-500/15 text-cyan-300 font-mono text-[9px] uppercase tracking-wider shrink-0"
                >
                  <Bot className="w-2.5 h-2.5" />
                  BOT
                </span>
              )}
              {isArchived && (
                <span className="px-1 py-0.2 rounded bg-zinc-800 text-zinc-400 font-mono text-[9px] uppercase shrink-0">
                  ARCH
                </span>
              )}
            </div>
            <div className="flex items-center gap-1.5 text-[10px] text-zinc-500 font-mono mt-0.5">
              <span>{node.id}</span>
              <span>·</span>
              <span>{node.context_mode}</span>
              {node.summary_version > 0 && (
                <>
                  <span>·</span>
                  <span className={node.is_dirty ? 'text-amber-400' : 'text-zinc-400'}>
                    v{node.summary_version}
                    {node.is_dirty ? '*' : ''}
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Badges */}
          <div className="flex items-center gap-1 shrink-0">
            {isStale && (
              <span
                title={`Stale dependencies: ${node.stale_dependencies
                  .map((d) => `${d.source_title} (${d.reason})`)
                  .join('; ')}`}
                className="flex items-center gap-0.5 px-1 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[9px] font-semibold"
              >
                <AlertTriangle className="w-2.5 h-2.5" />
                STALE
              </span>
            )}
            {hasPending && (
              <span
                title={`${node.pending_handoffs_count} pending PR handoff(s)`}
                className="flex items-center gap-0.5 px-1 py-0.5 rounded bg-fuchsia-500/20 text-fuchsia-300 text-[9px] font-semibold"
              >
                <Inbox className="w-2.5 h-2.5" />
                {node.pending_handoffs_count}
              </span>
            )}
          </div>

          {/* Hover actions (floating right overlay so title never shifts or traps row center-clicks) */}
          <div className="hidden group-hover:flex items-center gap-0.5 absolute right-1 top-1/2 -translate-y-1/2 z-10 rounded-md bg-zinc-900/95 border border-zinc-700/80 shadow-lg px-0.5 py-0.5 backdrop-blur-sm">
            <button
              type="button"
              title="Spawn child branch from this node"
              onClick={(e) => {
                e.stopPropagation()
                onSpawnFromNode(node)
              }}
              className="p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-zinc-100"
            >
              <GitBranch className="w-3 h-3" />
            </button>
            {node.id !== activeNodeId && (
              <button
                type="button"
                title="Run read-only shadow query against this node"
                onClick={(e) => {
                  e.stopPropagation()
                  onShadowQueryNode(node)
                }}
                className="p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-indigo-300"
              >
                <Search className="w-3 h-3" />
              </button>
            )}
            {!isArchived && (
              <button
                type="button"
                title="Non-destructively compact this branch"
                onClick={(e) => {
                  e.stopPropagation()
                  onCompactNode(node)
                }}
                className="p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-emerald-300"
              >
                <Minimize2 className="w-3 h-3" />
              </button>
            )}
            {node.parent_id && !isArchived && (
              <button
                type="button"
                title="Archive dead-end branch"
                onClick={(e) => {
                  e.stopPropagation()
                  onArchiveNode(node)
                }}
                className="p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-rose-300"
              >
                <Archive className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        {!isCollapsed && kids.length > 0 && (
          <div className="mt-0.5 space-y-0.5">
            {kids.map((child) => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <aside className="w-60 xl:w-72 shrink-0 border-r border-zinc-800/90 bg-zinc-950/95 flex flex-col h-full min-h-0">
      {/* Brand + Tree Picker */}
      <div className="p-3 border-b border-zinc-800/80 space-y-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-sm">
              <GitBranch className="w-3.5 h-3.5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-zinc-100 leading-none">
                Chatree
              </h1>
              <p className="text-[10px] text-zinc-500 mt-0.5">Dialogue Process Tree</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onCreateTree}
            title="New Dialogue Tree"
            className="flex items-center gap-1 px-2 py-1 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            New
          </button>
        </div>

        {/* Session selector */}
        <div className="flex items-center gap-1.5">
          <select
            value={activeTree?.id ?? ''}
            onChange={(e) => e.target.value && onSelectTree(e.target.value)}
            aria-label="Select dialogue tree"
            className="flex-1 min-w-0 bg-zinc-900 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            {trees.map((t) => (
              <option key={t.id} value={t.id}>
                {t.title}
              </option>
            ))}
          </select>
          {activeTree && trees.length > 1 && (
            <button
              type="button"
              onClick={() => onDeleteTree(activeTree.id)}
              title="Delete current tree"
              className="p-1.5 rounded-md border border-zinc-800 text-zinc-500 hover:text-rose-400 hover:border-rose-500/40 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Tree outline */}
      <div className="flex-1 overflow-y-auto px-2 py-2.5 space-y-1">
        <div className="px-2 pb-1 flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
          <span>Branch Outline ({nodes.length})</span>
          {archivedCount > 0 ? (
            <button
              type="button"
              onClick={() => setShowArchived((v) => !v)}
              className="font-mono text-[9px] text-zinc-400 hover:text-zinc-200 normal-case"
            >
              {showArchived ? `Hide arch (${archivedCount})` : `Show arch (${archivedCount})`}
            </button>
          ) : (
            <span className="font-mono text-[9px] text-zinc-600">Alt+←/→</span>
          )}
        </div>
        {roots.map((r) => renderNode(r, 0))}
      </div>

      {/* Economics footer */}
      <div className="p-3 border-t border-zinc-800/80 bg-zinc-900/40 text-[11px] space-y-1.5">
        <div className="flex items-center justify-between text-zinc-400">
          <span className="flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-indigo-400" />
            Tree Economics
          </span>
          <span className="font-mono text-emerald-400">
            {totals.cacheRate}% KV cache
          </span>
        </div>
        <div className="flex items-center justify-between font-mono text-[10px] text-zinc-500">
          <span>In: {totals.tokensIn.toLocaleString()}</span>
          <span>Out: {totals.tokensOut.toLocaleString()}</span>
          <span>Cached: {totals.cacheRead.toLocaleString()}</span>
        </div>
      </div>
    </aside>
  )
}
