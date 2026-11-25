import { useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  Archive,
  ChevronDown,
  ChevronRight,
  Cpu,
  GitBranch,
  Layers,
  Minimize2,
  RefreshCw,
  Search,
  Send,
  ShieldAlert,
  Sparkles,
  User,
} from 'lucide-react'
import type {
  IpcEventModel,
  MailboxItemModel,
  MessageModel,
  NodeModel,
} from '../types'
import { HandoffCard } from './HandoffCard'
import { MarkdownText } from './MarkdownText'

interface ChatPaneProps {
  node: NodeModel | null
  allNodes: NodeModel[]
  inheritedMessages: MessageModel[]
  messages: MessageModel[]
  pendingHandoffs: MailboxItemModel[]
  ipcEventsById: Record<string, IpcEventModel>
  streamingText: string
  streamingIpc: IpcEventModel[]
  ephemeralNotices: string[]
  isStreaming: boolean
  onSendMessage: (content: string) => void
  onRefreshStale: (sourceNodeId: string) => Promise<void>
  onDismissStale: (sourceNodeId: string) => Promise<void>
  onCommitHandoff: (mailboxId: string, summary: string, returnHead: boolean) => Promise<void>
  onDismissHandoff: (mailboxId: string) => Promise<void>
  onOpenSpawn: () => void
  onOpenShadowQuery: () => void
  onOpenAskTree: () => void
  onOpenCompact: () => void
  onArchiveNode: () => void
  archiving: boolean
  onJumpToNode: (nodeId: string) => void
}

function IpcChip({
  ev,
  nodeNames,
  onJumpToNode,
}: {
  ev: IpcEventModel
  nodeNames: Record<string, string>
  onJumpToNode: (id: string) => void
}) {
  const [open, setOpen] = useState(false)
  const isBlocked = ev.status.startsWith('rejected_') || ev.status === 'error'
  const toLabel = ev.to_node_id ? nodeNames[ev.to_node_id] || ev.to_node_id : 'tree'
  const customToolName = typeof ev.payload.tool === 'string' ? ev.payload.tool : 'custom_tool'
  const labelMap: Record<string, string> = {
    spawn: ev.payload.compact
      ? `dialogue_tree_compact_branch → ${toLabel}`
      : `dialogue_tree_create_branch → ${toLabel}`,
    send: `dialogue_tree_send_to_branch → ${toLabel}`,
    send_finish: `dialogue_tree_send_to_branch(finish=True) → ${toLabel}`,
    request_summary: `dialogue_tree_query_branch(summary) → ${toLabel}`,
    request_shadow: `dialogue_tree_query_branch(shadow) → ${toLabel}`,
    request_mutate: `dialogue_tree_query_branch(mutate) → ${toLabel}`,
    search: Array.isArray(ev.response.targets_queried)
      ? `dialogue_tree_search_and_query("${String(ev.payload.query ?? '')}")`
      : `dialogue_tree_search_branches("${String(ev.payload.query ?? '')}")`,
    reflog_jump: `dialogue_tree_switch_branch → ${toLabel}`,
    custom_tool: `${customToolName} → ${toLabel}`,
  }

  return (
    <div
      className={`rounded-lg border text-xs transition-colors ${
        isBlocked
          ? 'border-rose-500/40 bg-rose-950/25 text-rose-200'
          : 'border-indigo-500/30 bg-indigo-950/20 text-indigo-200'
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full px-2.5 py-1.5 flex items-center justify-between gap-2 text-left"
      >
        <div className="flex items-center gap-1.5 min-w-0">
          {isBlocked ? (
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400 shrink-0" />
          ) : (
            <Cpu className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          )}
          <span className="font-mono truncate">{labelMap[ev.event_type] || ev.event_type}</span>
          {ev.status === 'pending_review' && (
            <span className="px-1.5 py-0.2 rounded bg-fuchsia-500/20 text-fuchsia-300 font-mono text-[10px]">
              review
            </span>
          )}
          {isBlocked && (
            <span className="px-1.5 py-0.2 rounded bg-rose-500/20 text-rose-300 font-mono text-[10px]">
              {ev.status}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-zinc-400 shrink-0">
          <span>
            {ev.tokens_in + ev.tokens_out === 0
              ? '0 tok'
              : `${ev.tokens_in + ev.tokens_out} tok`}
          </span>
          <span>{ev.latency_ms}ms</span>
          {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        </div>
      </button>

      {open && (
        <div className="px-3 pb-2.5 pt-1 border-t border-zinc-800/70 space-y-2 text-[11px]">
          <div className="flex items-center justify-between text-zinc-400 font-mono">
            <span>
              hop={ev.hop_depth} · trace={ev.trace_id}
            </span>
            {ev.to_node_id && (
              <button
                type="button"
                onClick={() => onJumpToNode(ev.to_node_id!)}
                className="text-indigo-400 hover:underline"
              >
                Jump to {ev.to_node_id} →
              </button>
            )}
          </div>
          <pre className="p-2 rounded bg-zinc-950/90 border border-zinc-800 overflow-x-auto font-mono text-[10px] text-zinc-300 max-h-48">
            {JSON.stringify({ payload: ev.payload, response: ev.response }, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

export function ChatPane({
  node,
  allNodes,
  inheritedMessages,
  messages,
  pendingHandoffs,
  ipcEventsById,
  streamingText,
  streamingIpc,
  ephemeralNotices,
  isStreaming,
  onSendMessage,
  onRefreshStale,
  onDismissStale,
  onCommitHandoff,
  onDismissHandoff,
  onOpenSpawn,
  onOpenShadowQuery,
  onOpenAskTree,
  onOpenCompact,
  onArchiveNode,
  archiving,
  onJumpToNode,
}: ChatPaneProps) {
  const [input, setInput] = useState('')
  const [showInherited, setShowInherited] = useState(false)
  const [refreshingId, setRefreshingId] = useState<string | null>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  const nodeNames: Record<string, string> = {}
  for (const n of allNodes) nodeNames[n.id] = n.title

  useEffect(() => {
    const el = scrollContainerRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages.length, streamingText, streamingIpc.length, pendingHandoffs.length])

  const submit = () => {
    const text = input.trim()
    if (!text || isStreaming) return
    setInput('')
    onSendMessage(text)
  }

  if (!node) {
    return (
      <div className="flex-1 flex items-center justify-center text-zinc-500 text-sm">
        Create or select a dialogue tree to begin.
      </div>
    )
  }

  // Filter out raw tool role rows in the main conversational view — we display rich IPC chips inline!
  const visibleMessages = messages.filter(
    (m) => m.role === 'user' || (m.role === 'assistant' && m.content.trim().length > 0)
  )

  return (
    <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden bg-zinc-950">
      {/* Node Header Strip */}
      <div className="px-4 py-2.5 border-b border-zinc-800/70 bg-zinc-900/30 flex items-center justify-between gap-3 shrink-0 overflow-hidden">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-sm font-semibold text-zinc-100 truncate max-w-[240px]">
              {node.title}
            </h2>
            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 shrink-0">
              {node.id}
            </span>
            <span
              className={`font-mono text-[10px] px-1.5 py-0.5 rounded shrink-0 ${
                node.status === 'ARCHIVED'
                  ? 'bg-zinc-800 text-zinc-400 border border-zinc-700'
                  : node.status === 'COMPLETED'
                    ? 'bg-indigo-500/20 text-indigo-300'
                    : 'bg-emerald-500/15 text-emerald-300'
              }`}
            >
              {node.status}
            </span>
            {node.tool_permission === 'read_write' && (
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-300 shrink-0">
                rw
              </span>
            )}
            {node.summary_version > 0 && (
              <span className="font-mono text-[10px] text-zinc-500 shrink-0">
                v{node.summary_version}
                {node.is_dirty ? ' (dirty)' : ' (clean)'}
              </span>
            )}
          </div>
          {node.goal && (
            <p className="text-xs text-zinc-400 truncate mt-0.5">Goal: {node.goal}</p>
          )}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            type="button"
            onClick={onOpenAskTree}
            disabled={allNodes.length < 2}
            title="Ask across tree via unified TreeQueryEngine (Hop-1 + Tier-2.5 facts + synthesis)"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/35 text-xs text-indigo-200 transition-colors disabled:opacity-40"
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-300 shrink-0" />
            <span className="hidden md:inline">Ask Tree</span>
          </button>
          <button
            type="button"
            onClick={onOpenShadowQuery}
            disabled={allNodes.length < 2}
            title="Run read-only shadow query against a specific node"
            className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-xs text-zinc-300 transition-colors disabled:opacity-40"
          >
            <Search className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <span className="hidden xl:inline">Shadow</span>
          </button>
          {node.status !== 'ARCHIVED' && (
            <button
              type="button"
              onClick={onOpenCompact}
              title="Non-destructively compact this branch into a linked continuation child"
              className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-xs text-zinc-300 hover:text-emerald-300 transition-colors"
            >
              <Minimize2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              <span className="hidden lg:inline">Compact</span>
            </button>
          )}
          {node.parent_id && node.status !== 'ARCHIVED' && (
            <button
              type="button"
              disabled={archiving}
              onClick={onArchiveNode}
              title="Archive dead-end branch (quarantine from default 4-tier RAG & active facts)"
              className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-zinc-900 hover:bg-rose-950/40 border border-zinc-800 hover:border-rose-500/40 text-xs text-zinc-400 hover:text-rose-300 transition-colors disabled:opacity-40"
            >
              <Archive className="w-3.5 h-3.5 shrink-0" />
              <span className="hidden xl:inline">{archiving ? 'Archiving…' : 'Archive'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Archived Dead-End Quarantine Banner */}
      {node.status === 'ARCHIVED' && (
        <div className="px-4 py-2 border-b border-zinc-800 bg-zinc-900/70 shrink-0">
          <div className="max-w-3xl mx-auto flex items-center gap-2 text-xs text-zinc-400">
            <Archive className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
            <span>
              <strong className="text-zinc-200">Archived Dead-End Branch</strong> — quarantined from
              default 4-tier RAG, branch summaries, and active atomic facts while preserving verbatim
              history for audit.
            </span>
          </div>
        </div>
      )}

      {/* Pinned Stale Dependency Banners (always visible above scroll fold) */}
      {node.stale_dependencies.length > 0 && (
        <div className="px-4 py-2.5 border-b border-amber-500/30 bg-amber-500/10 shrink-0 space-y-2">
          {node.stale_dependencies.map((dep) => (
            <div
              key={dep.source_node_id}
              className="max-w-3xl mx-auto flex flex-wrap items-center justify-between gap-2"
            >
              <div className="flex items-start gap-2 min-w-0">
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <div className="text-xs min-w-0">
                  <div className="font-semibold text-amber-200 truncate">
                    Stale Dependency: {dep.source_title} ({dep.source_node_id})
                  </div>
                  <div className="text-amber-200/80 text-[11px]">{dep.reason}</div>
                </div>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  type="button"
                  disabled={refreshingId === dep.source_node_id}
                  onClick={async () => {
                    setRefreshingId(dep.source_node_id)
                    try {
                      await onRefreshStale(dep.source_node_id)
                    } finally {
                      setRefreshingId(null)
                    }
                  }}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-amber-500/20 hover:bg-amber-500/30 border border-amber-400/40 text-amber-100 text-xs font-medium"
                >
                  <RefreshCw
                    className={`w-3 h-3 ${refreshingId === dep.source_node_id ? 'animate-spin' : ''}`}
                  />
                  Refresh via request()
                </button>
                <button
                  type="button"
                  onClick={() => onDismissStale(dep.source_node_id)}
                  className="px-2 py-1 rounded-md text-xs text-amber-300/80 hover:text-amber-100 hover:bg-amber-500/15"
                >
                  Dismiss
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Scrollable Thread */}
      <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-4">
        <div className="max-w-3xl mx-auto space-y-4">
          {/* Pending PR-Style Handoff Review Cards */}
          {pendingHandoffs.map((mb) => (
            <HandoffCard
              key={mb.id}
              item={mb}
              fromNodeTitle={nodeNames[mb.from_node_id] || mb.from_node_id}
              toNodeTitle={nodeNames[mb.to_node_id] || mb.to_node_id}
              onCommit={onCommitHandoff}
              onDismiss={onDismissHandoff}
            />
          ))}

          {/* Collapsible Copy-on-Write Inherited Prefix */}
          {inheritedMessages.length > 0 && (
            <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 overflow-hidden">
              <button
                type="button"
                onClick={() => setShowInherited(!showInherited)}
                className="w-full px-3.5 py-2 flex items-center justify-between text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
              >
                <span className="flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5 text-indigo-400" />
                  <span>
                    Inherited <strong>{inheritedMessages.length}</strong> parent message(s) (
                    <code className="font-mono text-indigo-300">{node.context_mode}</code> prefix ·
                    copy-on-write)
                  </span>
                </span>
                {showInherited ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
              </button>
              {showInherited && (
                <div className="px-3.5 pb-3 pt-1 border-t border-zinc-800/70 space-y-2 max-h-72 overflow-y-auto">
                  {inheritedMessages
                    .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
                    .map((m) => (
                      <div key={m.id} className="text-xs text-zinc-400">
                        <span className="font-mono text-[10px] uppercase text-zinc-500 mr-2">
                          {m.role}:
                        </span>
                        <span className="whitespace-pre-wrap">{m.content}</span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}

          {/* Ephemeral orientation notices from the latest turn */}
          {ephemeralNotices.length > 0 && (
            <div className="rounded-lg border border-indigo-500/25 bg-indigo-500/5 px-3 py-2 text-[11px] text-indigo-200/90 space-y-1">
              <div className="font-mono text-[10px] uppercase tracking-wider text-indigo-400">
                Ephemeral Tail Orientation (KV-Cache Safe)
              </div>
              {ephemeralNotices.map((n, i) => (
                <div key={i}>• {n}</div>
              ))}
            </div>
          )}

          {/* Empty state */}
          {visibleMessages.length === 0 && !isStreaming && (
            <div className="py-12 text-center space-y-3">
              <div className="w-10 h-10 mx-auto rounded-xl bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-indigo-400" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium text-zinc-200">
                  {node.depth === 0 ? 'Start a hierarchical dialogue' : `Branch ready: ${node.title}`}
                </p>
                <p className="text-xs text-zinc-500 max-w-md mx-auto">
                  Try chatting normally, asking the model to{' '}
                  <code className="font-mono text-indigo-300">branch off to explore ...</code>, or
                  using the toolbar to spawn isolated child branches and run read-only shadow queries.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2 pt-2">
                <button
                  type="button"
                  onClick={() =>
                    onSendMessage('Let’s design a resilient auth system using short-lived JWTs.')
                  }
                  className="px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-xs text-zinc-300"
                >
                  “Design an auth system…”
                </button>
                <button
                  type="button"
                  onClick={() =>
                    onSendMessage('Please branch off to design the database schema')
                  }
                  className="px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-xs text-zinc-300"
                >
                  “Branch off to design the DB schema”
                </button>
              </div>
            </div>
          )}

          {/* Messages */}
          {visibleMessages.map((msg) => {
            const isUser = msg.role === 'user'
            const isHandoff = Boolean(msg.metadata?.ipc_handoff)
            const attachedIpcIds = Array.isArray(msg.metadata?.ipc_events)
              ? (msg.metadata.ipc_events as string[])
              : []
            const attachedIpcs = attachedIpcIds
              .map((id) => ipcEventsById[id])
              .filter(Boolean)

            return (
              <div key={msg.id} className="space-y-2">
                {attachedIpcs.length > 0 && (
                  <div className="space-y-1.5 pl-9">
                    {attachedIpcs.map((ev) => (
                      <IpcChip
                        key={ev.id}
                        ev={ev}
                        nodeNames={nodeNames}
                        onJumpToNode={onJumpToNode}
                      />
                    ))}
                  </div>
                )}

                <div
                  className={`flex gap-3 ${
                    isHandoff
                      ? 'rounded-xl border border-emerald-500/35 bg-emerald-950/20 p-3.5'
                      : isUser
                        ? 'rounded-xl bg-zinc-900/90 border border-zinc-800/80 p-3.5'
                        : 'p-1'
                  }`}
                >
                  <div
                    className={`w-6 h-6 rounded-md flex items-center justify-center shrink-0 mt-0.5 ${
                      isHandoff
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : isUser
                          ? 'bg-zinc-800 text-zinc-300'
                          : 'bg-indigo-500/20 text-indigo-300'
                    }`}
                  >
                    {isHandoff ? (
                      <GitBranch className="w-3.5 h-3.5" />
                    ) : isUser ? (
                      <User className="w-3.5 h-3.5" />
                    ) : (
                      <Sparkles className="w-3.5 h-3.5" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1 space-y-1.5">
                    <div className="flex items-center justify-between gap-2 text-[11px]">
                      <span className="font-medium text-zinc-300">
                        {isHandoff
                          ? `Merged Child Handoff (${String(msg.metadata?.from_node_title || '')})`
                          : isUser
                            ? 'You'
                            : 'Assistant'}
                      </span>
                      <span className="font-mono text-[10px] text-zinc-500">
                        #{msg.seq}
                        {msg.model_used ? ` · ${msg.model_used}` : ''}
                        {msg.tokens_out > 0 ? ` · ${msg.tokens_in}/${msg.tokens_out} tok` : ''}
                        {msg.cache_read_tokens > 0
                          ? ` · ${msg.cache_read_tokens} cached`
                          : ''}
                      </span>
                    </div>
                    <MarkdownText content={msg.content} />
                  </div>
                </div>
              </div>
            )
          })}

          {/* Live streaming IPC chips & assistant output */}
          {isStreaming && (
            <div className="space-y-2">
              {streamingIpc.length > 0 && (
                <div className="space-y-1.5 pl-9">
                  {streamingIpc.map((ev) => (
                    <IpcChip
                      key={ev.id}
                      ev={ev}
                      nodeNames={nodeNames}
                      onJumpToNode={onJumpToNode}
                    />
                  ))}
                </div>
              )}
              <div className="flex gap-3 p-1">
                <div className="w-6 h-6 rounded-md bg-indigo-500/20 text-indigo-300 flex items-center justify-center shrink-0 mt-0.5">
                  <Sparkles className="w-3.5 h-3.5 animate-pulse-soft" />
                </div>
                <div className="min-w-0 flex-1 space-y-1">
                  <div className="text-[11px] font-medium text-indigo-300">
                    Assistant <span className="text-zinc-500 font-normal">streaming…</span>
                  </div>
                  {streamingText ? (
                    <MarkdownText content={streamingText} />
                  ) : (
                    <div className="text-xs text-zinc-500 animate-pulse-soft">
                      Thinking / routing IPC…
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

        </div>
      </div>

      {/* Composer */}
      <div className="p-4 border-t border-zinc-800/80 bg-zinc-950/95">
        <div className="max-w-3xl mx-auto space-y-2">
          <div className="flex items-end gap-2 rounded-xl border border-zinc-800 bg-zinc-900/80 p-2 focus-within:border-indigo-500/60 focus-within:ring-1 focus-within:ring-indigo-500/30 transition-all">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  submit()
                }
              }}
              rows={2}
              disabled={isStreaming}
              placeholder={`Message "${node.title}"… (Enter to send, Shift+Enter newline)`}
              className="flex-1 bg-transparent resize-none px-2 py-1 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none max-h-40"
            />
            <button
              type="button"
              onClick={submit}
              disabled={isStreaming || !input.trim()}
              className="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors disabled:opacity-40 disabled:hover:bg-indigo-600 shrink-0"
              aria-label="Send message"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="flex items-center justify-between text-[11px] text-zinc-500 px-1">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={onOpenSpawn}
                className="hover:text-indigo-300 transition-colors"
              >
                + Spawn branch
              </button>
              {allNodes.length > 1 && (
                <button
                  type="button"
                  onClick={onOpenShadowQuery}
                  className="hover:text-indigo-300 transition-colors"
                >
                  ? Shadow query another branch
                </button>
              )}
            </div>
            <span className="font-mono text-[10px]">
              {node.status === 'COMPLETED'
                ? 'Sending a turn will reopen this branch and notify dependents'
                : 'Prefix cached · Ephemeral orientation tail'}
            </span>
          </div>
        </div>
      </div>
    </main>
  )
}
