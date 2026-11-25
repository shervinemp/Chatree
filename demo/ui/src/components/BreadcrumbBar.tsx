import { useState } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Command,
  Compass,
  GitBranch,
  GitMerge,
  PanelRight,
  Plus,
  Settings,
  Trash2,
} from 'lucide-react'
import type { NodeModel, TreeModel, TreeSessionModel } from '../types'

interface BreadcrumbBarProps {
  tree: TreeModel | null
  nodes: NodeModel[]
  sessions: TreeSessionModel[]
  activeSessionId: string
  ancestors: NodeModel[]
  activeNode: NodeModel | null
  estimatedTokens: number
  inspectorOpen: boolean
  onSelectSession: (sessionId: string) => void
  onCreateSession: (title: string) => Promise<void>
  onDeleteSession: (sessionId: string) => Promise<void>
  onBack: () => void
  onForward: () => void
  onSelectNode: (nodeId: string) => void
  onOpenSpawn: () => void
  onFinishBranch: () => void
  onOpenPalette: () => void
  onToggleInspector: () => void
  onOpenSettings: () => void
  finishing: boolean
}

export function BreadcrumbBar({
  tree,
  nodes,
  sessions,
  activeSessionId,
  ancestors,
  activeNode,
  estimatedTokens,
  inspectorOpen,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onBack,
  onForward,
  onSelectNode,
  onOpenSpawn,
  onFinishBranch,
  onOpenPalette,
  onToggleInspector,
  onOpenSettings,
  finishing,
}: BreadcrumbBarProps) {
  const [sessionMenuOpen, setSessionMenuOpen] = useState(false)
  const [newSessionTitle, setNewSessionTitle] = useState('')
  const [creatingSession, setCreatingSession] = useState(false)

  const ctxWindow = 128_000
  const fillPct = Math.min(100, Math.max(1, Math.round((estimatedTokens / ctxWindow) * 100)))
  const pathNodes = activeNode ? [...ancestors, activeNode] : []
  const activeModel = activeNode?.model_override || tree?.default_model || 'mock/chatree-demo'
  const activeSession =
    sessions.find((s) => s.id === activeSessionId) ?? sessions[0] ?? null
  const nodeTitleById = new Map(nodes.map((n) => [n.id, n.title]))

  const handleAddSession = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = newSessionTitle.trim() || `Session ${sessions.length + 1}`
    setCreatingSession(true)
    try {
      await onCreateSession(trimmed)
      setNewSessionTitle('')
      setSessionMenuOpen(false)
    } finally {
      setCreatingSession(false)
    }
  }

  return (
    <header className="h-12 shrink-0 border-b border-zinc-800/90 bg-zinc-950/90 backdrop-blur px-3 flex items-center justify-between gap-2 relative z-20">
      {/* Left: Reflog Back/Forward + Visitor Session Switcher + Breadcrumbs */}
      <div className="flex items-center gap-1.5 min-w-0 flex-1">
        <div className="flex items-center border border-zinc-800 rounded-md overflow-hidden shrink-0">
          <button
            type="button"
            onClick={onBack}
            title="Reflog Back (Alt+Left)"
            className="p-1.5 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onForward}
            title="Reflog Forward (Alt+Right)"
            className="p-1.5 border-l border-zinc-800 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100 transition-colors"
          >
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Multi-Session Visitor Switcher */}
        {tree && (
          <div className="relative shrink-0">
            <button
              type="button"
              onClick={() => setSessionMenuOpen((v) => !v)}
              title="Switch or open concurrent visitor sessions on this shared tree"
              className="flex items-center gap-1.5 px-2 py-1 rounded-md border border-zinc-800 bg-zinc-900/90 hover:bg-zinc-800 text-xs text-zinc-200 transition-colors"
            >
              <Compass className="w-3.5 h-3.5 text-violet-400 shrink-0" />
              <span className="max-w-[110px] truncate font-medium">
                {activeSession?.title || 'Main Session'}
              </span>
              {sessions.length > 1 && (
                <span className="px-1 rounded bg-violet-500/20 text-violet-300 font-mono text-[10px]">
                  {sessions.length}
                </span>
              )}
              <ChevronDown className="w-3 h-3 text-zinc-500" />
            </button>

            {sessionMenuOpen && (
              <div className="absolute left-0 top-full mt-1.5 w-72 rounded-lg border border-zinc-700/90 bg-zinc-950/98 shadow-2xl p-2 z-50">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-semibold px-1.5 pb-1.5">
                  Tree Visitor Sessions
                </div>
                <div className="space-y-1 max-h-48 overflow-y-auto">
                  {sessions.map((s) => {
                    const isCur = s.id === activeSessionId
                    const headTitle = s.head_node_id
                      ? nodeTitleById.get(s.head_node_id) || s.head_node_id
                      : 'root'
                    return (
                      <div
                        key={s.id}
                        className={`flex items-center justify-between gap-1.5 px-2 py-1.5 rounded-md text-xs cursor-pointer transition-colors ${
                          isCur
                            ? 'bg-violet-500/20 text-zinc-100 ring-1 ring-inset ring-violet-500/40'
                            : 'text-zinc-300 hover:bg-zinc-800/80'
                        }`}
                        onClick={() => {
                          onSelectSession(s.id)
                          setSessionMenuOpen(false)
                        }}
                      >
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span className="truncate font-medium">{s.title}</span>
                            {s.is_default && (
                              <span className="px-1 rounded bg-zinc-800 text-zinc-400 font-mono text-[9px]">
                                default
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-zinc-500 truncate font-mono">
                            HEAD → {headTitle}
                          </div>
                        </div>
                        {!s.is_default && (
                          <button
                            type="button"
                            title="Close session"
                            onClick={(e) => {
                              e.stopPropagation()
                              void onDeleteSession(s.id)
                            }}
                            className="p-1 rounded text-zinc-500 hover:text-rose-400 hover:bg-zinc-800"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    )
                  })}
                </div>
                <form
                  onSubmit={handleAddSession}
                  className="mt-2 pt-2 border-t border-zinc-800 flex items-center gap-1.5"
                >
                  <input
                    type="text"
                    value={newSessionTitle}
                    onChange={(e) => setNewSessionTitle(e.target.value)}
                    placeholder="New session name…"
                    className="flex-1 min-w-0 px-2 py-1 rounded bg-zinc-900 border border-zinc-700 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-violet-500"
                  />
                  <button
                    type="submit"
                    disabled={creatingSession}
                    className="flex items-center gap-1 px-2 py-1 rounded bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium disabled:opacity-50"
                  >
                    <Plus className="w-3 h-3" />
                    Add
                  </button>
                </form>
              </div>
            )}
          </div>
        )}

        <nav className="flex items-center gap-1 min-w-0 overflow-hidden text-xs">
          {pathNodes.map((n, idx) => {
            const isLast = idx === pathNodes.length - 1
            const isCollapsedMiddle = pathNodes.length > 3 && idx > 0 && idx < pathNodes.length - 2
            if (isCollapsedMiddle) {
              if (idx === 1) {
                return (
                  <div key="ellipsis" className="flex items-center gap-1 shrink-0 text-zinc-600">
                    <ChevronRight className="w-3.5 h-3.5" />
                    <span className="px-1 font-mono text-[11px]">…</span>
                  </div>
                )
              }
              return null
            }
            return (
              <div key={n.id} className="flex items-center gap-1 min-w-0 shrink">
                {idx > 0 && <ChevronRight className="w-3.5 h-3.5 text-zinc-600 shrink-0" />}
                <button
                  type="button"
                  onClick={() => onSelectNode(n.id)}
                  title={n.title}
                  className={`px-1.5 py-0.5 rounded transition-colors truncate ${
                    isLast
                      ? 'bg-zinc-800/90 text-zinc-100 font-medium max-w-[150px] xl:max-w-[190px]'
                      : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 max-w-[100px] xl:max-w-[135px]'
                  }`}
                >
                  {n.title}
                </button>
              </div>
            )
          })}
        </nav>

        {activeNode && (
          <div className="hidden xl:flex items-center gap-2 ml-2 pl-2 border-l border-zinc-800 text-[11px] font-mono text-zinc-500 shrink-0">
            <span className="text-zinc-400">{activeModel}</span>
            <span className="flex items-center gap-1.5" title={`~${estimatedTokens.toLocaleString()} tokens in branch prefix`}>
              <span className="w-12 h-1.5 rounded-full bg-zinc-800 overflow-hidden inline-block">
                <span
                  className={`block h-full ${
                    fillPct > 70 ? 'bg-amber-400' : 'bg-emerald-400'
                  }`}
                  style={{ width: `${Math.max(4, fillPct)}%` }}
                />
              </span>
              <span>{estimatedTokens.toLocaleString()} tok</span>
            </span>
          </div>
        )}
      </div>

      {/* Right: Branch actions + Command Palette + Inspector + Settings */}
      <div className="flex items-center gap-1.5 shrink-0">
        {activeNode && (
          <button
            type="button"
            onClick={onOpenSpawn}
            title="Spawn Child Branch"
            className="flex items-center gap-1.5 px-2 lg:px-2.5 py-1.5 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-100 text-xs font-medium border border-zinc-700/70 transition-colors"
          >
            <GitBranch className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <span className="hidden lg:inline">Spawn Child</span>
          </button>
        )}

        {activeNode && activeNode.parent_id && activeNode.status !== 'ARCHIVED' && (
          <button
            type="button"
            disabled={finishing}
            onClick={onFinishBranch}
            title="Finish Branch & Return to Parent"
            className="flex items-center gap-1.5 px-2 lg:px-2.5 py-1.5 rounded-md bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/35 text-xs font-medium transition-colors disabled:opacity-50"
          >
            <GitMerge className="w-3.5 h-3.5 shrink-0" />
            <span className="hidden lg:inline">
              {finishing ? 'Summarizing…' : 'Finish & Return'}
            </span>
          </button>
        )}

        <button
          type="button"
          onClick={onOpenPalette}
          title="Command Palette & Tree Search (Ctrl+K)"
          className="flex items-center gap-1.5 px-2 py-1.5 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100 border border-zinc-800 text-xs transition-colors"
        >
          <Command className="w-3.5 h-3.5" />
          <span className="hidden md:inline font-mono text-[10px]">Ctrl+K</span>
        </button>

        <button
          type="button"
          onClick={onToggleInspector}
          title="Toggle Reflog / IPC / Artifacts Inspector"
          className={`p-1.5 rounded-md border transition-colors ${
            inspectorOpen
              ? 'bg-indigo-500/15 border-indigo-500/40 text-indigo-300'
              : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-zinc-100'
          }`}
        >
          <PanelRight className="w-4 h-4" />
        </button>

        <button
          type="button"
          onClick={onOpenSettings}
          title="Model Routing & Guardrails Settings"
          className="p-1.5 rounded-md bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-400 hover:text-zinc-100 transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
  )
}
