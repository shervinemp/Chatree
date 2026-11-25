import {
  ArrowLeft,
  ArrowRight,
  ChevronRight,
  Command,
  GitBranch,
  GitMerge,
  PanelRight,
  Settings,
} from 'lucide-react'
import type { NodeModel, TreeModel } from '../types'

interface BreadcrumbBarProps {
  tree: TreeModel | null
  ancestors: NodeModel[]
  activeNode: NodeModel | null
  estimatedTokens: number
  inspectorOpen: boolean
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
  ancestors,
  activeNode,
  estimatedTokens,
  inspectorOpen,
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
  const ctxWindow = 128_000
  const fillPct = Math.min(100, Math.max(1, Math.round((estimatedTokens / ctxWindow) * 100)))
  const pathNodes = activeNode ? [...ancestors, activeNode] : []
  const activeModel = activeNode?.model_override || tree?.default_model || 'mock/chatree-demo'

  return (
    <header className="h-12 shrink-0 border-b border-zinc-800/90 bg-zinc-950/90 backdrop-blur px-3 flex items-center justify-between gap-2 overflow-hidden">
      {/* Left: Reflog Back/Forward + Breadcrumbs */}
      <div className="flex items-center gap-1.5 min-w-0 flex-1 overflow-hidden">
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
