import { useState } from 'react'
import { GitBranch, X } from 'lucide-react'
import type { ChildContextMode, NodeModel } from '../types'

interface SpawnModalProps {
  parentNode: NodeModel
  onClose: () => void
  onSpawn: (payload: {
    title: string
    goal: string
    context_mode: ChildContextMode
    summary_brief: string
    slice_turn_count: number
    model_override: string | null
    switch_head: boolean
  }) => Promise<void>
}

export function SpawnModal({ parentNode, onClose, onSpawn }: SpawnModalProps) {
  const [title, setTitle] = useState('')
  const [goal, setGoal] = useState('')
  const [contextMode, setContextMode] = useState<ChildContextMode>('fork')
  const [summaryBrief, setSummaryBrief] = useState('')
  const [sliceTurns, setSliceTurns] = useState(6)
  const [modelOverride, setModelOverride] = useState('')
  const [switchHead, setSwitchHead] = useState(true)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!goal.trim()) return
    setSubmitting(true)
    try {
      await onSpawn({
        title: title.trim() || goal.trim().split(/\s+/).slice(0, 5).join(' '),
        goal: goal.trim(),
        context_mode: contextMode,
        summary_brief: summaryBrief.trim(),
        slice_turn_count: sliceTurns,
        model_override: modelOverride.trim() || null,
        switch_head: switchHead,
      })
      onClose()
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-lg rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden"
      >
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-zinc-100">
              Spawn Child under “{parentNode.title}”
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-200 p-1"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4 text-xs">
          <div>
            <label className="block text-zinc-400 mb-1 font-medium">Branch Goal *</label>
            <input
              autoFocus
              required
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Design OAuth2 refresh token rotation and revocation"
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-zinc-400 mb-1 font-medium">Short Title</label>
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Auto-derived from goal"
                className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-zinc-400 mb-1 font-medium">
                Model Override (optional)
              </label>
              <input
                value={modelOverride}
                onChange={(e) => setModelOverride(e.target.value)}
                placeholder="Inherit tree default"
                className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-zinc-400 mb-1.5 font-medium">Context Mode</label>
            <div className="grid grid-cols-3 gap-2">
              {(
                [
                  {
                    id: 'fork',
                    label: 'Fork (CoW)',
                    desc: 'Exact parent prefix · KV cache hit',
                  },
                  {
                    id: 'slice',
                    label: 'Slice',
                    desc: `Last ${sliceTurns} turns + brief`,
                  },
                  {
                    id: 'clean',
                    label: 'Clean',
                    desc: 'Isolated brief only · zero noise',
                  },
                ] as const
              ).map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setContextMode(opt.id)}
                  className={`p-2.5 rounded-lg border text-left transition-colors ${
                    contextMode === opt.id
                      ? 'border-indigo-500 bg-indigo-500/15 text-zinc-100'
                      : 'border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:border-zinc-700'
                  }`}
                >
                  <div className="font-semibold text-xs">{opt.label}</div>
                  <div className="text-[10px] text-zinc-400 mt-0.5">{opt.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {contextMode === 'slice' && (
            <div>
              <label className="block text-zinc-400 mb-1 font-medium">
                Recent Parent Messages to Slice: {sliceTurns}
              </label>
              <input
                type="range"
                min={2}
                max={20}
                value={sliceTurns}
                onChange={(e) => setSliceTurns(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
            </div>
          )}

          <div>
            <label className="block text-zinc-400 mb-1 font-medium">Summary Brief (optional)</label>
            <textarea
              value={summaryBrief}
              onChange={(e) => setSummaryBrief(e.target.value)}
              rows={2}
              placeholder="Key constraints or context to hand to the child branch..."
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 p-2.5 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>

          <label className="flex items-center gap-2 text-zinc-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={switchHead}
              onChange={(e) => setSwitchHead(e.target.checked)}
              className="rounded border-zinc-700 bg-zinc-900 text-indigo-500"
            />
            Switch active HEAD into the new child branch immediately
          </label>
        </div>

        <div className="px-5 py-3 border-t border-zinc-800 bg-zinc-900/40 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-100"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting || !goal.trim()}
            className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium disabled:opacity-50"
          >
            {submitting ? 'Spawning…' : 'Spawn Child'}
          </button>
        </div>
      </form>
    </div>
  )
}
