import { useState } from 'react'
import { Minimize2, X } from 'lucide-react'
import type { NodeModel } from '../types'

interface CompactModalProps {
  sourceNode: NodeModel
  estimatedTokens: number
  onClose: () => void
  onCompact: (payload: {
    keep_recent_turns: number
    title?: string
    reason?: string
    switch_head: boolean
  }) => Promise<void>
}

export function CompactModal({
  sourceNode,
  estimatedTokens,
  onClose,
  onCompact,
}: CompactModalProps) {
  const [keepTurns, setKeepTurns] = useState(4)
  const [title, setTitle] = useState('')
  const [reason, setReason] = useState('')
  const [switchHead, setSwitchHead] = useState(true)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      await onCompact({
        keep_recent_turns: keepTurns,
        title: title.trim() || undefined,
        reason: reason.trim() || undefined,
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
            <Minimize2 className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-zinc-100">
              Non-Destructive Compact: “{sourceNode.title}”
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
          <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/5 p-3 text-emerald-200/90 leading-relaxed">
            Commits a structured 5-section anchor handoff on{' '}
            <code className="font-mono text-emerald-300">{sourceNode.id}</code> (~
            {estimatedTokens.toLocaleString()} tok) and spawns a linked continuation child. All
            verbatim turns and KV prefix on the source branch stay 100% intact and shadow-queryable.
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-zinc-400 font-medium">
                Keep Recent Verbatim Turns: <span className="text-zinc-100 font-mono">{keepTurns}</span>
              </label>
              <span className="font-mono text-[10px] text-zinc-500">
                {keepTurns === 0 ? 'clean mode (handoff only)' : `slice mode (last ${keepTurns} turns)`}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={12}
              value={keepTurns}
              onChange={(e) => setKeepTurns(Number(e.target.value))}
              className="w-full accent-emerald-500"
            />
          </div>

          <div>
            <label className="block text-zinc-400 mb-1 font-medium">
              Continuation Title (optional)
            </label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder={`${sourceNode.title} (cont. 2)`}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </div>

          <div>
            <label className="block text-zinc-400 mb-1 font-medium">Reason (optional)</label>
            <input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. Keep active working context bounded before phase 2"
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </div>

          <label className="flex items-center gap-2 text-zinc-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={switchHead}
              onChange={(e) => setSwitchHead(e.target.checked)}
              className="rounded border-zinc-700 bg-zinc-900 text-emerald-500"
            />
            Switch active HEAD into the new continuation branch immediately
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
            disabled={submitting}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium disabled:opacity-50"
          >
            <Minimize2 className="w-3.5 h-3.5" />
            {submitting ? 'Compacting…' : 'Compact Branch'}
          </button>
        </div>
      </form>
    </div>
  )
}
