import { useState } from 'react'
import { GitBranch, X } from 'lucide-react'

interface NewTreeModalProps {
  onClose: () => void
  onCreate: (title: string, goal: string) => Promise<void>
}

export function NewTreeModal({ onClose, onCreate }: NewTreeModalProps) {
  const [title, setTitle] = useState('')
  const [goal, setGoal] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim()) return
    setSubmitting(true)
    try {
      await onCreate(title.trim(), goal.trim())
      onClose()
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-md rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden"
      >
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-zinc-100">New Dialogue Tree</h3>
          </div>
          <button type="button" onClick={onClose} className="text-zinc-500 hover:text-zinc-200 p-1">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-5 space-y-3 text-xs">
          <div>
            <label className="block text-zinc-400 mb-1 font-medium">Tree Title *</label>
            <input
              autoFocus
              required
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Distributed Auth Architecture"
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
          <div>
            <label className="block text-zinc-400 mb-1 font-medium">Root Goal (optional)</label>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={2}
              placeholder="High-level objective for the root conversation..."
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 p-2.5 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
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
            disabled={submitting || !title.trim()}
            className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium disabled:opacity-50"
          >
            {submitting ? 'Creating…' : 'Create Tree'}
          </button>
        </div>
      </form>
    </div>
  )
}
