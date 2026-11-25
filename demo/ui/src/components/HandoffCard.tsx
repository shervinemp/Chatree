import { useState } from 'react'
import { Check, FileCode2, GitMerge, X } from 'lucide-react'
import type { MailboxItemModel } from '../types'

interface HandoffCardProps {
  item: MailboxItemModel
  fromNodeTitle: string
  toNodeTitle: string
  onCommit: (mailboxId: string, editedSummary: string, returnHead: boolean) => Promise<void>
  onDismiss: (mailboxId: string) => Promise<void>
}

export function HandoffCard({
  item,
  fromNodeTitle,
  toNodeTitle,
  onCommit,
  onDismiss,
}: HandoffCardProps) {
  const [summary, setSummary] = useState(item.summary)
  const [returnHead, setReturnHead] = useState(true)
  const [submitting, setSubmitting] = useState(false)

  const handleCommit = async () => {
    setSubmitting(true)
    try {
      await onCommit(item.id, summary, returnHead)
    } finally {
      setSubmitting(false)
    }
  }

  const handleDismiss = async () => {
    setSubmitting(true)
    try {
      await onDismiss(item.id)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="rounded-xl border border-fuchsia-500/40 bg-fuchsia-950/20 p-4 space-y-3 shadow-lg">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-fuchsia-500/20 text-fuchsia-300">
            <GitMerge className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-semibold text-fuchsia-200">
              PR-Style Handoff Review: {fromNodeTitle} → {toNodeTitle}
            </div>
            <div className="text-[11px] text-zinc-400">
              Edit the narrative summary below before committing it into the parent branch context.
            </div>
          </div>
        </div>
        <span className="px-2 py-0.5 rounded-full bg-fuchsia-500/20 text-fuchsia-300 font-mono text-[10px]">
          pending_review
        </span>
      </div>

      <textarea
        value={summary}
        onChange={(e) => setSummary(e.target.value)}
        rows={4}
        className="w-full rounded-lg border border-zinc-700/80 bg-zinc-950/90 p-2.5 text-xs text-zinc-100 font-sans focus:outline-none focus:ring-1 focus:ring-fuchsia-400"
        placeholder="Summary to squash-merge into parent..."
      />

      {item.artifact_refs.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {item.artifact_refs.map((art) => (
            <span
              key={art.id}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-zinc-900 border border-zinc-700/80 text-[11px] font-mono text-indigo-300"
            >
              <FileCode2 className="w-3 h-3" />
              {art.slug}@v{art.version}
            </span>
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
        <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={returnHead}
            onChange={(e) => setReturnHead(e.target.checked)}
            className="rounded border-zinc-700 bg-zinc-900 text-indigo-500"
          />
          <span>Return HEAD to parent after commit</span>
        </label>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            disabled={submitting}
            onClick={handleDismiss}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs transition-colors"
          >
            <X className="w-3.5 h-3.5" />
            Discard
          </button>
          <button
            type="button"
            disabled={submitting || !summary.trim()}
            onClick={handleCommit}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-fuchsia-600 hover:bg-fuchsia-500 text-white text-xs font-medium transition-colors disabled:opacity-50"
          >
            <Check className="w-3.5 h-3.5" />
            {submitting ? 'Committing…' : 'Commit to Parent'}
          </button>
        </div>
      </div>
    </div>
  )
}
