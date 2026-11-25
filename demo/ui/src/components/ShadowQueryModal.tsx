import { useState } from 'react'
import { Eye, Search, X } from 'lucide-react'
import type { NodeModel } from '../types'
import { MarkdownText } from './MarkdownText'

interface ShadowQueryModalProps {
  callerNode: NodeModel
  candidateNodes: NodeModel[]
  initialTargetId?: string
  onClose: () => void
  onRunRequest: (
    targetId: string,
    prompt: string | null,
    mutate: boolean
  ) => Promise<Record<string, unknown>>
}

export function ShadowQueryModal({
  callerNode,
  candidateNodes,
  initialTargetId,
  onClose,
  onRunRequest,
}: ShadowQueryModalProps) {
  const [targetId, setTargetId] = useState(
    initialTargetId || candidateNodes[0]?.id || ''
  )
  const [prompt, setPrompt] = useState('')
  const [mutate, setMutate] = useState(false)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!targetId) return
    setRunning(true)
    try {
      const res = await onRunRequest(targetId, prompt.trim() ? prompt.trim() : null, mutate)
      setResult(res)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full max-w-xl rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-zinc-100">
              IPC request() from “{callerNode.title}”
            </h3>
          </div>
          <button type="button" onClick={onClose} className="text-zinc-500 hover:text-zinc-200 p-1">
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleRun} className="p-5 space-y-4 text-xs">
          <div>
            <label className="block text-zinc-400 mb-1 font-medium">Target Node</label>
            <select
              value={targetId}
              onChange={(e) => setTargetId(e.target.value)}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              {candidateNodes.map((n) => (
                <option key={n.id} value={n.id}>
                  {n.title} ({n.id} · {n.status} · v{n.summary_version}
                  {n.is_dirty ? '*' : ''})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-zinc-400 mb-1 font-medium">
              Targeted Prompt{' '}
              <span className="text-zinc-500 font-normal">
                (leave blank for zero-cost cached summary)
              </span>
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder="e.g. Which exact database indexes or token TTLs did this branch settle on?"
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900 p-2.5 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>

          <label className="flex items-start gap-2 text-zinc-400 cursor-pointer select-none leading-relaxed">
            <input
              type="checkbox"
              checked={mutate}
              onChange={(e) => setMutate(e.target.checked)}
              className="mt-0.5 rounded border-zinc-700 bg-zinc-900 text-indigo-500 shrink-0"
            />
            <span>
              Persist Q&amp;A into target history (<code className="font-mono text-zinc-300">mutate=True</code>{' '}
              — leave unchecked for a clean read-only shadow turn)
            </span>
          </label>

          <div className="flex justify-end gap-2">
            <button
              type="submit"
              disabled={running || !targetId}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium disabled:opacity-50"
            >
              <Search className="w-3.5 h-3.5" />
              {running
                ? 'Querying…'
                : prompt.trim()
                  ? 'Run Shadow Turn'
                  : 'Fetch Cached Summary'}
            </button>
          </div>

          {result && (
            <div className="rounded-xl border border-indigo-500/35 bg-indigo-950/20 p-3.5 space-y-2 max-h-64 overflow-y-auto">
              <div className="flex items-center justify-between font-mono text-[10px] text-indigo-300">
                <span>
                  mode={String(result.mode || result.error || 'ok')}
                  {result.cache_hit ? ' · 0-token cache hit' : ''}
                </span>
                <span>v{String(result.summary_version ?? 0)}</span>
              </div>
              <MarkdownText
                content={String(
                  result.answer || result.summary || result.detail || JSON.stringify(result)
                )}
              />
            </div>
          )}
        </form>
      </div>
    </div>
  )
}
