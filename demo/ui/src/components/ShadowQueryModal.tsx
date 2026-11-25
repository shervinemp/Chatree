import { useState } from 'react'
import { ChevronDown, ChevronRight, Eye, Layers, Search, Sparkles, X } from 'lucide-react'
import type { AskTreeResponse, NodeModel, SearchScope } from '../types'
import { MarkdownText } from './MarkdownText'

interface ShadowQueryModalProps {
  callerNode: NodeModel
  candidateNodes: NodeModel[]
  initialTargetId?: string
  initialMode?: 'direct' | 'ask'
  onClose: () => void
  onRunRequest: (
    targetId: string,
    prompt: string | null,
    mutate: boolean
  ) => Promise<Record<string, unknown>>
  onRunAskTree: (
    query: string,
    scope: SearchScope,
    maxTargets: number,
    deep: boolean
  ) => Promise<AskTreeResponse['result']>
  onJumpToNode?: (nodeId: string) => void
}

export function ShadowQueryModal({
  callerNode,
  candidateNodes,
  initialTargetId,
  initialMode = 'direct',
  onClose,
  onRunRequest,
  onRunAskTree,
  onJumpToNode,
}: ShadowQueryModalProps) {
  const [mode, setMode] = useState<'direct' | 'ask'>(initialMode)
  const [targetId, setTargetId] = useState(initialTargetId || candidateNodes[0]?.id || '')
  const [prompt, setPrompt] = useState('')
  const [mutate, setMutate] = useState(false)

  const [askQuery, setAskQuery] = useState('')
  const [scope, setScope] = useState<SearchScope>('all')
  const [maxTargets, setMaxTargets] = useState(3)
  const [deep, setDeep] = useState(true)
  const [showEvidence, setShowEvidence] = useState(false)

  const [running, setRunning] = useState(false)
  const [directResult, setDirectResult] = useState<Record<string, unknown> | null>(null)
  const [askResult, setAskResult] = useState<AskTreeResponse['result'] | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const nodeNames: Record<string, string> = {}
  for (const n of candidateNodes) nodeNames[n.id] = n.title
  nodeNames[callerNode.id] = callerNode.title

  const handleRunDirect = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!targetId) return
    setRunning(true)
    setErrorMsg(null)
    try {
      const res = await onRunRequest(targetId, prompt.trim() ? prompt.trim() : null, mutate)
      setDirectResult(res)
    } catch (err) {
      setErrorMsg(String(err))
    } finally {
      setRunning(false)
    }
  }

  const handleRunAsk = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!askQuery.trim()) return
    setRunning(true)
    setErrorMsg(null)
    try {
      const res = await onRunAskTree(askQuery.trim(), scope, maxTargets, deep)
      setAskResult(res)
    } catch (err) {
      setErrorMsg(String(err))
    } finally {
      setRunning(false)
    }
  }

  const evidenceList = Array.isArray(askResult?.found_answers)
    ? askResult.found_answers
    : Array.isArray(askResult?.answers)
      ? askResult.answers
      : []

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full max-w-2xl max-h-[88vh] flex flex-col rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            {mode === 'ask' ? (
              <Sparkles className="w-4 h-4 text-indigo-400" />
            ) : (
              <Eye className="w-4 h-4 text-indigo-400" />
            )}
            <h3 className="text-sm font-semibold text-zinc-100">
              {mode === 'ask'
                ? `Ask Tree from “${callerNode.title}”`
                : `IPC request() from “${callerNode.title}”`}
            </h3>
          </div>
          <button type="button" onClick={onClose} className="text-zinc-500 hover:text-zinc-200 p-1">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Mode Switcher */}
        <div className="px-5 pt-3 pb-0 shrink-0">
          <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-zinc-900 border border-zinc-800 text-xs">
            <button
              type="button"
              onClick={() => {
                setMode('ask')
                setErrorMsg(null)
              }}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-md font-medium transition-colors ${
                mode === 'ask'
                  ? 'bg-indigo-600 text-white'
                  : 'text-zinc-400 hover:text-zinc-100'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              Ask Tree (Multi-Hop / Deep)
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('direct')
                setErrorMsg(null)
              }}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-md font-medium transition-colors ${
                mode === 'direct'
                  ? 'bg-indigo-600 text-white'
                  : 'text-zinc-400 hover:text-zinc-100'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              Direct Branch request()
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-5 text-xs">
          {errorMsg && (
            <div className="mb-3 rounded-lg border border-rose-500/40 bg-rose-950/30 px-3 py-2 text-rose-200">
              {errorMsg}
            </div>
          )}

          {mode === 'ask' ? (
            <form onSubmit={handleRunAsk} className="space-y-4">
              <div>
                <label className="block text-zinc-400 mb-1 font-medium">Question Across Tree</label>
                <textarea
                  autoFocus
                  value={askQuery}
                  onChange={(e) => setAskQuery(e.target.value)}
                  rows={3}
                  placeholder="e.g. Which signing algorithm and TTL did we settle on across branches?"
                  className="w-full rounded-lg border border-zinc-800 bg-zinc-900 p-2.5 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-zinc-400 mb-1 font-medium">Search Scope</label>
                  <select
                    value={scope}
                    onChange={(e) => setScope(e.target.value as SearchScope)}
                    className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  >
                    <option value="all">all (entire active tree)</option>
                    <option value="ancestors">ancestors (root-to-HEAD chain)</option>
                    <option value="subtree">subtree (descendants)</option>
                    <option value="siblings">siblings (peer branches)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-zinc-400 mb-1 font-medium">
                    Max Parallel Shadow Targets
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    value={maxTargets}
                    onChange={(e) => setMaxTargets(Number(e.target.value))}
                    className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-100"
                  />
                </div>
              </div>

              <label className="flex items-start gap-2.5 rounded-lg border border-indigo-500/25 bg-indigo-500/5 p-3 text-zinc-300 cursor-pointer select-none leading-relaxed">
                <input
                  type="checkbox"
                  checked={deep}
                  onChange={(e) => setDeep(e.target.checked)}
                  className="mt-0.5 rounded border-zinc-700 bg-zinc-900 text-indigo-500 shrink-0"
                />
                <span>
                  <strong className="text-indigo-200">Deep Multi-Hop Synthesis (`deep=True`)</strong>
                  <span className="block text-[11px] text-zinc-400 mt-0.5">
                    Runs the unified <code className="font-mono text-indigo-300">TreeQueryEngine</code>:
                    Hop-1 parallel read-only shadow Q&amp;A + Tier-2.5 dated atomic facts + optional
                    Hop-2 summary-map drill-down + chronological evidence synthesis.
                  </span>
                </span>
              </label>

              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={running || !askQuery.trim()}
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium disabled:opacity-50"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  {running ? 'Synthesizing…' : deep ? 'Run Deep Tree Ask' : 'Run Compound Ask'}
                </button>
              </div>

              {askResult && (
                <div className="rounded-xl border border-indigo-500/35 bg-indigo-950/20 p-3.5 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-indigo-300">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="px-1.5 py-0.5 rounded bg-indigo-500/20 border border-indigo-400/30">
                        {askResult.deep ? 'TreeQueryEngine (deep)' : 'ask_tree (1-hop)'}
                      </span>
                      {typeof askResult.facts_used === 'number' && (
                        <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                          {askResult.facts_used} Tier-2.5 fact(s)
                        </span>
                      )}
                      {typeof askResult.synth_tokens_in === 'number' && askResult.synth_tokens_in > 0 && (
                        <span className="text-zinc-400">{askResult.synth_tokens_in} synth tok</span>
                      )}
                    </div>
                    <span className="text-zinc-400">
                      {(askResult.targets_queried || []).length} branch(es) queried
                    </span>
                  </div>

                  {askResult.answer ? (
                    <MarkdownText content={String(askResult.answer)} />
                  ) : evidenceList.length > 0 ? (
                    <div className="space-y-2">
                      {evidenceList.map((a, idx) => (
                        <div
                          key={String(a.target_id || idx)}
                          className="rounded-lg bg-zinc-950/80 border border-zinc-800 p-2.5 space-y-1"
                        >
                          <div className="font-mono text-[10px] text-indigo-300">
                            {String(a.target_title || a.target_id || 'branch')}
                          </div>
                          <MarkdownText content={String(a.answer || a.summary || '')} />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-zinc-400 italic">No matching evidence found in active branches.</div>
                  )}

                  {(askResult.targets_queried || []).length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 pt-1 border-t border-zinc-800/70">
                      <span className="text-[10px] text-zinc-500 font-mono">Targets:</span>
                      {(askResult.targets_queried || []).map((tid) => (
                        <button
                          key={tid}
                          type="button"
                          onClick={() => {
                            onJumpToNode?.(tid)
                            onClose()
                          }}
                          className="px-2 py-0.5 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-[10px] font-mono text-indigo-300"
                        >
                          {nodeNames[tid] || tid}
                        </button>
                      ))}
                    </div>
                  )}

                  {askResult.answer && evidenceList.length > 0 && (
                    <div className="pt-1 border-t border-zinc-800/70">
                      <button
                        type="button"
                        onClick={() => setShowEvidence(!showEvidence)}
                        className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200"
                      >
                        {showEvidence ? (
                          <ChevronDown className="w-3.5 h-3.5" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5" />
                        )}
                        <Layers className="w-3 h-3 text-indigo-400" />
                        Verified Branch Shadow Evidence ({evidenceList.length})
                      </button>
                      {showEvidence && (
                        <div className="mt-2 space-y-2 max-h-48 overflow-y-auto">
                          {evidenceList.map((a, idx) => (
                            <div
                              key={String(a.target_id || idx)}
                              className="rounded-lg bg-zinc-950/90 border border-zinc-800 p-2 space-y-1"
                            >
                              <div className="font-mono text-[10px] text-indigo-300">
                                {String(a.target_title || a.target_id || 'branch')}
                              </div>
                              <div className="text-[11px] text-zinc-300 whitespace-pre-wrap">
                                {String(a.answer || a.summary || '')}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </form>
          ) : (
            <form onSubmit={handleRunDirect} className="space-y-4">
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
                  Persist Q&amp;A into target history (
                  <code className="font-mono text-zinc-300">mutate=True</code> — leave unchecked for
                  a clean read-only shadow turn)
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

              {directResult && (
                <div className="rounded-xl border border-indigo-500/35 bg-indigo-950/20 p-3.5 space-y-2 max-h-64 overflow-y-auto">
                  <div className="flex items-center justify-between font-mono text-[10px] text-indigo-300">
                    <span>
                      mode={String(directResult.mode || directResult.error || 'ok')}
                      {directResult.cache_hit ? ' · 0-token cache hit' : ''}
                    </span>
                    <span>v{String(directResult.summary_version ?? 0)}</span>
                  </div>
                  <MarkdownText
                    content={String(
                      directResult.answer ||
                        directResult.summary ||
                        directResult.detail ||
                        JSON.stringify(directResult)
                    )}
                  />
                </div>
              )}
            </form>
          )}
        </div>
      </div>
    </div>
  )
}
