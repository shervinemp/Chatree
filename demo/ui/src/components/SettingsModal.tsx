import { useState } from 'react'
import { CheckCircle2, KeyRound, Settings, X } from 'lucide-react'
import type { LLMBackendKind, RuntimeSettingsModel, TreeModel } from '../types'

interface SettingsModalProps {
  settings: RuntimeSettingsModel
  activeTree: TreeModel | null
  onClose: () => void
  onSave: (payload: {
    llm_provider: LLMBackendKind
    default_model: string
    shadow_model: string
    summary_model: string
    api_base: string
    max_hop_depth: number
    token_budget: number
    auto_commit_handoffs: boolean
    api_keys?: Record<string, string>
    applyToActiveTree: boolean
  }) => Promise<void>
}

const MODEL_PRESETS = [
  { label: 'Demo Mode (Zero-Key Mock)', value: 'mock/chatree-demo' },
  { label: 'DeepSeek · Chat', value: 'deepseek/deepseek-chat' },
  { label: 'Anthropic · Claude Sonnet 4.5', value: 'anthropic/claude-sonnet-4-5' },
  { label: 'OpenAI · GPT-4o', value: 'openai/gpt-4o' },
  { label: 'Gemini · 2.5 Flash', value: 'gemini/gemini-2.5-flash' },
  { label: 'OpenRouter · Claude Sonnet 4', value: 'openrouter/anthropic/claude-sonnet-4' },
  { label: 'Groq · Llama 3.3 70B', value: 'groq/llama-3.3-70b-versatile' },
  { label: 'Ollama · Llama 3.2 (Local)', value: 'ollama/llama3.2' },
]

const PROVIDER_DRIVERS: Array<{ id: LLMBackendKind; label: string; desc: string }> = [
  { id: 'auto', label: 'auto', desc: 'LiteLLM when installed, else zero-dep openai_compat' },
  { id: 'openai_compat', label: 'openai_compat', desc: 'Zero-LiteLLM native HTTP/SSE (OpenAI, DeepSeek, Ollama, vLLM)' },
  { id: 'litellm', label: 'litellm', desc: '100+ providers via LiteLLM router' },
  { id: 'mock', label: 'mock', desc: 'Offline deterministic demo provider' },
]

export function SettingsModal({
  settings,
  activeTree,
  onClose,
  onSave,
}: SettingsModalProps) {
  const [llmProvider, setLlmProvider] = useState<LLMBackendKind>(settings.llm_provider || 'auto')
  const [defaultModel, setDefaultModel] = useState(
    activeTree?.default_model || settings.default_model
  )
  const [shadowModel, setShadowModel] = useState(
    activeTree?.shadow_model || settings.shadow_model
  )
  const [summaryModel, setSummaryModel] = useState(
    activeTree?.summary_model || settings.summary_model
  )
  const [apiBase, setApiBase] = useState(settings.api_base || '')
  const [maxHopDepth, setMaxHopDepth] = useState(settings.max_hop_depth)
  const [tokenBudget, setTokenBudget] = useState(settings.token_budget)
  const [autoCommit, setAutoCommit] = useState(
    activeTree ? activeTree.auto_commit_handoffs : settings.auto_commit_handoffs
  )
  const [applyToActiveTree, setApplyToActiveTree] = useState(true)
  const [keys, setKeys] = useState<Record<string, string>>({
    DEEPSEEK_API_KEY: '',
    OPENAI_API_KEY: '',
    ANTHROPIC_API_KEY: '',
    GEMINI_API_KEY: '',
    OPENROUTER_API_KEY: '',
    GROQ_API_KEY: '',
  })
  const [saving, setSaving] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    try {
      const nonEmptyKeys: Record<string, string> = {}
      for (const [k, v] of Object.entries(keys)) {
        if (v.trim()) nonEmptyKeys[k] = v.trim()
      }
      await onSave({
        llm_provider: llmProvider,
        default_model: defaultModel.trim() || 'mock/chatree-demo',
        shadow_model: shadowModel.trim() || 'inherit',
        summary_model: summaryModel.trim() || 'inherit',
        api_base: apiBase.trim(),
        max_hop_depth: maxHopDepth,
        token_budget: tokenBudget,
        auto_commit_handoffs: autoCommit,
        api_keys: Object.keys(nonEmptyKeys).length > 0 ? nonEmptyKeys : undefined,
        applyToActiveTree,
      })
      onClose()
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-xl max-h-[88vh] flex flex-col rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden"
      >
        <div className="px-5 py-3.5 border-b border-zinc-800 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-zinc-100">
              Model Routing, Providers &amp; Guardrails
            </h3>
          </div>
          <button type="button" onClick={onClose} className="text-zinc-500 hover:text-zinc-200 p-1">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5 text-xs">
          {/* Provider Driver Selector */}
          <div>
            <label className="block text-zinc-400 mb-1.5 font-medium">
              LLM Backend Driver (<code className="font-mono text-indigo-300">llm_provider</code>)
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {PROVIDER_DRIVERS.map((drv) => (
                <button
                  key={drv.id}
                  type="button"
                  onClick={() => setLlmProvider(drv.id)}
                  className={`p-2 rounded-lg border text-left transition-colors ${
                    llmProvider === drv.id
                      ? 'border-indigo-500 bg-indigo-500/15 text-zinc-100'
                      : 'border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:border-zinc-700'
                  }`}
                >
                  <div className="font-mono font-semibold text-[11px]">{drv.label}</div>
                  <div className="text-[10px] text-zinc-400 mt-0.5 leading-snug">{drv.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Quick Presets */}
          <div>
            <label className="block text-zinc-400 mb-1.5 font-medium">Quick Model Presets</label>
            <div className="flex flex-wrap gap-1.5">
              {MODEL_PRESETS.map((p) => (
                <button
                  key={p.value}
                  type="button"
                  onClick={() => setDefaultModel(p.value)}
                  className={`px-2 py-1 rounded-md border text-[11px] transition-colors ${
                    defaultModel === p.value
                      ? 'border-indigo-500 bg-indigo-500/20 text-indigo-200'
                      : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Model Inputs */}
          <div className="grid grid-cols-1 gap-3">
            <div>
              <label className="block text-zinc-400 mb-1 font-medium">
                Default Model (LiteLLM ID)
              </label>
              <input
                value={defaultModel}
                onChange={(e) => setDefaultModel(e.target.value)}
                className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-100"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-zinc-400 mb-1 font-medium">
                  Shadow Query Model
                </label>
                <input
                  value={shadowModel}
                  onChange={(e) => setShadowModel(e.target.value)}
                  placeholder="inherit (maximizes KV cache)"
                  className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-100"
                />
              </div>
              <div>
                <label className="block text-zinc-400 mb-1 font-medium">
                  Summary Model
                </label>
                <input
                  value={summaryModel}
                  onChange={(e) => setSummaryModel(e.target.value)}
                  placeholder="inherit"
                  className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-100"
                />
              </div>
            </div>

            <div>
              <label className="block text-zinc-400 mb-1 font-medium">
                Custom OpenAI-Compatible API Base (optional)
              </label>
              <input
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                placeholder="e.g. http://localhost:11434/v1"
                className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-100"
              />
            </div>
          </div>

          {/* Guardrails & Handoffs */}
          <div className="grid grid-cols-2 gap-3 pt-1 border-t border-zinc-800/80">
            <div>
              <label className="block text-zinc-400 mb-1 font-medium">
                Max IPC Hop Depth (Cycle Guard)
              </label>
              <input
                type="number"
                min={1}
                max={8}
                value={maxHopDepth}
                onChange={(e) => setMaxHopDepth(Number(e.target.value))}
                className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-100"
              />
            </div>
            <div>
              <label className="block text-zinc-400 mb-1 font-medium">
                Shared Trace Token Budget
              </label>
              <input
                type="number"
                min={1000}
                max={200000}
                step={1000}
                value={tokenBudget}
                onChange={(e) => setTokenBudget(Number(e.target.value))}
                className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-100"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2 text-zinc-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoCommit}
                onChange={(e) => setAutoCommit(e.target.checked)}
                className="rounded border-zinc-700 bg-zinc-900 text-indigo-500"
              />
              Auto-commit child <code className="font-mono">send(finish=True)</code> handoffs
              (skip PR review card)
            </label>
            {activeTree && (
              <label className="flex items-center gap-2 text-zinc-300 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={applyToActiveTree}
                  onChange={(e) => setApplyToActiveTree(e.target.checked)}
                  className="rounded border-zinc-700 bg-zinc-900 text-indigo-500"
                />
                Apply model &amp; handoff defaults to current tree (“{activeTree.title}”)
              </label>
            )}
          </div>

          {/* Provider Keys */}
          <div className="pt-2 border-t border-zinc-800/80 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 font-medium text-zinc-300">
                <KeyRound className="w-3.5 h-3.5 text-indigo-400" />
                Provider API Keys (kept in runtime memory; never echoed back)
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2.5">
              {(
                [
                  ['deepseek', 'DEEPSEEK_API_KEY'],
                  ['openai', 'OPENAI_API_KEY'],
                  ['anthropic', 'ANTHROPIC_API_KEY'],
                  ['gemini', 'GEMINI_API_KEY'],
                  ['openrouter', 'OPENROUTER_API_KEY'],
                  ['groq', 'GROQ_API_KEY'],
                ] as const
              ).map(([prov, envKey]) => {
                const isSet = !!settings.configured_providers[prov]
                return (
                  <div key={envKey}>
                    <div className="flex items-center justify-between mb-1">
                      <label className="font-mono text-[10px] text-zinc-400">{envKey}</label>
                      {isSet && (
                        <span className="flex items-center gap-0.5 text-[10px] text-emerald-400">
                          <CheckCircle2 className="w-2.5 h-2.5" /> configured
                        </span>
                      )}
                    </div>
                    <input
                      type="password"
                      value={keys[envKey] || ''}
                      onChange={(e) => setKeys((prev) => ({ ...prev, [envKey]: e.target.value }))}
                      placeholder={isSet ? '•••••••• (unchanged)' : 'Paste key…'}
                      className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 font-mono text-[11px] text-zinc-100"
                    />
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        <div className="px-5 py-3 border-t border-zinc-800 bg-zinc-900/40 flex justify-end gap-2 shrink-0">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-100"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={saving}
            className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save Settings'}
          </button>
        </div>
      </form>
    </div>
  )
}
