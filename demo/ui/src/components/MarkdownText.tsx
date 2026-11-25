import React from 'react'

function renderInline(text: string): React.ReactNode[] {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code
          key={i}
          className="rounded bg-zinc-800/90 px-1.5 py-0.5 font-mono text-[0.82em] text-indigo-300 border border-zinc-700/60"
        >
          {part.slice(1, -1)}
        </code>
      )
    }
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return (
        <strong key={i} className="font-semibold text-zinc-100">
          {part.slice(2, -2)}
        </strong>
      )
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return (
        <em key={i} className="italic text-zinc-200">
          {part.slice(1, -1)}
        </em>
      )
    }
    return <React.Fragment key={i}>{part}</React.Fragment>
  })
}

export function MarkdownText({ content }: { content: string }) {
  const blocks = content.split(/(```[\s\S]*?```)/g)

  return (
    <div className="space-y-2 text-sm leading-relaxed text-zinc-200">
      {blocks.map((block, bIdx) => {
        if (block.startsWith('```') && block.endsWith('```')) {
          const inner = block.slice(3, -3)
          const firstNl = inner.indexOf('\n')
          const lang = firstNl > -1 ? inner.slice(0, firstNl).trim() : ''
          const code = firstNl > -1 ? inner.slice(firstNl + 1) : inner
          return (
            <div
              key={bIdx}
              className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950/90"
            >
              {lang && (
                <div className="border-b border-zinc-800/80 px-3 py-1 font-mono text-[11px] text-zinc-400">
                  {lang}
                </div>
              )}
              <pre className="p-3 font-mono text-xs text-zinc-200 whitespace-pre">
                <code>{code}</code>
              </pre>
            </div>
          )
        }

        const lines = block.split('\n')
        return (
          <div key={bIdx} className="space-y-1.5">
            {lines.map((line, lIdx) => {
              const trimmed = line.trim()
              if (!trimmed) return <div key={lIdx} className="h-1" />
              if (trimmed.startsWith('> ')) {
                return (
                  <blockquote
                    key={lIdx}
                    className="border-l-2 border-amber-500/70 bg-amber-500/5 pl-3 py-1 text-zinc-300 rounded-r"
                  >
                    {renderInline(trimmed.slice(2))}
                  </blockquote>
                )
              }
              if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                return (
                  <div key={lIdx} className="flex gap-2 pl-1">
                    <span className="text-indigo-400 select-none">•</span>
                    <span className="flex-1">{renderInline(trimmed.slice(2))}</span>
                  </div>
                )
              }
              if (trimmed.startsWith('### ')) {
                return (
                  <h4 key={lIdx} className="font-semibold text-zinc-100 pt-1">
                    {renderInline(trimmed.slice(4))}
                  </h4>
                )
              }
              if (trimmed.startsWith('## ')) {
                return (
                  <h3 key={lIdx} className="text-base font-semibold text-zinc-100 pt-1">
                    {renderInline(trimmed.slice(3))}
                  </h3>
                )
              }
              return (
                <p key={lIdx} className="whitespace-pre-wrap break-words">
                  {renderInline(line)}
                </p>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}
