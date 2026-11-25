import { useEffect } from 'react'

interface KeyboardNavOptions {
  onBack: () => void
  onForward: () => void
  onOpenPalette: () => void
}

export function useKeyboardNav({ onBack, onForward, onOpenPalette }: KeyboardNavOptions) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        onOpenPalette()
        return
      }
      if (e.altKey && e.key === 'ArrowLeft') {
        e.preventDefault()
        onBack()
        return
      }
      if (e.altKey && e.key === 'ArrowRight') {
        e.preventDefault()
        onForward()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onBack, onForward, onOpenPalette])
}
