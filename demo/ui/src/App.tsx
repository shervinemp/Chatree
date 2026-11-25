import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from './api/client'
import { BreadcrumbBar } from './components/BreadcrumbBar'
import { ChatPane } from './components/ChatPane'
import { CommandPalette } from './components/CommandPalette'
import { InspectorDrawer } from './components/InspectorDrawer'
import { NewTreeModal } from './components/NewTreeModal'
import { SettingsModal } from './components/SettingsModal'
import { ShadowQueryModal } from './components/ShadowQueryModal'
import { SpawnModal } from './components/SpawnModal'
import { TreeOutline } from './components/TreeOutline'
import { useKeyboardNav } from './hooks/useKeyboardNav'
import type {
  ChildContextMode,
  IpcEventModel,
  NodeDetailResponse,
  NodeModel,
  RuntimeSettingsModel,
  TreeModel,
  TreeSnapshotResponse,
} from './types'

export function App() {
  const [trees, setTrees] = useState<TreeModel[]>([])
  const [snapshot, setSnapshot] = useState<TreeSnapshotResponse | null>(null)
  const [nodeDetail, setNodeDetail] = useState<NodeDetailResponse | null>(null)
  const [settings, setSettings] = useState<RuntimeSettingsModel | null>(null)

  const [inspectorOpen, setInspectorOpen] = useState(
    () => (typeof window !== 'undefined' ? window.innerWidth >= 1100 : true)
  )
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [newTreeOpen, setNewTreeOpen] = useState(false)
  const [spawnParent, setSpawnParent] = useState<NodeModel | null>(null)
  const [shadowTarget, setShadowTarget] = useState<NodeModel | null>(null)
  const [shadowModalOpen, setShadowModalOpen] = useState(false)

  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [streamingIpc, setStreamingIpc] = useState<IpcEventModel[]>([])
  const [ephemeralNotices, setEphemeralNotices] = useState<string[]>([])
  const [finishing, setFinishing] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const showToast = useCallback((msg: string) => {
    setToast(msg)
    setTimeout(() => {
      setToast((curr) => (curr === msg ? null : curr))
    }, 4000)
  }, [])

  const loadTreeAndHead = useCallback(async (treeId: string, preferredNodeId?: string) => {
    const snap = await api.getTreeSnapshot(treeId)
    setSnapshot(snap)
    const targetNodeId = preferredNodeId || snap.tree.head_node_id || snap.tree.root_node_id
    if (targetNodeId) {
      const detail = await api.getNodeDetail(targetNodeId)
      setNodeDetail(detail)
    } else {
      setNodeDetail(null)
    }
  }, [])

  const bootstrap = useCallback(async () => {
    const [treeList, rtSettings] = await Promise.all([api.listTrees(), api.getSettings()])
    setSettings(rtSettings)
    if (treeList.length === 0) {
      const created = await api.createTree({
        title: 'Architecture Exploration',
        goal: 'Explore system design decisions using isolated branches and read-only shadow turns.',
      })
      setTrees([created.tree])
      setSnapshot(created)
      if (created.tree.head_node_id) {
        const detail = await api.getNodeDetail(created.tree.head_node_id)
        setNodeDetail(detail)
      }
      return
    }
    setTrees(treeList)
    await loadTreeAndHead(treeList[0].id)
  }, [loadTreeAndHead])

  useEffect(() => {
    bootstrap().catch((err) => showToast(`Failed to connect to backend: ${String(err)}`))
  }, [bootstrap, showToast])

  const activeTree = snapshot?.tree ?? null
  const activeNode = nodeDetail?.node ?? null

  const handleJump = useCallback(
    async (target: string, note = '') => {
      if (!activeTree) return
      try {
        const res = await api.reflogJump(activeTree.id, target, note)
        setEphemeralNotices([])
        await loadTreeAndHead(activeTree.id, res.head_node.id)
      } catch (err) {
        showToast(String(err))
      }
    },
    [activeTree, loadTreeAndHead, showToast]
  )

  useKeyboardNav({
    onBack: () => handleJump('back'),
    onForward: () => handleJump('forward'),
    onOpenPalette: () => setPaletteOpen((v) => !v),
  })

  const handleSelectTree = async (treeId: string) => {
    setEphemeralNotices([])
    await loadTreeAndHead(treeId)
  }

  const handleCreateTree = async (title: string, goal: string) => {
    const created = await api.createTree({ title, goal })
    setTrees((prev) => [created.tree, ...prev])
    setSnapshot(created)
    setEphemeralNotices([])
    if (created.tree.head_node_id) {
      setNodeDetail(await api.getNodeDetail(created.tree.head_node_id))
    }
    showToast(`Created tree “${created.tree.title}”`)
  }

  const handleDeleteTree = async (treeId: string) => {
    await api.deleteTree(treeId)
    const remaining = await api.listTrees()
    setTrees(remaining)
    if (remaining.length > 0) {
      await loadTreeAndHead(remaining[0].id)
    } else {
      await bootstrap()
    }
  }

  const handleSendMessage = async (content: string) => {
    if (!activeTree || !activeNode || isStreaming) return
    setIsStreaming(true)
    setStreamingText('')
    setStreamingIpc([])
    let nextHeadId: string | null = null

    try {
      await api.streamTurn(activeNode.id, content, {
        onTurnStart: (data) => {
          setEphemeralNotices(data.ephemeral_notices || [])
          setNodeDetail((prev) =>
            prev
              ? { ...prev, messages: [...prev.messages, data.user_message] }
              : prev
          )
          if (data.reopened) {
            showToast('Branch reopened — dependent nodes flagged STALE.')
          }
        },
        onToken: (delta) => {
          setStreamingText((prev) => prev + delta)
        },
        onIpcActivity: (ev) => {
          setStreamingIpc((prev) => [...prev, ev])
        },
        onHeadChanged: (data) => {
          nextHeadId = data.to_node_id
          showToast(`HEAD moved to ${data.to_node_id} (press Alt+← to go back)`)
        },
        onHandoffPending: () => {
          showToast('PR-style handoff ready for review!')
        },
        onStalenessUpdated: (data) => {
          showToast(
            `Staleness propagated to ${data.stale_node_ids.length} dependent node(s).`
          )
        },
        onError: (err) => {
          showToast(`${err.code}: ${err.message}`)
        },
      })
    } finally {
      setIsStreaming(false)
      setStreamingText('')
      setStreamingIpc([])
      if (nextHeadId && nextHeadId !== activeNode.id) {
        setEphemeralNotices([])
      }
      await loadTreeAndHead(activeTree.id, nextHeadId || activeNode.id)
    }
  }

  const handleSpawnChild = async (payload: {
    title: string
    goal: string
    context_mode: ChildContextMode
    summary_brief: string
    slice_turn_count: number
    model_override: string | null
    switch_head: boolean
  }) => {
    if (!activeTree || !spawnParent) return
    const res = await api.spawnChild(spawnParent.id, payload)
    const nextId =
      payload.switch_head && res.child ? res.child.id : activeNode?.id || spawnParent.id
    await loadTreeAndHead(activeTree.id, nextId)
    if (res.child) {
      showToast(`Spawned ${res.child.context_mode} branch “${res.child.title}”`)
    }
  }

  const handleFinishBranch = async () => {
    if (!activeTree || !activeNode || !activeNode.parent_id) return
    setFinishing(true)
    try {
      const res = await api.finishNode(activeNode.id, { return_to_parent: true })
      const nextNodeId = res.head_changed?.to_node_id || activeNode.id
      await loadTreeAndHead(activeTree.id, nextNodeId)
      if (res.handoff_pending) {
        showToast('Summary generated — review & edit the PR handoff card to commit to parent.')
      } else {
        showToast('Branch completed and merged into parent.')
      }
    } catch (err) {
      showToast(String(err))
    } finally {
      setFinishing(false)
    }
  }

  const handleCommitHandoff = async (
    mailboxId: string,
    summary: string,
    returnHead: boolean
  ) => {
    if (!activeTree) return
    const res = await api.commitHandoff(mailboxId, {
      summary,
      return_head_to_target: returnHead,
    })
    await loadTreeAndHead(
      activeTree.id,
      returnHead ? res.target_node.id : activeNode?.id
    )
    showToast(`Committed handoff into “${res.target_node.title}”`)
  }

  const handleDismissHandoff = async (mailboxId: string) => {
    if (!activeTree) return
    await api.dismissHandoff(mailboxId)
    await loadTreeAndHead(activeTree.id, activeNode?.id)
  }

  const handleRefreshStale = async (sourceNodeId: string) => {
    if (!activeTree || !activeNode) return
    const res = await api.directRequest(activeNode.id, {
      target_id: sourceNodeId,
      prompt: null,
      mutate: false,
    })
    await loadTreeAndHead(activeTree.id, activeNode.id)
    showToast(
      `Refreshed summary v${String(res.result.summary_version ?? '?')} from ${sourceNodeId}`
    )
  }

  const handleDismissStale = async (sourceNodeId: string) => {
    if (!activeTree || !activeNode) return
    await api.dismissStale(activeNode.id, sourceNodeId)
    await loadTreeAndHead(activeTree.id, activeNode.id)
  }

  const handleDirectRequest = async (
    targetId: string,
    prompt: string | null,
    mutate: boolean
  ) => {
    if (!activeTree || !activeNode) return {}
    const res = await api.directRequest(activeNode.id, {
      target_id: targetId,
      prompt,
      mutate,
    })
    await loadTreeAndHead(activeTree.id, activeNode.id)
    return res.result
  }

  const handleCreateArtifact = async (payload: {
    slug: string
    title: string
    kind: string
    content: string
    summary: string
  }) => {
    if (!activeTree) return
    await api.createArtifact(activeTree.id, {
      ...payload,
      created_by_node_id: activeNode?.id ?? null,
    })
    await loadTreeAndHead(activeTree.id, activeNode?.id)
    showToast(`Saved artifact ${payload.slug}`)
  }

  const handleSaveSettings = async (payload: {
    default_model: string
    shadow_model: string
    summary_model: string
    api_base: string
    max_hop_depth: number
    token_budget: number
    auto_commit_handoffs: boolean
    api_keys?: Record<string, string>
    applyToActiveTree: boolean
  }) => {
    const updated = await api.updateSettings({
      default_model: payload.default_model,
      shadow_model: payload.shadow_model,
      summary_model: payload.summary_model,
      api_base: payload.api_base,
      max_hop_depth: payload.max_hop_depth,
      token_budget: payload.token_budget,
      auto_commit_handoffs: payload.auto_commit_handoffs,
      api_keys: payload.api_keys,
    })
    setSettings(updated)

    if (payload.applyToActiveTree && activeTree) {
      await api.updateTree(activeTree.id, {
        default_model: payload.default_model,
        shadow_model: payload.shadow_model,
        summary_model: payload.summary_model,
        auto_commit_handoffs: payload.auto_commit_handoffs,
      })
      await loadTreeAndHead(activeTree.id, activeNode?.id)
    }
    showToast(`Settings saved (model: ${payload.default_model})`)
  }

  const ipcEventsById = useMemo(() => {
    const map: Record<string, IpcEventModel> = {}
    for (const ev of snapshot?.recent_ipc ?? []) map[ev.id] = ev
    return map
  }, [snapshot?.recent_ipc])

  // Surface pending PR handoffs relevant to the active node (either authored here or targeting here)
  const relevantHandoffs = useMemo(() => {
    if (!activeNode || !snapshot) return []
    return snapshot.pending_handoffs.filter(
      (mb) => mb.from_node_id === activeNode.id || mb.to_node_id === activeNode.id
    )
  }, [activeNode, snapshot])

  const shadowCandidates = useMemo(() => {
    if (!activeNode || !snapshot) return []
    return snapshot.nodes.filter((n) => n.id !== activeNode.id)
  }, [activeNode, snapshot])

  return (
    <div className="h-screen w-screen overflow-hidden flex bg-zinc-950 text-zinc-100">
      {/* Left Sidebar: Tree Outline */}
      <TreeOutline
        trees={trees}
        activeTree={activeTree}
        nodes={snapshot?.nodes ?? []}
        activeNodeId={activeNode?.id ?? null}
        onSelectTree={handleSelectTree}
        onCreateTree={() => setNewTreeOpen(true)}
        onDeleteTree={handleDeleteTree}
        onSelectNode={(nodeId) => handleJump(nodeId)}
        onSpawnFromNode={(node) => setSpawnParent(node)}
        onShadowQueryNode={(node) => {
          setShadowTarget(node)
          setShadowModalOpen(true)
        }}
      />

      {/* Center Column: Breadcrumbs + Focused Linear Chat */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 h-full overflow-hidden relative">
        <BreadcrumbBar
          tree={activeTree}
          ancestors={nodeDetail?.ancestors ?? []}
          activeNode={activeNode}
          estimatedTokens={nodeDetail?.estimated_context_tokens ?? 0}
          inspectorOpen={inspectorOpen}
          onBack={() => handleJump('back')}
          onForward={() => handleJump('forward')}
          onSelectNode={(nodeId) => handleJump(nodeId)}
          onOpenSpawn={() => activeNode && setSpawnParent(activeNode)}
          onFinishBranch={handleFinishBranch}
          onOpenPalette={() => setPaletteOpen(true)}
          onToggleInspector={() => setInspectorOpen((v) => !v)}
          onOpenSettings={() => setSettingsOpen(true)}
          finishing={finishing}
        />

        <ChatPane
          node={activeNode}
          allNodes={snapshot?.nodes ?? []}
          inheritedMessages={nodeDetail?.inherited_messages ?? []}
          messages={nodeDetail?.messages ?? []}
          pendingHandoffs={relevantHandoffs}
          ipcEventsById={ipcEventsById}
          streamingText={streamingText}
          streamingIpc={streamingIpc}
          ephemeralNotices={ephemeralNotices}
          isStreaming={isStreaming}
          onSendMessage={handleSendMessage}
          onRefreshStale={handleRefreshStale}
          onDismissStale={handleDismissStale}
          onCommitHandoff={handleCommitHandoff}
          onDismissHandoff={handleDismissHandoff}
          onOpenSpawn={() => activeNode && setSpawnParent(activeNode)}
          onOpenShadowQuery={() => {
            setShadowTarget(null)
            setShadowModalOpen(true)
          }}
          onJumpToNode={(nodeId) => handleJump(nodeId)}
        />

        {/* Floating Toast (top-right below header so it never occludes composer) */}
        {toast && (
          <div className="pointer-events-none absolute top-14 right-4 z-40 px-3.5 py-2 rounded-lg bg-zinc-900/95 border border-indigo-500/40 text-xs text-zinc-100 shadow-xl backdrop-blur max-w-sm">
            {toast}
          </div>
        )}
      </div>

      {/* Right Sidebar: Reflog, IPC Audit Trace & Artifacts */}
      {inspectorOpen && (
        <InspectorDrawer
          nodes={snapshot?.nodes ?? []}
          reflog={snapshot?.reflog ?? []}
          reflogCursor={activeTree?.reflog_cursor ?? 0}
          ipcEvents={snapshot?.recent_ipc ?? []}
          artifacts={snapshot?.artifacts ?? []}
          activeNodeId={activeNode?.id ?? null}
          onJumpReflog={(target, note) => handleJump(target, note)}
          onCreateArtifact={handleCreateArtifact}
        />
      )}

      {/* Modals */}
      {spawnParent && (
        <SpawnModal
          parentNode={spawnParent}
          onClose={() => setSpawnParent(null)}
          onSpawn={handleSpawnChild}
        />
      )}

      {shadowModalOpen && activeNode && shadowCandidates.length > 0 && (
        <ShadowQueryModal
          callerNode={activeNode}
          candidateNodes={shadowCandidates}
          initialTargetId={shadowTarget?.id}
          onClose={() => {
            setShadowModalOpen(false)
            setShadowTarget(null)
          }}
          onRunRequest={handleDirectRequest}
        />
      )}

      {paletteOpen && activeTree && snapshot && (
        <CommandPalette
          treeId={activeTree.id}
          activeNodeId={activeNode?.id ?? null}
          nodes={snapshot.nodes}
          reflog={snapshot.reflog}
          artifacts={snapshot.artifacts}
          onClose={() => setPaletteOpen(false)}
          onSelectNode={(nodeId) => handleJump(nodeId)}
          onJumpReflog={(headRef) => handleJump(headRef)}
        />
      )}

      {settingsOpen && settings && (
        <SettingsModal
          settings={settings}
          activeTree={activeTree}
          onClose={() => setSettingsOpen(false)}
          onSave={handleSaveSettings}
        />
      )}

      {newTreeOpen && (
        <NewTreeModal
          onClose={() => setNewTreeOpen(false)}
          onCreate={handleCreateTree}
        />
      )}
    </div>
  )
}

export default App
