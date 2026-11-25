import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from './api/client'
import { BreadcrumbBar } from './components/BreadcrumbBar'
import { ChatPane } from './components/ChatPane'
import { CommandPalette } from './components/CommandPalette'
import { CompactModal } from './components/CompactModal'
import { InspectorDrawer } from './components/InspectorDrawer'
import { NewTreeModal } from './components/NewTreeModal'
import { SettingsModal } from './components/SettingsModal'
import { ShadowQueryModal } from './components/ShadowQueryModal'
import { SpawnModal } from './components/SpawnModal'
import { TreeOutline } from './components/TreeOutline'
import { useKeyboardNav } from './hooks/useKeyboardNav'
import type {
  BranchFactModel,
  ChildContextMode,
  IpcEventModel,
  LLMBackendKind,
  NodeDetailResponse,
  NodeModel,
  RuntimeSettingsModel,
  SearchScope,
  ToolPermission,
  TreeInvariantModel,
  TreeModel,
  TreeSnapshotResponse,
} from './types'

export function App() {
  const [trees, setTrees] = useState<TreeModel[]>([])
  const [snapshot, setSnapshot] = useState<TreeSnapshotResponse | null>(null)
  const [activeSessionId, setActiveSessionId] = useState<string>('default')
  const [nodeDetail, setNodeDetail] = useState<NodeDetailResponse | null>(null)
  const [invariants, setInvariants] = useState<TreeInvariantModel[]>([])
  const [facts, setFacts] = useState<BranchFactModel[]>([])
  const [includeArchivedFacts, setIncludeArchivedFacts] = useState(false)
  const [settings, setSettings] = useState<RuntimeSettingsModel | null>(null)

  const [inspectorOpen, setInspectorOpen] = useState(
    () => (typeof window !== 'undefined' ? window.innerWidth >= 1100 : true)
  )
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [newTreeOpen, setNewTreeOpen] = useState(false)
  const [spawnParent, setSpawnParent] = useState<NodeModel | null>(null)
  const [compactTarget, setCompactTarget] = useState<NodeModel | null>(null)
  const [shadowTarget, setShadowTarget] = useState<NodeModel | null>(null)
  const [shadowModalMode, setShadowModalMode] = useState<'direct' | 'ask'>('direct')
  const [shadowModalOpen, setShadowModalOpen] = useState(false)

  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [streamingIpc, setStreamingIpc] = useState<IpcEventModel[]>([])
  const [ephemeralNotices, setEphemeralNotices] = useState<string[]>([])
  const [finishing, setFinishing] = useState(false)
  const [archiving, setArchiving] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const showToast = useCallback((msg: string) => {
    setToast(msg)
    setTimeout(() => {
      setToast((curr) => (curr === msg ? null : curr))
    }, 4000)
  }, [])

  const loadTreeAndHead = useCallback(
    async (
      treeId: string,
      preferredNodeId?: string,
      includeArch?: boolean,
      sessionIdOverride?: string
    ) => {
      const showArch = includeArch ?? includeArchivedFacts
      const sessId = sessionIdOverride ?? activeSessionId
      const [snap, invList, factList] = await Promise.all([
        api.getTreeSnapshot(treeId, sessId),
        api.listInvariants(treeId).catch(() => [] as TreeInvariantModel[]),
        api.listFacts(treeId, { includeArchived: showArch }).catch(() => [] as BranchFactModel[]),
      ])
      const resolvedSessId = snap.active_session_id || 'default'
      setActiveSessionId(resolvedSessId)
      setSnapshot(snap)
      setInvariants(invList)
      setFacts(factList)
      const activeSess = (snap.sessions ?? []).find((s) => s.id === resolvedSessId)
      const targetNodeId =
        preferredNodeId ||
        activeSess?.head_node_id ||
        snap.tree.head_node_id ||
        snap.tree.root_node_id
      if (targetNodeId) {
        const detail = await api.getNodeDetail(targetNodeId)
        setNodeDetail(detail)
      } else {
        setNodeDetail(null)
      }
    },
    [includeArchivedFacts, activeSessionId]
  )

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
      setInvariants([])
      setFacts([])
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
  const sessions = snapshot?.sessions ?? []
  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? sessions[0] ?? null

  const handleJump = useCallback(
    async (target: string, note = '') => {
      if (!activeTree) return
      try {
        const res = await api.reflogJump(activeTree.id, target, note, activeSessionId)
        setEphemeralNotices([])
        await loadTreeAndHead(activeTree.id, res.head_node.id, undefined, activeSessionId)
      } catch (err) {
        showToast(String(err))
      }
    },
    [activeTree, activeSessionId, loadTreeAndHead, showToast]
  )

  const handleSelectSession = async (sessionId: string) => {
    if (!activeTree) return
    setEphemeralNotices([])
    setActiveSessionId(sessionId)
    await loadTreeAndHead(activeTree.id, undefined, undefined, sessionId)
  }

  const handleCreateSession = async (title: string) => {
    if (!activeTree) return
    const created = await api.createSession(activeTree.id, {
      title,
      start_node_id: activeNode?.id ?? activeTree.root_node_id,
    })
    setActiveSessionId(created.id)
    setEphemeralNotices([])
    await loadTreeAndHead(activeTree.id, created.head_node_id ?? undefined, undefined, created.id)
    showToast(`Opened visitor session “${created.title}”`)
  }

  const handleDeleteSession = async (sessionId: string) => {
    if (!activeTree) return
    await api.deleteSession(activeTree.id, sessionId)
    const nextSess = sessionId === activeSessionId ? 'default' : activeSessionId
    setActiveSessionId(nextSess)
    await loadTreeAndHead(activeTree.id, undefined, undefined, nextSess)
    showToast('Closed visitor session')
  }

  useKeyboardNav({
    onBack: () => handleJump('back'),
    onForward: () => handleJump('forward'),
    onOpenPalette: () => setPaletteOpen((v) => !v),
  })

  const handleSelectTree = async (treeId: string) => {
    setEphemeralNotices([])
    setActiveSessionId('default')
    await loadTreeAndHead(treeId, undefined, undefined, 'default')
  }

  const handleCreateTree = async (title: string, goal: string) => {
    const created = await api.createTree({ title, goal })
    setTrees((prev) => [created.tree, ...prev])
    setSnapshot(created)
    setInvariants([])
    setFacts([])
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
      await api.streamTurn(
        activeNode.id,
        content,
        {
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
        },
        undefined,
        activeSessionId
      )
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
    tool_permission: ToolPermission
    switch_head: boolean
  }) => {
    if (!activeTree || !spawnParent) return
    const res = await api.spawnChild(spawnParent.id, {
      ...payload,
      session_id: activeSessionId,
    })
    const nextId =
      payload.switch_head && res.child ? res.child.id : activeNode?.id || spawnParent.id
    await loadTreeAndHead(activeTree.id, nextId)
    if (res.child) {
      showToast(`Spawned ${res.child.context_mode} branch “${res.child.title}”`)
    }
  }

  const handleCompactBranch = async (payload: {
    keep_recent_turns: number
    title?: string
    reason?: string
    switch_head: boolean
  }) => {
    if (!activeTree || !compactTarget) return
    const info = await api.compactNode(compactTarget.id, {
      ...payload,
      session_id: activeSessionId,
    })
    const nextId = info.switched_head ? info.to_node_id : activeNode?.id || compactTarget.id
    await loadTreeAndHead(activeTree.id, nextId)
    showToast(
      `Compacted “${info.from_node_title}” → “${info.to_node_title}” (${info.tokens_before}→${info.tokens_after} tok, ${info.compression_ratio}×)`
    )
  }

  const handleArchiveBranch = async (nodeToArchive?: NodeModel) => {
    const target = nodeToArchive ?? activeNode
    if (!activeTree || !target || !target.parent_id) return
    setArchiving(true)
    try {
      const res = await api.archiveNode(target.id, {
        reason: 'Archived via UI',
        switch_to_parent: true,
        session_id: activeSessionId,
      })
      const nextId =
        res.switched_head_to ||
        (activeNode?.id === target.id ? target.parent_id : activeNode?.id) ||
        undefined
      await loadTreeAndHead(activeTree.id, nextId)
      showToast(`Archived “${res.archived_node_title}” — quarantined from default 4-tier RAG.`)
    } catch (err) {
      showToast(String(err))
    } finally {
      setArchiving(false)
    }
  }

  const handleAskTree = async (
    query: string,
    scope: SearchScope,
    maxTargets: number,
    deep: boolean
  ) => {
    if (!activeTree || !activeNode) return {}
    const res = await api.askTree(activeNode.id, {
      query,
      scope,
      max_targets: maxTargets,
      deep,
    })
    await loadTreeAndHead(activeTree.id, activeNode.id)
    return res.result
  }

  const handlePinInvariant = async (payload: {
    key: string
    value: string
    source_node_id?: string | null
  }) => {
    if (!activeTree) return
    await api.pinInvariant(activeTree.id, payload)
    await loadTreeAndHead(activeTree.id, activeNode?.id)
    showToast(`Pinned invariant ${payload.key}`)
  }

  const handleUnpinInvariant = async (key: string) => {
    if (!activeTree) return
    await api.unpinInvariant(activeTree.id, key)
    await loadTreeAndHead(activeTree.id, activeNode?.id)
    showToast(`Unpinned invariant ${key}`)
  }

  const handleAddFact = async (payload: {
    node_id: string
    fact: string
    resolved_date?: string
    entities: string[]
  }) => {
    if (!activeTree) return
    const existingForNode = facts.filter((f) => f.node_id === payload.node_id)
    const merged = [
      ...existingForNode.map((f) => ({
        fact: f.fact_text,
        resolved_date: f.resolved_date || undefined,
        entities: f.entities,
      })),
      {
        fact: payload.fact,
        resolved_date: payload.resolved_date,
        entities: payload.entities,
      },
    ]
    await api.addFacts(activeTree.id, { node_id: payload.node_id, facts: merged })
    await loadTreeAndHead(activeTree.id, activeNode?.id)
    showToast('Added Tier-2.5 atomic temporal fact')
  }

  const handleToggleIncludeArchivedFacts = async (include: boolean) => {
    setIncludeArchivedFacts(include)
    if (activeTree) {
      const factList = await api
        .listFacts(activeTree.id, { includeArchived: include })
        .catch(() => [] as BranchFactModel[])
      setFacts(factList)
    }
  }

  const handleFinishBranch = async () => {
    if (!activeTree || !activeNode || !activeNode.parent_id) return
    setFinishing(true)
    try {
      const res = await api.finishNode(activeNode.id, {
        return_to_parent: true,
        session_id: activeSessionId,
      })
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
      session_id: activeSessionId,
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
  }) => {
    const updated = await api.updateSettings({
      llm_provider: payload.llm_provider,
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
    showToast(`Settings saved (${payload.llm_provider} · ${payload.default_model})`)
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
        sessions={sessions}
        activeSessionId={activeSessionId}
        activeNodeId={activeNode?.id ?? null}
        onSelectTree={handleSelectTree}
        onCreateTree={() => setNewTreeOpen(true)}
        onDeleteTree={handleDeleteTree}
        onSelectNode={(nodeId) => handleJump(nodeId)}
        onSpawnFromNode={(node) => setSpawnParent(node)}
        onShadowQueryNode={(node) => {
          setShadowTarget(node)
          setShadowModalMode('direct')
          setShadowModalOpen(true)
        }}
        onCompactNode={(node) => setCompactTarget(node)}
        onArchiveNode={(node) => handleArchiveBranch(node)}
      />

      {/* Center Column: Breadcrumbs + Focused Linear Chat */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 h-full overflow-hidden relative">
        <BreadcrumbBar
          tree={activeTree}
          nodes={snapshot?.nodes ?? []}
          sessions={sessions}
          activeSessionId={activeSessionId}
          ancestors={nodeDetail?.ancestors ?? []}
          activeNode={activeNode}
          estimatedTokens={nodeDetail?.estimated_context_tokens ?? 0}
          inspectorOpen={inspectorOpen}
          onSelectSession={handleSelectSession}
          onCreateSession={handleCreateSession}
          onDeleteSession={handleDeleteSession}
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
            setShadowModalMode('direct')
            setShadowModalOpen(true)
          }}
          onOpenAskTree={() => {
            setShadowTarget(null)
            setShadowModalMode('ask')
            setShadowModalOpen(true)
          }}
          onOpenCompact={() => activeNode && setCompactTarget(activeNode)}
          onArchiveNode={() => handleArchiveBranch()}
          archiving={archiving}
          onJumpToNode={(nodeId) => handleJump(nodeId)}
        />

        {/* Floating Toast (top-right below header so it never occludes composer) */}
        {toast && (
          <div className="pointer-events-none absolute top-14 right-4 z-40 px-3.5 py-2 rounded-lg bg-zinc-900/95 border border-indigo-500/40 text-xs text-zinc-100 shadow-xl backdrop-blur max-w-sm">
            {toast}
          </div>
        )}
      </div>

      {/* Right Sidebar: Reflog, IPC Audit Trace, Artifacts & Memory (Pins + Facts) */}
      {inspectorOpen && (
        <InspectorDrawer
          nodes={snapshot?.nodes ?? []}
          reflog={snapshot?.reflog ?? []}
          reflogCursor={activeSession?.reflog_cursor ?? activeTree?.reflog_cursor ?? 0}
          ipcEvents={snapshot?.recent_ipc ?? []}
          artifacts={snapshot?.artifacts ?? []}
          invariants={invariants}
          facts={facts}
          activeNodeId={activeNode?.id ?? null}
          includeArchivedFacts={includeArchivedFacts}
          onToggleIncludeArchivedFacts={handleToggleIncludeArchivedFacts}
          onJumpReflog={(target, note) => handleJump(target, note)}
          onJumpToNode={(nodeId) => handleJump(nodeId)}
          onCreateArtifact={handleCreateArtifact}
          onPinInvariant={handlePinInvariant}
          onUnpinInvariant={handleUnpinInvariant}
          onAddFact={handleAddFact}
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

      {compactTarget && (
        <CompactModal
          sourceNode={compactTarget}
          estimatedTokens={
            compactTarget.id === activeNode?.id
              ? nodeDetail?.estimated_context_tokens ?? 0
              : compactTarget.tokens_in
          }
          onClose={() => setCompactTarget(null)}
          onCompact={handleCompactBranch}
        />
      )}

      {shadowModalOpen && activeNode && shadowCandidates.length > 0 && (
        <ShadowQueryModal
          callerNode={activeNode}
          candidateNodes={shadowCandidates}
          initialTargetId={shadowTarget?.id}
          initialMode={shadowModalMode}
          onClose={() => {
            setShadowModalOpen(false)
            setShadowTarget(null)
          }}
          onRunRequest={handleDirectRequest}
          onRunAskTree={handleAskTree}
          onJumpToNode={(nodeId) => handleJump(nodeId)}
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
