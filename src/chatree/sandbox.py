from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from chatree.types import ArtifactInput


@dataclass(frozen=True)
class SandboxWorkspace:
    node_id: str
    path: Path
    branch_name: str
    base_ref: str


@runtime_checkable
class BranchSandbox(Protocol):
    """
    Pluggable world-state isolation protocol binding writable (`read_write`) dialogue
    branches to isolated filesystems / Git worktrees / ephemeral containers.
    """

    async def create_workspace(
        self, node_id: str, *, base_node_id: str | None = None
    ) -> SandboxWorkspace: ...

    def get_workspace(self, node_id: str) -> SandboxWorkspace | None: ...

    async def diff(self, node_id: str) -> str: ...

    async def diff_artifact(self, node_id: str, *, title: str | None = None) -> ArtifactInput | None: ...

    async def commit_all(self, node_id: str, message: str) -> str | None: ...

    async def merge_into(
        self, child_node_id: str, *, target_node_id: str | None = None
    ) -> dict[str, Any]: ...

    async def cleanup(self, node_id: str) -> None: ...


class GitWorktreeSandbox:
    """
    Git worktree isolation for parallel `read_write` coding branches:
      - Each writable child gets its own working copy (`<worktrees_dir>/<node_id>`)
        on branch `<branch_prefix>/<node_id>` so parallel branches cannot clobber files.
      - Captures unified diffs as versioned `ArtifactInput` payloads on `finish()`.
      - Merges winning branches cleanly into the parent worktree (or repo root) on `promote_child()`
        and cleans up archived sibling worktrees.
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        worktrees_dir: str | Path | None = None,
        branch_prefix: str = "chatree",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.worktrees_dir = (
            Path(worktrees_dir).resolve()
            if worktrees_dir is not None
            else (self.repo_root.parent / f".{self.repo_root.name}-chatree-worktrees").resolve()
        )
        self.branch_prefix = branch_prefix
        self._workspaces: dict[str, SandboxWorkspace] = {}
        self._repo: Any = None

    def attach_repository(self, repo: Any) -> None:
        """Attach a Chatree repository so workspace mappings persist across restarts."""
        self._repo = repo

    async def hydrate(self) -> None:
        """Reload persisted workspaces from SQLite and reconcile with `git worktree list`."""
        if self._repo is not None and hasattr(self._repo, "list_sandbox_workspaces"):
            for row in await self._repo.list_sandbox_workspaces():
                p = Path(row["path"])
                if p.exists():
                    self._workspaces[row["node_id"]] = SandboxWorkspace(
                        node_id=row["node_id"],
                        path=p,
                        branch_name=row["branch_name"],
                        base_ref=row["base_ref"],
                    )
        # Also discover any live chatree worktrees on disk even without a persisted row
        rc, out, _ = await self._run_git(self.repo_root, "worktree", "list", "--porcelain")
        if rc != 0:
            return
        cur_path: Path | None = None
        cur_branch: str | None = None
        prefix_ref = f"refs/heads/{self.branch_prefix}/"
        for line in [*out.splitlines(), ""]:
            if line.startswith("worktree "):
                cur_path = Path(line.split(" ", 1)[1].strip()).resolve()
                cur_branch = None
            elif line.startswith("branch "):
                cur_branch = line.split(" ", 1)[1].strip()
            elif not line.strip():
                if cur_path and cur_branch and cur_branch.startswith(prefix_ref):
                    nid = cur_branch.removeprefix(prefix_ref)
                    if nid not in self._workspaces and cur_path.exists():
                        _rc2, head_sha, _ = await self._run_git(self.repo_root, "rev-parse", "HEAD")
                        self._workspaces[nid] = SandboxWorkspace(
                            node_id=nid,
                            path=cur_path,
                            branch_name=f"{self.branch_prefix}/{nid}",
                            base_ref=head_sha or "HEAD",
                        )
                cur_path = None
                cur_branch = None

    async def _run_git(
        self, cwd: Path, *args: str, env_extra: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout_b.decode("utf-8", errors="replace").strip(),
            stderr_b.decode("utf-8", errors="replace").strip(),
        )

    def _branch_name(self, node_id: str) -> str:
        return f"{self.branch_prefix}/{node_id}"

    async def create_workspace(
        self, node_id: str, *, base_node_id: str | None = None
    ) -> SandboxWorkspace:
        if node_id in self._workspaces and self._workspaces[node_id].path.exists():
            return self._workspaces[node_id]

        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        wt_path = self.worktrees_dir / node_id
        branch = self._branch_name(node_id)

        if base_node_id and base_node_id in self._workspaces:
            base_ref = self._workspaces[base_node_id].branch_name
        else:
            rc, head_sha, err = await self._run_git(self.repo_root, "rev-parse", "HEAD")
            if rc != 0:
                raise RuntimeError(f"Git repo at '{self.repo_root}' has no valid HEAD: {err}")
            base_ref = head_sha

        rc, _out, err = await self._run_git(
            self.repo_root, "worktree", "add", "-b", branch, str(wt_path), base_ref
        )
        if rc != 0:
            raise RuntimeError(f"Failed to create git worktree for '{node_id}': {err}")

        ws = SandboxWorkspace(
            node_id=node_id, path=wt_path, branch_name=branch, base_ref=base_ref
        )
        self._workspaces[node_id] = ws
        if self._repo is not None and hasattr(self._repo, "upsert_sandbox_workspace"):
            await self._repo.upsert_sandbox_workspace(node_id, str(wt_path), branch, base_ref)
        return ws

    def get_workspace(self, node_id: str) -> SandboxWorkspace | None:
        return self._workspaces.get(node_id)

    async def commit_all(self, node_id: str, message: str) -> str | None:
        ws = self._workspaces.get(node_id)
        if ws is None:
            return None
        await self._run_git(ws.path, "add", "-A")
        _rc, status_out, _err = await self._run_git(ws.path, "status", "--porcelain")
        if not status_out:
            return None
        rc, _out, err = await self._run_git(
            ws.path,
            "-c",
            "user.name=Chatree",
            "-c",
            "user.email=chatree@local",
            "commit",
            "-m",
            message,
        )
        if rc != 0:
            raise RuntimeError(f"Git commit failed in worktree '{node_id}': {err}")
        _rc2, sha, _ = await self._run_git(ws.path, "rev-parse", "HEAD")
        return sha or None

    async def _read_only_diff(self, ws: SandboxWorkspace, extra_args: tuple[str, ...] = ()) -> str:
        """Compute unified diff including untracked files using a temp GIT_INDEX_FILE (zero index mutation)."""
        _rc, git_index_raw, _ = await self._run_git(ws.path, "rev-parse", "--git-path", "index")
        orig_index = Path(git_index_raw)
        if not orig_index.is_absolute():
            orig_index = (ws.path / orig_index).resolve()
        with tempfile.TemporaryDirectory(prefix="chatree-idx-") as tmpdir:
            tmp_index = Path(tmpdir) / "index"
            if orig_index.exists():
                shutil.copy2(orig_index, tmp_index)
            env_extra = {"GIT_INDEX_FILE": str(tmp_index)}
            await self._run_git(ws.path, "add", "-N", ".", env_extra=env_extra)
            _rc2, diff_out, _ = await self._run_git(
                ws.path, "diff", *extra_args, ws.base_ref, env_extra=env_extra
            )
            return diff_out

    async def diff(self, node_id: str) -> str:
        ws = self._workspaces.get(node_id)
        if ws is None:
            return ""
        return await self._read_only_diff(ws)

    async def diff_artifact(
        self, node_id: str, *, title: str | None = None
    ) -> ArtifactInput | None:
        patch = await self.diff(node_id)
        if not patch.strip():
            return None
        ws = self._workspaces[node_id]
        stat = await self._read_only_diff(ws, ("--stat",))
        summary_line = " ".join(stat.splitlines()[-1].split()) if stat else "Worktree diff patch"
        return ArtifactInput(
            slug=f"{node_id}.diff",
            title=title or f"Worktree Patch ({ws.branch_name})",
            kind="diff",
            content=patch,
            summary=summary_line,
        )

    async def merge_into(
        self, child_node_id: str, *, target_node_id: str | None = None
    ) -> dict[str, Any]:
        child_ws = self._workspaces.get(child_node_id)
        if child_ws is None:
            return {"merged": False, "reason": f"No sandbox workspace for '{child_node_id}'"}

        await self.commit_all(child_node_id, f"Chatree snapshot for branch {child_node_id}")

        target_path = (
            self._workspaces[target_node_id].path
            if target_node_id and target_node_id in self._workspaces
            else self.repo_root
        )
        rc, out, err = await self._run_git(
            target_path,
            "-c",
            "user.name=Chatree",
            "-c",
            "user.email=chatree@local",
            "merge",
            "--no-ff",
            "-m",
            f"Merge Chatree branch {child_ws.branch_name}",
            child_ws.branch_name,
        )
        if rc != 0:
            await self._run_git(target_path, "merge", "--abort")
            return {
                "merged": False,
                "branch": child_ws.branch_name,
                "error": err or out,
            }
        _rc2, sha, _ = await self._run_git(target_path, "rev-parse", "HEAD")
        return {
            "merged": True,
            "branch": child_ws.branch_name,
            "commit_sha": sha,
            "target_path": str(target_path),
        }

    async def cleanup(self, node_id: str) -> None:
        ws = self._workspaces.pop(node_id, None)
        if self._repo is not None and hasattr(self._repo, "delete_sandbox_workspace"):
            await self._repo.delete_sandbox_workspace(node_id)
        if ws is None:
            return
        await self._run_git(self.repo_root, "worktree", "remove", "--force", str(ws.path))
        await self._run_git(self.repo_root, "branch", "-D", ws.branch_name)
