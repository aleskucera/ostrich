# Renaming the project directory: `axion` → `ostrich`

The code rename (package, imports, files, docs) is already committed on branch
`rename-axion-to-ostrich`. This note covers the remaining step: renaming the
**local working directory** and re-linking everything that pins its absolute path.

> Run these in a **regular terminal**, not from inside a Claude Code session —
> renaming the directory invalidates the running session's working directory.
> Start a fresh `claude` afterward.

## Steps

```bash
# 0. Make sure work is committed (done: branch rename-axion-to-ostrich).
#    Nothing should be in-flight before moving.

# 1. Rename the project directory
mv ~/projects/axion ~/projects/ostrich

# 2. Move into it and rebuild the virtualenv (venvs are not relocatable)
cd ~/projects/ostrich
rm -rf .venv
uv sync --extra sim          # add the extras you use, e.g. --extra docs

# 3. Verify the package resolves at the new path
.venv/bin/python -c "import ostrich; print('ok:', ostrich.__file__)"

# 4. Carry over Claude Code session history + memory (keyed by the path)
mv ~/.claude/projects/-home-kuceral4-projects-axion \
   ~/.claude/projects/-home-kuceral4-projects-ostrich

# 5. Git remote (do once GitHub repo is renamed)
git remote set-url origin git@github.com:aleskucera/ostrich.git
```

Then:

```bash
cd ~/projects/ostrich
claude            # fresh session; --resume shows old history, memory is recalled
```

## Why each step

- **Step 1** — Git is path-independent (zero tracked files hardcode the path),
  so the repo just works under any directory name.
- **Step 2** — The only thing the move breaks is `.venv`: its `bin/*` shebangs and
  the editable-install `.pth` contain the old absolute path. `uv sync` regenerates
  them. (Warp's kernel cache in `~/.cache/warp` and Hydra's relative `outputs/`
  are path-independent.)
- **Step 4** — Claude Code keys session transcripts and auto-memory by the
  encoded directory path (`/home/kuceral4/projects/axion` →
  `-home-kuceral4-projects-axion`). Without this rename, the old sessions and
  memories stay on disk but are not found from the new path — Claude starts fresh.
  Old transcripts keep the previous `cwd` in their metadata; that is cosmetic and
  does not affect resuming.

## Alternative

Instead of moving, you could `git clone` fresh into `~/projects/ostrich`. The repo
would be correct, but you would start with empty Claude history/memory — which is
why the `mv` in step 4 matters if you want continuity.
