# DMD Transfer

This repository tracks the BSA-DMD migration workspace for Wan2.2 TI2V training.

## Layout

- `DiffSynth-Studio_BSA_DMD/`: DiffSynth-Studio fork used for BSA-DMD implementation work. It is tracked as a Git submodule pointing to the existing public fork.
- `Wan2.2-TI2V-5B-Turbo/`: reference DMD training framework and Wan2.2 template code used for migration.
- `*_实现文档.md`: implementation plans for LoRA and full-parameter BSA-DMD modes.
- `*_验收文档.md`: acceptance criteria and smoke-test gates.
- `模拟环境服务器.txt`: AutoDL validation notes.

## Version Management

Root-level documentation and reference material are versioned in this repository.
Implementation changes inside `DiffSynth-Studio_BSA_DMD/` should be committed in that submodule repository, then the root repository should update the submodule pointer.

Common flow:

```bash
git status -sb
git -C DiffSynth-Studio_BSA_DMD status -sb
```

After committing code changes inside the submodule:

```bash
git add DiffSynth-Studio_BSA_DMD
git commit -m "chore: update diffsynth bsa-dmd pointer"
git push
```

