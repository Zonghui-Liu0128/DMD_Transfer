# BSA-DMD LoRA 模式实现文档

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `DiffSynth-Studio_BSA_DMD` 中实现 Wan2.2-5B BSA-DMD LoRA 训练：以 full-attention 50-step teacher + LoRA(rank=32) 蒸馏得到 BSA 4-step student + LoRA(rank=32)。

**Architecture:** 新增 DMD 专用训练路径，保留现有 SFT/direct_distill 路径。DMD 路径包含 frozen teacher、trainable fake_score LoRA、trainable student LoRA 三个 DiT；BSA 只配置到 student；最终只导出 student LoRA。

**Tech Stack:** PyTorch、Accelerate/DeepSpeed、DiffSynth WanVideoPipeline、PEFT LoRA、Wan BSA、FlowMatchScheduler。

---

## 1. 当前状态

- 当前 Wan 训练入口 `examples/wanvideo/model_training/train.py` 只注册 `sft`、`direct_distill`。
- 当前 runner `diffsynth/diffusion/runner.py::launch_training_task` 是单 loss、单 optimizer。
- 当前 LoRA 注入在 `DiffusionTrainingModule.switch_pipe_to_training_mode` 中完成，适合单模型 LoRA SFT，但不能直接表达 teacher/fake/student 三模型。
- 当前 BSA 能力已存在于 `diffsynth/models/wan_bsa.py`，可直接配置 student DiT。

## 2. 目标语义

- teacher：Wan2.2-5B full attention，冻结，加载 teacher LoRA；无内网 LoRA 时允许 base teacher 或 mock teacher LoRA smoke。
- student：Wan2.2-5B BSA，训练 LoRA(rank=32)，从 teacher/mock LoRA 初始化；这是最终导出对象。
- fake_score：Wan2.2-5B full attention，训练 LoRA(rank=32)，从 teacher/mock LoRA 初始化；只用于 DMD 训练和 resume。
- DMD 步表固定为 `[1000,750,500,250]`。
- TTUR：fake_score 每个 outer step 更新；student 每 `fake_score_updates_per_generator_update=5` 个 outer step 更新一次。

## 3. 文件修改清单

| 文件 | 动作 | 责任 |
| --- | --- | --- |
| `DiffSynth-Studio_BSA_DMD/examples/wanvideo/model_training/train.py` | 修改 | 增加 `dmd_lora` / `dmd_lora:train` task，解析 DMD LoRA 参数，接入 DMD runner。 |
| `DiffSynth-Studio_BSA_DMD/examples/wanvideo/model_training/wan_dmd_lora_training.py` | 新增 | 构建 LoRA DMD 三模型模块，负责加载 teacher/student/fake_score、BSA 配置、pipeline inputs。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/dmd_loss.py` | 新增 | 实现 student 4-step rollout、random exit、DMD gradient、fake_score x0 loss、Wan flow->x0。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/dmd_runner.py` | 新增 | 实现 DMD LoRA 训练循环、双 optimizer、TTUR、metrics、checkpoint 调用。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/dmd_checkpoint.py` | 新增 | 保存/恢复 student LoRA、fake_score LoRA、两个 optimizer/scheduler、trainer state；final 只导出 student LoRA。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/training_metrics.py` | 修改 | 增加 DMD 字段：student/fake loss、grad norm、DMD grad、forward count、effective tokens。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/__init__.py` | 修改 | 导出 DMD loss/runner/checkpoint 工具。 |
| `DiffSynth-Studio_BSA_DMD/train_Wan2.2_5B_BSA_DMD_LoRA.sh` | 新增 | 提供 LoRA BSA-DMD 启动脚本。 |
| `DiffSynth-Studio_BSA_DMD/tests/test_wan_dmd_lora_training.py` | 新增 | 覆盖 LoRA trainable 范围、BSA 只作用 student、TTUR、checkpoint key。 |

## 4. 关键实现任务

### Task 1: 训练入口与参数

- [ ] 在 `train.py` 中新增 task：`dmd_lora`、`dmd_lora:train`。
- [ ] 新增参数：
  - `--teacher_model_paths`
  - `--teacher_lora_checkpoint`
  - `--student_init_lora_checkpoint`
  - `--fake_score_init_lora_checkpoint`
  - `--student_lora_rank`
  - `--fake_score_lora_rank`
  - `--dmd_denoising_steps`
  - `--real_guidance_scale`
  - `--fake_guidance_scale`
  - `--fake_score_updates_per_generator_update`
  - `--student_learning_rate`
  - `--fake_score_learning_rate`
  - `--fake_score_loss_type`
- [ ] `launcher_map` 将 `dmd_lora` 和 `dmd_lora:train` 映射到 `launch_dmd_lora_training_task`。

### Task 2: LoRA 三模型模块

- [ ] 新建 `WanDMDLoRATrainingModule`。
- [ ] 复用 `WanVideoPipeline.from_pretrained` 加载共享 VAE、text encoder、scheduler。
- [ ] 创建三个互相独立的 DiT：
  - `teacher_dit.requires_grad_(False)`
  - `student_dit` 注入 LoRA rank=32
  - `fake_score_dit` 注入 LoRA rank=32
- [ ] 只对 `student_dit` 调用 `configure_wan_bsa`。
- [ ] teacher/fake_score 保持 full attention。
- [ ] 有 teacher/mock LoRA 时，teacher、student、fake_score 从同一 LoRA 初始化；无 LoRA smoke 时 student/fake_score 用空 LoRA 初始化。

### Task 3: DMD loss

- [ ] 在 `dmd_loss.py` 中实现 `run_student_rollout`：
  - 输入 first-frame latent、prompt embedding、noise latent。
  - 使用 `[1000,750,500,250]`。
  - 非 exit step 使用 `torch.no_grad()`。
  - exit step 保留 student 梯度。
  - 每步后使用 fresh noise 和下一 timestep re-noise。
  - 每次 re-noise 后替换第 0 帧为 first-frame latent。
- [ ] 实现 `compute_dmd_gradient`：
  - teacher cond/uncond forward，CFG scale 默认 6.0。
  - fake_score cond forward，保留 fake CFG 参数。
  - flow prediction 转 x0。
  - `grad = pred_fake_x0 - pred_real_x0`。
  - `grad /= abs(x0 - pred_real_x0).mean(...) + eps`。
- [ ] 实现 stop-gradient MSE，把手工 DMD 梯度传给 student LoRA。
- [ ] 实现 `compute_fake_score_loss`，默认 `x0` target。

### Task 4: DMD runner

- [ ] 创建两个 optimizer：
  - `optimizer_student`：只包含 student LoRA 参数，默认 lr `5e-7`。
  - `optimizer_fake_score`：只包含 fake_score LoRA 参数，默认 lr `1e-7`。
- [ ] 每个 outer step 都执行 fake_score loss/backward/step。
- [ ] 当 `outer_step % 5 == 0` 时执行 student DMD loss/backward/step。
- [ ] teacher 永远不进入 optimizer。
- [ ] 记录 student/fake/teacher 参数 delta debug 信息。

### Task 5: Checkpoint 与导出

- [ ] 中间 checkpoint 保存：
  - `student_lora.safetensors`
  - `fake_score_lora.safetensors`
  - `student_optimizer.pt`
  - `fake_score_optimizer.pt`
  - `trainer_state.json`
- [ ] resume 时恢复两个 LoRA、两个 optimizer、global step、随机状态。
- [ ] final export 只保存 `student_lora.safetensors`。

### Task 6: 训练脚本

- [ ] 新增 `train_Wan2.2_5B_BSA_DMD_LoRA.sh`。
- [ ] 默认值：
  - `LORA_RANK=32`
  - `DMD_DENOISING_STEPS=1000,750,500,250`
  - `REAL_GUIDANCE_SCALE=6.0`
  - `FAKE_SCORE_UPDATES_PER_GENERATOR_UPDATE=5`
  - `STUDENT_LR=5e-7`
  - `FAKE_SCORE_LR=1e-7`
  - `BSA_BLOCK_SIZE=3,7,3`
  - `BSA_SPARSE_RATIO=0.85`
  - `BSA_BACKEND=sdpa_chunked`

## 5. 测试要求

- parser 能识别 `dmd_lora` 参数。
- student/fake_score 只有 LoRA 参数 `requires_grad=True`。
- teacher 参数全部冻结。
- BSA module 只存在于 student。
- fake_score 每步更新，student 每 5 步更新。
- final export 不包含 fake_score key。
- 256x256@81 的 `tokens_per_sample=1344`。

## 6. 不在本方案内

- 不训练 full DiT 参数。
- 不保存 full student/fake_score DiT checkpoint 作为最终产物。
- 不修改 BSA attention 内核，除非验收发现 BSA 与 Wan2.2 separated timestep 不兼容。
