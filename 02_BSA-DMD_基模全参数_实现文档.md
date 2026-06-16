# BSA-DMD 基模全参数模式实现文档

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `DiffSynth-Studio_BSA_DMD` 中实现 Wan2.2-5B 基模全参数 BSA-DMD 训练：不使用 LoRA，直接更新 student BSA DiT 和 fake_score DiT 的全量参数。

**Architecture:** 复用 DMD 三模型结构，但去掉 LoRA 注入和 LoRA checkpoint 语义。teacher 是冻结 full-attention DiT；student 是 BSA DiT 且全参数可训练；fake_score 是 full-attention DiT 且全参数可训练。BSA 本身不新增训练权重，但必须保证梯度能穿过 BSA attention 回到 student 的 q/k/v/o/MLP 等原始参数。

**Tech Stack:** PyTorch、Accelerate/DeepSpeed 或 FSDP、DiffSynth WanVideoPipeline、Wan BSA、FlowMatchScheduler、sharded checkpoint。

---

## 1. 当前状态

- 当前框架没有 DMD task。
- 当前 Wan 训练可以通过 `--trainable_models dit` 让单个 pipeline 的 DiT 全参可训练，但这只适用于 SFT/direct_distill 单模型路径。
- 当前 checkpoint logger 只导出 trainable state dict，不支持 DMD 三模型全量 checkpoint。
- 当前 BSA 作为 attention 计算路径存在，但还没有专门验证 full-param 训练下梯度覆盖所有原始 DiT 权重。

## 2. 目标语义

- teacher：Wan2.2-5B full attention，冻结，可来自 base checkpoint 或单独 teacher full checkpoint。
- student：Wan2.2-5B BSA，全参数训练，最终导出 full student DiT checkpoint。
- fake_score：Wan2.2-5B full attention，全参数训练，用于拟合 student fake distribution，只保存 resume checkpoint，不进入最终推理产物。
- 无 LoRA 注入、无 LoRA rank、无 LoRA final export。
- DMD 步表、CFG、TTUR 与 LoRA 方案一致。

## 3. 文件修改清单

| 文件 | 动作 | 责任 |
| --- | --- | --- |
| `DiffSynth-Studio_BSA_DMD/examples/wanvideo/model_training/train.py` | 修改 | 增加 `dmd_full` / `dmd_full:train` task，解析 full-param DMD 参数。 |
| `DiffSynth-Studio_BSA_DMD/examples/wanvideo/model_training/wan_dmd_full_training.py` | 新增 | 构建 full-param DMD 三模型模块，显式冻结 teacher，显式解冻 student/fake_score 全参数。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/dmd_loss.py` | 新增/复用 | 使用与 LoRA 相同的 DMD loss，但不得依赖 LoRA key 或 adapter API。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/dmd_runner.py` | 新增/扩展 | 支持 full-param 大模型训练、TTUR、梯度裁剪、显存/offload 统计。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/dmd_checkpoint.py` | 新增/扩展 | 保存 full student/fake_score checkpoint、optimizer state、trainer state；final 只导出 full student DiT。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/models/wan_bsa.py` | 可能修改 | 增加 full-param 梯度自检 helper，确认 BSA 不 detach q/k/v/o/MLP 梯度。 |
| `DiffSynth-Studio_BSA_DMD/diffsynth/diffusion/training_metrics.py` | 修改 | 增加 full-param 参数量、optimizer state 显存、peak memory、effective tokens。 |
| `DiffSynth-Studio_BSA_DMD/train_Wan2.2_5B_BSA_DMD_FULL.sh` | 新增 | 提供全参 DMD 启动脚本，默认启用更保守的 batch/resolution/offload 配置。 |
| `DiffSynth-Studio_BSA_DMD/tests/test_wan_dmd_full_training.py` | 新增 | 覆盖 full-param trainable 范围、BSA 梯度、checkpoint 语义。 |

## 4. 关键实现任务

### Task 1: 训练入口与参数

- [ ] 新增 task：`dmd_full`、`dmd_full:train`。
- [ ] 新增参数：
  - `--teacher_model_paths`
  - `--student_init_model_paths`
  - `--fake_score_init_model_paths`
  - `--student_checkpoint`
  - `--fake_score_checkpoint`
  - `--dmd_denoising_steps`
  - `--real_guidance_scale`
  - `--fake_guidance_scale`
  - `--fake_score_updates_per_generator_update`
  - `--student_learning_rate`
  - `--fake_score_learning_rate`
  - `--student_weight_decay`
  - `--fake_score_weight_decay`
  - `--full_param_checkpoint_format`
  - `--enable_dmd_activation_checkpointing`
  - `--enable_dmd_optimizer_offload`
- [ ] 禁止同时传入 LoRA 参数和 `dmd_full` task；传入时直接报错。

### Task 2: 全参三模型模块

- [ ] 新建 `WanDMDFullTrainingModule`。
- [ ] 加载 teacher/student/fake_score 三个独立 DiT。
- [ ] teacher 调用 `requires_grad_(False)` 并进入 eval。
- [ ] student 调用 `requires_grad_(True)`，再配置 BSA。
- [ ] fake_score 调用 `requires_grad_(True)`，保持 full attention。
- [ ] 不调用 `add_lora_to_model`、`load_lora`、`mapping_lora_state_dict`。
- [ ] 输出 trainable parameter count：
  - teacher trainable = 0
  - student trainable ≈ full DiT 参数量
  - fake_score trainable ≈ full DiT 参数量

### Task 3: BSA 支持全参数梯度

- [ ] 在 BSA 单元测试中构造小 Wan attention block。
- [ ] `sparse_ratio=0.0` 时，对比 full attention 与 BSA 输出误差。
- [ ] 对一个标量 loss backward，确认 q/k/v/o/MLP 参数都有非空梯度。
- [ ] `sparse_ratio=0.85` 时确认梯度有限，无 NaN/Inf。
- [ ] 若 BSA 当前实现使用 detach、no_grad 或非 differentiable mask，需要改为只对 mask/index no-grad，attention value 路径保持可导。

### Task 4: DMD loss

- [ ] 复用 LoRA DMD loss 的 rollout、DMD gradient、fake_score x0 loss。
- [ ] 所有梯度检查从 LoRA key 改为 full parameter name。
- [ ] student generator pass 应让 q/k/v/o/ffn 参数产生梯度。
- [ ] critic pass 应让 fake_score q/k/v/o/ffn 参数产生梯度。

### Task 5: Runner 与 optimizer

- [ ] 默认实现两个 optimizer：
  - `optimizer_student`：student 全参数。
  - `optimizer_fake_score`：fake_score 全参数。
- [ ] 如果当前 Accelerate/DeepSpeed 配置不能稳定支持两个 full-param optimizer，则实现 full-param fallback：单 optimizer、两个 param group、按 pass 控制 inactive group 的 grad 为 `None`。
- [ ] 每步记录 optimizer state memory、peak GPU memory、step time。
- [ ] 默认启用 gradient checkpointing；推荐开启 optimizer CPU offload。
- [ ] 对 full-param 模式强制打印资源警告：三份 5B DiT + 两份 optimizer state 显存成本远高于 LoRA。

### Task 6: Checkpoint 与导出

- [ ] 中间 checkpoint 保存：
  - `student_dit/` sharded safetensors 或 DeepSpeed checkpoint
  - `fake_score_dit/` sharded safetensors 或 DeepSpeed checkpoint
  - optimizer state
  - scheduler state
  - `trainer_state.json`
  - BSA 配置
- [ ] final export 只保存 full student DiT checkpoint 和 BSA 推理配置。
- [ ] final export 不保存 fake_score，不保存 optimizer。
- [ ] 提供从 full student checkpoint 加载到 4-step BSA inference 的验证入口。

### Task 7: 启动脚本

- [ ] 新增 `train_Wan2.2_5B_BSA_DMD_FULL.sh`。
- [ ] 默认使用小分辨率 smoke：
  - `HEIGHT=256`
  - `WIDTH=256`
  - `NUM_FRAMES=81`
  - `BATCH_SIZE=1`
- [ ] 默认打开：
  - `BSA_ENABLE=1`
  - `BSA_BLOCK_SIZE=3,7,3`
  - `BSA_SPARSE_RATIO=0.85`
  - `ENABLE_DMD_ACTIVATION_CHECKPOINTING=1`
  - `ENABLE_DMD_OPTIMIZER_OFFLOAD=1`
- [ ] 脚本顶部明确提示：全参 5B DMD 需要多卡大显存，AutoDL 小卡只用于低分辨率 smoke。

## 5. 测试要求

- `dmd_full` 禁止 LoRA 参数。
- teacher 无 trainable params。
- student/fake_score trainable params 是 full DiT 参数，不是 LoRA adapter。
- BSA full-param backward 覆盖 attention 和 MLP 权重。
- student 参数只在 student update step 变化。
- fake_score 参数每步变化。
- final export 是 full student checkpoint，不是 LoRA safetensors。

## 6. 风险与资源要求

- 全参 DMD 同时持有 teacher、student、fake_score 三个 5B DiT，且 student/fake_score 有 optimizer state，资源成本远高于 LoRA。
- DeepSpeed ZeRO-2 可能不足以支撑实际 5B 全参 DMD；需要评估 ZeRO-3、FSDP、CPU offload 或更低分辨率 smoke。
- 全参模式的验收必须先从 256x256@81、极短 step 开始，不能直接上 832x480 长训。

## 7. 不在本方案内

- 不使用 LoRA 初始化、LoRA adapter 或 LoRA final export。
- 不保证普通 AutoDL 单卡能跑通 Wan2.2-5B 全参真实训练；单卡只作为小模型或低分辨率代码 smoke 环境。
- 不把 fake_score 作为最终推理产物。
