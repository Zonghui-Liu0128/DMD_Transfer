# Wan2.2-5B BSA DMD 迁移修改清单

目标：在 `DiffSynth-Studio_BSA_DMD` 中实现 Wan2.2-TI2V-5B 的 DMD 训练，把 **Wan2.2-5B full attention 50 step + LoRA(rank=32)** 作为冻结 teacher，蒸馏得到 **Wan2.2-5B BSA DMD 4 step + LoRA(rank=32)** student。

本文只列迁移范围和源码对应关系，不涉及代码实现。

## 现有基础

- BSA 已在目标框架中实现：`diffsynth/models/wan_video_dit.py::BlockSparseAttention`，`diffsynth/models/wan_bsa.py::configure_wan_bsa`。
- Wan2.2-TI2V-5B 模型配置已存在：`diffsynth/configs/model_configs.py` 中 `seperated_timestep=True`、`fuse_vae_embedding_in_latents=True`。
- 当前目标训练入口是单模型 SFT：`examples/wanvideo/model_training/train.py::WanTrainingModule` + `diffsynth/diffusion/runner.py::launch_training_task`。
- DMD 需要三模型结构和双 optimizer，不能只在 `FlowMatchSFTLoss` 中加一个 loss。

## 需要新增或修改的文件

| 目标文件 | 类型 | 需要做的事 | 对应 Turbo 源实现 |
| --- | --- | --- | --- |
| `examples/wanvideo/model_training/train.py` | 修改 | 增加 `dmd` / `dmd:train` task；接入 DMD 参数；实例化 DMD 专用 training module；launcher 映射到 DMD runner。 | `Wan2.2-TI2V-5B-Turbo/train.py::main`，`trainer/wan22_distillation.py::Trainer` |
| `examples/wanvideo/model_training/wan_dmd_training.py` | 新增 | 新建 `WanDMDTrainingModule`：加载 teacher / fake_score / student 三个 DiT；teacher 冻结 full attention；fake_score 可训练 full attention；student 可训练 BSA；三者共享 tokenizer / VAE / scheduler / pipeline units。 | `model/base.py::BaseModel._initialize_models`，`trainer/wan22_distillation.py::__init__` |
| `diffsynth/diffusion/dmd_loss.py` | 新增 | 实现 DMD 核心：4-step student rollout、random exit、DMD KL gradient、stop-gradient MSE、fake_score x0 denoising loss、flow->x0 转换。 | `model/dmd.py::generator_loss`、`compute_distribution_matching_loss`、`_compute_kl_grad`、`critic_loss`；`utils/wan_wrapper.py::_convert_flow_pred_to_x0` |
| `diffsynth/diffusion/dmd_runner.py` | 新增 | 实现 DMD 训练循环：student optimizer 和 fake_score optimizer 分开；TTUR 更新比例；Accelerate/DeepSpeed prepare；日志和保存。 | `trainer/wan22_distillation.py::train`、`fwdbwd_one_step`、`save`、`load` |
| `diffsynth/diffusion/logger.py` | 修改或扩展 | 支持 DMD checkpoint：分别保存 student LoRA、fake_score LoRA、必要时保存 optimizer state / global step；最终导出只保留 student LoRA。 | `trainer/wan22_distillation.py::save/load`，目标现有 `ModelLogger.save_model` |
| `diffsynth/diffusion/parsers.py` | 修改 | 增加 DMD 参数：teacher/fake/student 模型路径、teacher LoRA、student LoRA rank、fake_score LoRA rank、`denoising_step_list`、`real_guidance_scale`、`fake_guidance_scale`、`fake_score_updates_per_generator_update`、student/fake lr/betas。 | `configs/self_forcing_wan22_dmd.yaml` |
| `diffsynth/diffusion/__init__.py` | 修改 | 导出 `DMDLoss` / `launch_dmd_training_task`，供训练入口引用。 | 目标框架现有 `loss.py`、`runner.py` 导出模式 |
| `diffsynth/pipelines/wan_video.py` | 尽量少改 | 复用 `model_fn_wan_video`；如必要，抽出 CFG 调用、first-frame latent 替换、4-step re-noise helper，避免 DMD loss 内重复 pipeline 私有逻辑。 | `pipeline/bidirectional_training.py::inference_with_trajectory`，`pipeline/wan22_fewstep_inference.py::inference` |
| `diffsynth/models/wan_bsa.py` | 可能不改 | 现有 `configure_wan_bsa(model, ...)` 可直接用于 student；DMD module 需要保证只对 student 开 BSA，不影响 teacher/fake_score。 | Turbo 无对应；这是目标框架已有 BSA 能力 |
| `diffsynth/models/wan_video_dit.py` | 可能不改 | BSA attention 已实现；仅在 DMD rollout 发现 video_shape 或 separated timestep 与 BSA 不兼容时再修。 | Turbo 无对应；替代 Turbo 中未支持的 bidirectional BSA student |
| `train_Wan2.2_5B_LoRA.sh` 或新增 `train_Wan2.2_5B_BSA_DMD_LoRA.sh` | 新增/修改 | 提供正式 DMD 启动脚本：teacher full-attn LoRA、student BSA 参数、4 step `[1000,750,500,250]`、TTUR=5、rank=32、负面提示、数据路径。 | `running_scripts/train/Wan2.2/dmd.sh` |
| `tests/test_wan_dmd_training.py` | 新增 | 覆盖 parser、flow->x0、random exit rollout、TTUR 调度、BSA 只作用 student、checkpoint key 区分。 | 源端无测试；基于目标现有 `tests/test_bsa_training_tools.py` 风格补齐 |

## 关键迁移点

### 1. 三模型结构

Turbo 源实现：

- `model/base.py::BaseModel._initialize_models`
- `trainer/wan22_distillation.py::__init__`

目标实现建议：

- `teacher_dit`：full attention，加载 base + teacher LoRA，`requires_grad_(False)`。
- `fake_score_dit`：full attention，可训练，建议 LoRA(rank=32)，用于拟合 student 当前 fake distribution。
- `student_dit`：BSA attention，可训练，LoRA(rank=32)，作为最终导出的模型。

### 2. 4-step rollout / random exit

Turbo 源实现：

- `pipeline/bidirectional_training.py::BidirectionalTrainingPipeline.inference_with_trajectory`
- `model/base.py::_run_generator`
- `pipeline/wan22_fewstep_inference.py::Wan22FewstepInferencePipeline.inference`

目标实现建议：

- 在 `dmd_loss.py` 中实现 student training-time rollout。
- timestep 固定为 `[1000, 750, 500, 250]`，按 DiffSynth `FlowMatchScheduler("Wan")` 映射到实际 sigma。
- 非 exit step 用 `torch.no_grad()`；exit step 保留 student 梯度。
- 每步后用 `scheduler.add_noise(pred_x0, fresh_noise, next_timestep)` 模拟 4-step 推理链路。

### 3. DMD generator loss

Turbo 源实现：

- `model/dmd.py::_compute_kl_grad`
- `model/dmd.py::compute_distribution_matching_loss`
- `model/dmd.py::generator_loss`
- `utils/wan_wrapper.py::_convert_flow_pred_to_x0`

目标实现建议：

- 对 rollout 得到的 student `pred_x0` 重新加噪。
- teacher 与 fake_score 在同一个 noisy latent 上预测 flow，再转成 x0。
- teacher 使用 CFG：`pred_cond + (pred_cond - pred_uncond) * real_guidance_scale`，默认 6.0。
- fake_score 默认 conditional；保留 `fake_guidance_scale` 参数。
- DMD 梯度：`pred_fake_x0 - pred_real_x0`，按 `abs(x0 - pred_real_x0).mean()` 归一化。
- 用 stop-gradient MSE 把手工梯度传给 student。

### 4. fake_score / critic loss

Turbo 源实现：

- `model/dmd.py::critic_loss`
- `utils/loss.py::X0PredLoss`

目标实现建议：

- 用当前 student 在 `torch.no_grad()` 下 rollout 生成 fake sample。
- 对 fake sample 重新加噪。
- fake_score 预测 x0，目标是 student 生成的 fake sample。
- 默认只实现 `denoising_loss_type=x0`，先不迁移 `flow/noise` 变体，除非后续需要。

### 5. Wan2.2 TI2V first-frame 逻辑

Turbo 源实现：

- `trainer/wan22_distillation.py::fwdbwd_one_step`
- `model/dmd.py::_compute_kl_grad`
- `model/dmd.py::critic_loss`
- `utils/dataset.py::masks_like`

目标实现建议：

- 复用目标框架 `WanVideoUnit_ImageEmbedderFused` 产出的 `first_frame_latents`。
- DMD rollout、DMD 加噪、fake_score loss 都必须把 latent 第 0 帧替换回 `first_frame_latents`。
- separated timestep 中首帧应保持 0，后续 latent token 使用当前 timestep；可复用 `model_fn_wan_video` 的 `fuse_vae_embedding_in_latents` 分支。

### 6. TTUR 双优化器

Turbo 源实现：

- `trainer/wan22_distillation.py::__init__`
- `trainer/wan22_distillation.py::train`

目标实现建议：

- `optimizer_student`：只更新 student LoRA。
- `optimizer_fake_score`：只更新 fake_score LoRA。
- `teacher_dit` 不进 optimizer。
- `fake_score_updates_per_generator_update=5`：fake_score 每步更新，student 每 5 个 outer step 更新一次。
- 日志分开记录 `student_dmd_loss`、`fake_score_loss`、`student_grad_norm`、`fake_score_grad_norm`、`dmd_gradient_norm`、`bsa_sparse_ratio`。

## 配置参数映射

| DMD 参数 | Turbo 配置 | DiffSynth 建议参数 |
| --- | --- | --- |
| 4-step timesteps | `denoising_step_list: [1000,750,500,250]` | `--dmd_denoising_steps 1000,750,500,250` |
| timestep warp / sigma shift | `warp_denoising_step: true`，`timestep_shift: 5.0` | 使用 `FlowMatchScheduler("Wan")`，`--sigma_shift 5.0` |
| teacher CFG | `guidance_scale: 6.0` | `--real_guidance_scale 6.0` |
| fake CFG | 默认 `0.0` | `--fake_guidance_scale 0.0` |
| fake/student 更新比 | `dfake_gen_update_ratio: 5` | `--fake_score_updates_per_generator_update 5` |
| student lr | `lr: 5e-7` | `--student_learning_rate 5e-7` |
| fake_score lr | `lr_critic: 1e-7` | `--fake_score_learning_rate 1e-7` |
| denoising target | `denoising_loss_type: x0` | `--fake_score_loss_type x0` |
| student BSA | Turbo 不支持 | 复用 `--bsa_enable --bsa_block_size --bsa_sparse_ratio` |
| LoRA rank | 目标需求 rank=32 | `--student_lora_rank 32 --fake_score_lora_rank 32` |

## 暂不建议改动

- `diffsynth/models/wan_video_dit.py`：BSA attention 已存在，先作为稳定依赖使用。
- `diffsynth/models/wan_bsa.py`：已有通用配置函数，DMD module 只需正确选择 student。
- `config_wan22_5B.yaml`：这是 Accelerate/DeepSpeed 配置，不放 DMD 超参。

## 需要你确认的点

1. teacher LoRA checkpoint 路径是否就是当前 `LORA_CHECKPOINT=models/step-66900.safetensors`，还是另有 full-attn 50-step teacher LoRA。-- 是
2. fake_score 是否也采用 LoRA(rank=32) 训练并从 teacher LoRA 初始化；我建议是，否则全量训练 fake_score 显存和保存成本会明显更高。--是
3. 最终 checkpoint 是否只导出 student LoRA；我建议 fake_score 只作为 resume checkpoint 保存，不作为最终推理产物。--只导出 student LoRA
