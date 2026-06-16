# BSA-DMD 真实模拟环境验收检测逻辑

目标：在 AutoDL 模拟内网环境中，用上传的 Wan2.2-5B 模型、可用的 teacher/mock LoRA 和 BridgeData2 smoke 数据，验证 DiffSynth-Studio_BSA_DMD 中 BSA-DMD 训练实现的代码逻辑正确，再开始长训。验收不以短训视觉质量作为唯一标准，而是先证明模型角色、DMD 梯度、BSA 开关、first-frame 约束、checkpoint 和指标统计都走在正确路径上。

## 0. 验收输入

- Teacher：Wan2.2-5B full attention，50 步推理；若有真实或 mock teacher LoRA，则 LoRA rank=32，路径沿用当前 `LORA_CHECKPOINT=models/step-66900.safetensors` 或显式传入的新路径。
- Student：Wan2.2-5B BSA，4 步 DMD，LoRA rank=32，最终只导出 student LoRA。
- Fake score：Wan2.2-5B full attention 或与 DMD 设计一致的 score 网络，LoRA rank=32，从 teacher LoRA 初始化，只用于训练/恢复，不进入最终导出。
- DMD 参考配置：`Wan2.2-TI2V-5B-Turbo/configs/self_forcing_wan22_dmd.yaml`，关键参数为 `denoising_step_list=[1000,750,500,250]`、`guidance_scale=6.0`、`dfake_gen_update_ratio=5`、`lr=5e-7`、`lr_critic=1e-7`、`denoising_loss_type=x0`。
- Smoke 数据：`nvidia/BridgeData2-Subset-Synthetic-Captions` 的 `sft_dataset_bridge/train`，训练 manifest 应为 1,222 条，视频为 256x256 MP4，每条有 caption。

## 0.1 无内网 LoRA 的 AutoDL 验收方案

如果内网 teacher LoRA 不能上传到 AutoDL，则 AutoDL 不能验证最终目标 LoRA 的真实蒸馏质量，但仍可做真实训练级别的代码验收。推荐两级替代：

- 基模 teacher smoke：teacher 使用不带 LoRA 的 Wan2.2-5B full attention；student/fake_score 仍注入 rank=32 LoRA，但从零初始化或空 LoRA 初始化。该方案验证 DMD、BSA、first-frame、checkpoint、metrics、分布式训练是否能真实跑通。
- mock teacher LoRA smoke：先在 AutoDL 上用 BridgeData2 跑一个很短的 full-attention SFT LoRA，得到 `mock_teacher_lora.safetensors`；再用它作为 teacher/fake/student 的初始化 LoRA 进行 BSA-DMD。该方案最接近最终路径，可以验证 LoRA 加载、从 teacher LoRA 初始化 fake_score/student、最终只导出 student LoRA 等关键语义。

验收边界：

- AutoDL 的基模或 mock LoRA 验收只能证明代码逻辑、数值稳定性和训练链路正确。
- 内网真实 teacher LoRA 的蒸馏质量、最终 loss 量级和视觉质量，必须在内网环境再跑同一套 Gate 0-4 才能确认。
- 当 fake_score 和 teacher 从完全相同的基模/LoRA 初始化时，最早的 generator update 出现接近 0 的 DMD 梯度是允许的；更关键的是 fake_score 经过若干 critic update 后，下一次 student update 的 DMD 梯度和 student 参数 delta 必须变为非零。

## 1. 环境与数据验收

通过标准：

- 模型目录完整：DiT、VAE、T5/text encoder、tokenizer/scheduler 所需文件都能加载；若本轮使用真实或 mock teacher LoRA，则 LoRA 文件存在且 safetensors 可读。
- 数据 manifest 可解析：每行至少能得到 `video_path`、`prompt/caption`、首帧图像；随机抽 10 条视频均可解码。
- 数据字段转换正确：BridgeData2 的 `vision_path` 和 `t2w_windows[0].caption` 能转换为 DiffSynth 训练需要的 `video`、`prompt`、`input_image/first_frame`。
- latent shape 正确：输入 256x256、81 帧时，Wan VAE latent 应为 `[B, 48, 21, 16, 16]`；first-frame latent 应为 `[B, 48, 1, 16, 16]`。
- token 统计正确：DiffSynth 当前公式为 `latent_frames=((frames-1)//4+1)`，`token_h=ceil(height/(16*2))`，`token_w=ceil(width/(16*2))`，`tokens=latent_frames*token_h*token_w`。因此 256x256@81 为 `21*8*8=1344 tokens/video`；832x480@81 为 `21*26*15=8190 tokens/video`。

失败即停止：模型加载失败、caption/video 数量不一致、随机样本不能解码、first-frame latent shape 不符合预期。

## 2. 静态与单元验收

通过标准：

- 参数入口完整：DMD 训练脚本能接收 teacher/fake/student checkpoint、LoRA rank、DMD 步表、CFG、fake/student 学习率、`dfake_gen_update_ratio`、BSA 参数、metrics 参数。
- 三个模型角色是独立对象：`teacher/real_score` 冻结；`student/generator` 只训练 LoRA；`fake_score` 只训练 LoRA；三个对象不能共享同一个 Python module 实例。
- BSA 只作用于 student：student attention 启用 `BSA_BLOCK_SIZE=3,7,3`、`BSA_SPARSE_RATIO=0.85` 或 warmup 值；teacher 和 fake_score 保持 full attention。
- LoRA 初始化正确：有 teacher/mock LoRA 时，teacher 加载并冻结该 LoRA，student 和 fake_score 从同一 LoRA 初始化；无 teacher LoRA 的基模 smoke 中，student/fake_score 使用空 LoRA 或零初始化，并在训练报告中标记为 base-teacher 验收。
- checkpoint 语义正确：中间恢复 checkpoint 保存 student LoRA、fake_score LoRA、两个 optimizer/scheduler 状态和 trainer state；最终导出目录只包含 student LoRA。
- flow/x0 公式正确：Wan flow prediction 转 x0 与 scheduler 的 `add_noise`、training target 互相一致；抽样张量 round-trip 误差应在 bf16/fp32 可解释范围内。
- first-frame 约束正确：student rollout、re-noise、teacher/fake score 输入、4-step inference 每一步之后，latent 第 0 帧都等于 VAE first-frame latent，误差应接近 0。

失败即停止：teacher 有梯度、fake/student 参数混在同一 optimizer、teacher/fake 被 BSA 改写、final export 含 fake_score 权重。

## 3. 单 batch 不更新调试

用固定 seed、单 GPU 或单进程、batch size=1，关闭 optimizer step，只跑 forward/backward 检查。强制 exit step 分别为 0、1、2、3。

每个 exit step 都必须满足：

- student DiT forward 次数等于 exit step 对应的 4-step rollout 深度；exit 前步骤在 no-grad 路径，exit 步保留梯度图。
- `denoised_timestep_from/to` 与 `[1000,750,500,250]` 区间匹配；DMD 采样 timestep 在合法范围内。
- `pred_fake_x0`、`pred_real_x0`、`dmd_grad=pred_fake_x0-pred_real_x0` shape 与 student latent 一致，值有限，无 NaN/Inf。
- DMD normalizer `abs(x0-pred_real_x0).mean()` 为正且不是极小值；若 fake_score 已经过 critic update，下一次 generator pass 的 `dmd_gradient_norm` 应大于 0。
- generator pass 后只有 student LoRA 有梯度；teacher 无梯度；fake_score 无梯度或未被 optimizer step。
- critic/fake pass 后只有 fake_score LoRA 有梯度；teacher 和 student 均无更新。

失败定位：

- `dmd_gradient_norm` 在 fake_score 多次更新后仍为 0：通常是 teacher/fake 使用了同一对象、CFG 没生效、fake_score 没被训练、或 DMD 输入错误。
- normalizer 接近 0：通常是 x0/flow/sigma 公式错，或误把 teacher prediction 当成 x0 本身。
- first-frame drift：通常是 re-noise 后没有重新替换首帧，或 Wan2.2 separated timestep/mask 没传对。

## 4. 优化器与 TTUR 验收

用 10 个 outer steps 进行真实 optimizer step，`dfake_gen_update_ratio=5`。

通过标准：

- outer step `0,5,10...` 更新 student；每个 outer step 都更新 fake_score。
- 参数 delta 统计满足：teacher delta 恒为 0；student delta 只在 student update step 非 0；fake_score delta 每步非 0。
- 两套 optimizer state 分开保存；student lr 为 `5e-7`，fake_score lr 为 `1e-7`，beta 与 DMD 源配置一致。
- gradient norm、loss、DMD gradient norm 均为有限值；梯度裁剪后没有长期为 0。
- 分布式/DeepSpeed 模式下没有未使用参数导致的 silent skip；LoRA trainable parameter count 与 rank=32 预期一致。

失败即停止：teacher 参数变化、fake_score 只在 student step 更新、student 每步都更新、两个 optimizer 共享同一参数组。

## 5. 短训验收流程

建议分三段，每段都固定 seed、保存 metrics、保存至少一个 checkpoint 和一组 inference 样例。

| 阶段 | 设置 | 目的 | 通过标准 |
| --- | --- | --- | --- |
| A | 5-10 steps，student BSA 关闭或 sparse=0 | 先隔离 DMD 逻辑 | 单步检查全部通过，loss/grad 有限，checkpoint 可恢复 |
| B | 20 steps，BSA sparse=0 或最小稀疏 | 验证 BSA 接入不破坏数值路径 | 与 A 相同 seed 的输出/shape/first-frame 约束一致；无 NaN/OOM |
| C | 50-100 steps，BSA sparse warmup 到 0.85 | 验证真实 BSA-DMD 训练路径 | 曲线有界，吞吐稳定，4-step BSA inference 不退化为黑屏/静态噪声 |

注意：BridgeData2 smoke 数据较小且分辨率 256x256，50-100 steps 不要求视觉质量显著提升，只要求训练路径、数值和指标合理。

## 6. Loss 曲线验收

DMD 训练的 loss 不应按普通 SFT 要求单调下降。正确曲线应满足以下特征：

- 所有 loss 均有限，无 NaN/Inf；连续窗口内没有持续 10 倍以上爆炸。
- `student_dmd_loss` 只在 student update step 记录或标记有效；曲线允许噪声大，但 rolling mean 应有界，不应在 fake_score 多次更新后仍恒为 0。
- `fake_score_loss/critic_loss` 每步记录；短训前期可波动，但 rolling mean 应逐步稳定或下降。student 更新步附近出现尖峰可以接受。
- `dmd_gradient_norm` 在 fake_score 更新后应大于 0 且有稳定量级；长期趋近 0 表示 DMD 梯度没有传入 student；长期爆炸表示 timestep/sigma/x0 公式或 normalizer 有问题。
- `teacher_real_score_norm` 与 `fake_score_norm` 不应完全一致；若二者从头到尾完全相同，优先排查 teacher/fake 是否共享模块或 fake_score 未更新。
- `student_grad_norm` 只在 student update step 非 0；`fake_score_grad_norm` 每步非 0；teacher grad norm 必须为 0 或不存在。

必须新增或确认的 metrics 字段：

- `outer_step`
- `student_update`
- `fake_score_update`
- `student_dmd_loss`
- `fake_score_loss`
- `dmd_gradient_norm`
- `student_grad_norm`
- `fake_score_grad_norm`
- `teacher_grad_norm`
- `lr_student`
- `lr_fake_score`
- `denoised_timestep_from`
- `denoised_timestep_to`
- `sampled_score_timestep`
- `bsa_sparse_ratio`

## 7. Token 吞吐曲线验收

需要区分两个概念：

- `tokens_per_sample`：按视频 latent token 数计算，用于跨实验对齐，必须等于公式结果。
- `effective_tokens_per_second`：按 DMD 实际 DiT forward 次数折算，用于反映训练成本。DMD 每个 outer step 包含 student rollout、fake_score forward、teacher conditional/unconditional forward，不能只按一次 SFT forward 计。

通过标准：

- 256x256@81 的 `tokens_per_sample` 必须为 1344；若使用 832x480@81，必须为 8190。
- `tokens_this_step=samples_this_step*tokens_per_sample`，累计 token 与样本数一致。
- 记录 `dit_forward_count_student`、`dit_forward_count_fake`、`dit_forward_count_teacher`，并用它们计算 `effective_tokens_this_step`。
- 吞吐曲线前几步允许因为编译/cache/显存预热较低，之后 step time 应进入稳定区间；BSA sparse warmup 改变时允许阶跃变化。
- 在 256x256 smoke 数据上，不强制要求 BSA 比 full attention 更快，因为小分辨率下 kernel/调度开销可能占主导；在 832x480 以上，BSA 应至少降低显存峰值，并通常改善或维持吞吐。
- `bsa_sparse_ratio` 曲线必须与配置一致：固定 0.85 时恒定；warmup 时按 step 单调上升到 0.85。

失败定位：

- token 数不是 1344/8190：检查 height、width、frames、patch size、VAE factor 是否传错。
- throughput 每步随机大幅抖动：检查 dataloader 解码阻塞、metrics 同步、checkpoint/inference 是否混入训练计时。
- BSA sparse ratio 已上升但显存/forward count 没变化：检查 BSA 是否真的只配置到 student DiT。

## 8. Checkpoint 与恢复验收

短训至少执行一次保存、恢复、继续训练。

通过标准：

- 中间 checkpoint 包含：`student_lora.safetensors`、`fake_score_lora.safetensors`、student optimizer、fake optimizer、scheduler、global step、random state、BSA 配置、DMD 配置。
- 从 checkpoint 恢复后，同 seed 下一步的 loss、timestep、参数 delta 与未中断训练基本一致。
- final export 只包含 student LoRA；加载到 BSA 4-step inference 脚本时不依赖 fake_score 权重。
- teacher LoRA 和 fake_score LoRA 不被误写入 final student LoRA 文件。

失败即停止：恢复后 step 重复/跳步、fake_score 丢失导致曲线重置、final LoRA key 带 fake_score 前缀。

## 9. Inference 验收

对固定 3-5 个 validation prompt/image，保存训练前、短训后、恢复后继续训练后的结果。

通过标准：

- 4-step 推理步表为 `[1000,750,500,250]`；student BSA 参数与训练一致。
- 输出视频不是全黑、全白、静态纯噪声；首帧视觉上与输入图一致，latent 首帧误差接近 0。
- 同 seed、同 checkpoint 的推理结果可复现。
- 与 teacher 50-step full attention 结果相比，短训阶段不要求质量接近，但运动方向、场景结构不应完全脱离输入首帧和 prompt。
- 加载 final student LoRA 时不需要 fake_score/teacher 训练状态。

## 10. 最终验收门槛

只有下面五关全部通过，才进入正式长训：

- Gate 0：环境和数据通过，随机样本可解码，token 公式正确。
- Gate 1：静态/单元检查通过，三模型角色、LoRA、BSA、first-frame 约束无误。
- Gate 2：单 batch 不更新调试通过，DMD 梯度、timestep、grad ownership 正确。
- Gate 3：10-20 step 真实短训通过，TTUR、optimizer、checkpoint、metrics 正确。
- Gate 4：50-100 step BSA-DMD smoke 通过，loss 有界、吞吐稳定、4-step inference 非退化。

若任何 gate 失败，先修复该层问题，不进入下一层。这样可以避免把 scheduler、LoRA、BSA、checkpoint 或 metrics 的逻辑错误误判为“训练还没收敛”。

## 参考来源

- 本地迁移清单：`DMD迁移到DiffSynth-Studio_BSA_DMD修改清单.md`
- 源 DMD 实现：`Wan2.2-TI2V-5B-Turbo/model/dmd.py`、`trainer/wan22_distillation.py`、`pipeline/bidirectional_training.py`、`utils/wan_wrapper.py`
- 目标 BSA/训练实现：`DiffSynth-Studio_BSA_DMD/examples/wanvideo/model_training/train.py`、`diffsynth/models/wan_bsa.py`、`diffsynth/diffusion/training_metrics.py`、`diffsynth/diffusion/flow_match.py`
- Smoke 数据：`nvidia/BridgeData2-Subset-Synthetic-Captions`，Hugging Face dataset card 与 `sft_dataset_bridge/train` 文件树
