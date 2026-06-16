# BSA-DMD 基模全参数模式验收文档

目标：验证不使用 LoRA、直接更新 Wan2.2-5B student/fake_score 全参数的 BSA-DMD 训练路径是否正确。该验收重点是 full-param trainable 范围、BSA 梯度可导、资源/显存、full checkpoint 和恢复。

## 1. 验收环境

- Wan2.2-5B base 或 teacher full checkpoint 可加载。
- 无 LoRA 参数、无 LoRA checkpoint、无 LoRA final export。
- 建议从 256x256@81、batch size=1、10-20 steps 开始。
- 必须记录 GPU 型号、GPU 数、显存、DeepSpeed/FSDP/offload 配置。

## 2. 资源预检

必须通过：

- teacher、student、fake_score 三个 DiT 可独立实例化。
- teacher 冻结后不分配 optimizer state。
- student/fake_score optimizer state 可创建；若 OOM，不能进入训练验收。
- 开启 gradient checkpointing。
- 记录 `peak_memory_gb`、`optimizer_state_memory_gb`、`activation_checkpointing=true/false`。

失败即停止：optimizer 初始化 OOM、DeepSpeed/Accelerate 不支持当前双 optimizer/full-param 配置、模型只能通过 LoRA 路径启动。

## 3. 静态检查

必须通过：

- task `dmd_full` / `dmd_full:train` 可启动。
- 传入任何 LoRA 参数时直接报错。
- teacher trainable param count = 0。
- student trainable param count 接近完整 Wan2.2-5B DiT。
- fake_score trainable param count 接近完整 Wan2.2-5B DiT。
- BSA 只配置到 student。
- teacher/fake_score 保持 full attention。
- final export 配置声明 `export_type=full_student_dit`。

失败即停止：student/fake_score 只训练 LoRA、teacher 有梯度、BSA 作用到 teacher/fake_score。

## 4. BSA 全参数梯度验收

在小模型或单 block 上先跑单元测试。

通过标准：

- `sparse_ratio=0.0` 时 BSA 输出与 dense attention 输出误差在可解释范围内。
- `sparse_ratio=0.0` backward 后 q/k/v/o/ffn 参数均有非空梯度。
- `sparse_ratio=0.85` backward 后梯度有限，无 NaN/Inf。
- BSA mask/index 可以 no-grad，但 value path 不能 detach。

失败即停止：BSA 模式下 attention 权重没有梯度、梯度全 0、或 sparse 后出现 NaN。

## 5. 单 batch DMD 调试

固定 seed，batch size=1，强制 exit step 为 0、1、2、3。

通过标准：

- student rollout timestep 为 `[1000,750,500,250]`。
- first-frame latent 在 rollout/re-noise/score 输入中保持不变。
- teacher/fake/student 输出 shape 一致，无 NaN/Inf。
- generator pass 后 student 的 q/k/v/o/ffn 参数有梯度。
- critic pass 后 fake_score 的 q/k/v/o/ffn 参数有梯度。
- teacher 参数无梯度。
- fake_score 多步更新后，下一次 student update 的 DMD gradient 非零。

## 6. TTUR 与参数 delta

跑 10 个 outer steps。

通过标准：

- fake_score 全参数每步有 delta。
- student 全参数只在 step 0、5、10... 有 delta。
- teacher delta 恒为 0。
- 若使用单 optimizer fallback，inactive param group 的参数 delta 必须为 0。
- grad clipping 后 grad norm 有限，不长期为 0。

失败即停止：student 每步更新、fake_score 不更新、teacher 变化、inactive group 被 optimizer 更新。

## 7. Loss 与数值曲线

短训 20-100 steps。

合理表现：

- `student_dmd_loss` 只在 student update step 有效。
- `fake_score_loss` 每步有效。
- `dmd_gradient_norm` 在 fake_score 更新后非零且有界。
- full-param grad norm 比 LoRA 模式更大，必须记录 clipping 前后数值。
- 无 NaN/Inf，无持续 10 倍以上爆炸。

异常定位：

- full-param loss 立刻爆炸：检查学习率是否仍使用 LoRA 量级以外的值、normalizer、bf16 overflow。
- grad norm 长期为 0：检查 full-param requires_grad、BSA detach、optimizer param group。
- fake_score_loss 不变：检查 fake_score optimizer/offload 状态是否正确。

## 8. 吞吐与显存

必须记录：

- `tokens_per_sample`
- `effective_tokens_per_second`
- `dit_forward_count_student/fake/teacher`
- `step_seconds`
- `peak_memory_gb`
- `optimizer_state_memory_gb`
- `activation_checkpointing`
- `optimizer_offload`
- `bsa_sparse_ratio`

通过标准：

- 256x256@81 时 `tokens_per_sample=1344`。
- 吞吐曲线前几步预热后稳定。
- full-param step time 明显高于 LoRA 属于正常现象。
- BSA sparse ratio 与配置一致。
- 记录的显存峰值能解释是否可扩到 832x480。

失败即停止：metrics 未记录显存/optimizer state、token 公式错误、BSA sparse ratio 不生效。

## 9. Checkpoint 与恢复

通过标准：

- 中间 checkpoint 包含 full student DiT、full fake_score DiT、optimizer state、trainer state、BSA 配置。
- checkpoint 可以从中断 step 恢复。
- 恢复后下一步 loss/timestep/参数 delta 与未中断训练基本一致。
- final export 只包含 full student DiT 和 BSA 推理配置。
- final export 不包含 fake_score、不包含 optimizer state、不包含 LoRA key。

失败即停止：final 仍是 LoRA 文件、fake_score 被导出为推理产物、resume 后 optimizer state 丢失。

## 10. Inference smoke

通过标准：

- 4-step BSA inference 能加载 full student checkpoint。
- 不需要 LoRA loader。
- 输出非全黑、非全白、非纯噪声。
- first-frame 视觉和 latent 均保持。
- 同 seed 同 checkpoint 可复现。

## 11. 通过门槛

- Gate 0：资源预检通过，full-param optimizer 可创建。
- Gate 1：无 LoRA 静态检查通过。
- Gate 2：BSA full-param backward 单元测试通过。
- Gate 3：单 batch DMD 梯度和 first-frame 约束通过。
- Gate 4：10-step TTUR 参数 delta 正确。
- Gate 5：20-100 step smoke 曲线有界，full checkpoint/resume/final export 正确。

## 12. 失败后处理

- 如果资源预检失败，不应降级为 LoRA 并宣称 full-param 验收通过。
- 如果 BSA backward 失败，先修 BSA 可导性，再调 DMD。
- 如果 checkpoint 过大或保存慢，改 sharded checkpoint，不改为 LoRA export。
