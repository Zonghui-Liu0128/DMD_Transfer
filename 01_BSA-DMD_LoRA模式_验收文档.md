# BSA-DMD LoRA 模式验收文档

目标：验证 LoRA 模式 BSA-DMD 实现能在真实训练链路中正确运行。该验收关注三模型 LoRA 语义、DMD 梯度、BSA 作用范围、checkpoint/final export 和指标曲线；短训不以最终视觉质量为唯一标准。

## 1. 验收环境

- Wan2.2-5B base 模型可加载。
- teacher LoRA 可以是真实内网 LoRA、AutoDL mock teacher LoRA，或空缺时使用 base teacher smoke。
- BridgeData2 smoke 数据可解码，manifest 为 1,222 条。
- 默认分辨率建议先用 256x256@81；token 数应为 1344/video。

## 2. 静态检查

必须通过：

- task `dmd_lora` / `dmd_lora:train` 可启动。
- teacher、student、fake_score 是三个独立 DiT 对象。
- teacher 全部 `requires_grad=False`。
- student 只有 LoRA 参数可训练。
- fake_score 只有 LoRA 参数可训练。
- BSA 只配置到 student；teacher/fake_score 保持 full attention。
- 有 teacher/mock LoRA 时，student/fake_score 初始 LoRA key 与 teacher LoRA key 对齐。
- 无 teacher LoRA 时，训练日志明确标记 `base_teacher_smoke=true`。

失败即停止：teacher 有梯度、BSA 作用到 teacher/fake_score、student/fake_score 训练了 base 权重。

## 3. 单 batch 调试

固定 seed，batch size=1，强制 exit step 为 0、1、2、3，分别执行 generator pass 和 critic pass。

通过标准：

- student rollout 的 timestep 为 `[1000,750,500,250]`。
- 每次 rollout/re-noise 后首帧 latent 与 first-frame latent 误差接近 0。
- teacher/fake/student 输出 shape 一致，无 NaN/Inf。
- fake_score 更新前，若 teacher/fake/student 从同一 LoRA 初始化，DMD gradient 允许接近 0。
- fake_score 经多步 critic update 后，下一次 student update 的 `dmd_gradient_norm > 0`。
- generator pass 只产生 student LoRA grad。
- critic pass 只产生 fake_score LoRA grad。

## 4. TTUR 与参数更新

跑 10 个 outer steps，`fake_score_updates_per_generator_update=5`。

通过标准：

- fake_score LoRA 每步有参数 delta。
- student LoRA 只在 step 0、5、10... 有参数 delta。
- teacher 参数 delta 恒为 0。
- 两个 optimizer state 分开保存。
- student lr 默认 `5e-7`，fake_score lr 默认 `1e-7`。

失败即停止：student 每步更新、fake_score 不更新、teacher 发生参数变化。

## 5. Loss 曲线

短训 50-100 steps。

合理表现：

- `student_dmd_loss` 只在 student update step 有效。
- `fake_score_loss` 每步记录，允许波动，但 rolling mean 不应持续爆炸。
- `dmd_gradient_norm` 在 fake_score 更新后非零且有界。
- `student_grad_norm` 只在 student update step 非零。
- `fake_score_grad_norm` 每步非零。
- `teacher_grad_norm` 为 0 或不存在。

异常定位：

- `dmd_gradient_norm` 长期为 0：检查 fake_score 是否真的更新、teacher/fake 是否共享同一对象。
- `fake_score_loss` 从头为 0：检查 target 泄漏或 fake_score 输出与 target 直接同源。
- loss 爆炸：检查 flow->x0、sigma shift、timestep 区间、normalizer。

## 6. 吞吐与指标

必须记录：

- `tokens_per_sample`
- `tokens_this_step`
- `effective_tokens_this_step`
- `dit_forward_count_student`
- `dit_forward_count_fake`
- `dit_forward_count_teacher`
- `tokens_per_second`
- `effective_tokens_per_second`
- `bsa_sparse_ratio`
- `peak_memory_gb`

通过标准：

- 256x256@81 时 `tokens_per_sample=1344`。
- `bsa_sparse_ratio` 与 warmup/固定配置一致。
- 前几步可慢，稳定后 step time 不应持续漂移。
- 256x256 上不强制 BSA 提速；832x480 以上应观察显存下降或吞吐不劣化。

## 7. Checkpoint 与恢复

通过标准：

- 中间 checkpoint 包含 student LoRA、fake_score LoRA、两个 optimizer、trainer state。
- resume 后下一步 loss/timestep/参数 delta 与未中断训练基本一致。
- final export 只包含 student LoRA。
- final student LoRA 可被 BSA 4-step inference 加载。

失败即停止：final export 含 fake_score key，resume 后 fake_score 重新初始化，global step 错乱。

## 8. Inference smoke

使用固定 3-5 个 validation prompt/image。

通过标准：

- 4-step 推理使用 `[1000,750,500,250]`。
- student BSA 参数与训练一致。
- 输出非全黑、非全白、非纯噪声。
- 首帧视觉和 latent 均保持输入首帧。
- 加载 final student LoRA 不依赖 teacher/fake_score。

## 9. AutoDL 无内网 LoRA 的边界

- base teacher smoke 只能证明代码链路正确。
- mock teacher LoRA smoke 能验证 LoRA 初始化和导出语义。
- 真实内网 teacher LoRA 的最终质量必须回内网再跑同一套验收。

## 10. 通过门槛

- Gate 0：数据/模型/token 通过。
- Gate 1：三模型 LoRA 静态检查通过。
- Gate 2：单 batch DMD 梯度和 first-frame 约束通过。
- Gate 3：10-step TTUR 参数 delta 正确。
- Gate 4：50-100 step smoke 曲线有界，checkpoint/resume/final export 正确。
