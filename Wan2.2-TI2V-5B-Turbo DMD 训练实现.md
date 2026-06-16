# Wan2.2-TI2V-5B-Turbo DMD 训练实现

本文档分析Wan2.2-TI2V-5B-Turbo框架中的 DMD 训练实现，并给出将这套训练框架迁移到 DiffSynth-Studio 的适配清单。

## 1. 结论摘要

当前仓库已经支持一个==典型的三模型 DMD 训练结构==：

- `real_score`：冻结的真实分布 teacher (双向full attention, 多步模型)，当前实现里总是非因果 / 双向 full attention。
- `fake_score`：可训练的 fake 分布 score 模型，当前实现里总是非因果 / 双向 full attention。
- `generator`：可训练的 student / generator，可以是 causal，也可以是 bidirectional。Wan2.2 DMD 配置使用的是 `bidirectional`。

当前 Wan2.2 DMD 配置已经是 4 步目标：

```yaml
denoising_step_list: [1000, 750, 500, 250]
```

当前框架==支持 Two Time-scale Update Rule (TTUR)==。现有 `dfake_gen_update_ratio: 5` 的实际含义是：fake_score 每个 outer step 都更新，generator 每 5 个 outer step 才更新一次。

当前仓库不支持 “双向 Block Sparse Attention 的 Wan student/generator”。当前 `WanDiffusionWrapper` 只有 causal / bidirectional 两个分支, 我们关注bidirectional分支；bidirectional Wan2.2 路径调用的是 full attention `Wan22Model`，block/local attention 只存在于 causal 模型路径中，且不是双向 full-context block sparse。

迁移到 DiffSynth-Studio 时，主要工作不是简单迁移 loss，而是要补齐以下能力：

- 三个 Wan DiT 角色的加载、冻结、训练和 checkpoint 管理。
- DMD loss 管理:  generator loss 和 fake_score denoising loss等
- 4-step backward simulation / random exit 轨迹生成。
- real_score CFG、fake_score 可选 CFG、flow prediction 到 x0 prediction 的转换。
- 双 optimizer / 多 optimizer 训练循环，并显式支持 `fake_score_updates_per_generator_update = 5` 这类 TTUR 比例。
- training-time rollout / backward simulation，与推理时 few-step re-noise 链路保持一致。
- Wan2.2 TI2V first-frame latent 融合和 timestep/mask 逻辑。
- 双向 Block Sparse Attention student 的模型结构或 attention kernel 适配 (此前代码已实现, 基于Wan2.2-5B with LoRA)。

## 2. 当前仓库 DMD 相关文件与职责

| 文件 | 关键函数 / 类 | 职责 |
| --- | --- | --- |
| `running_scripts/train/Wan2.2/dmd.sh` | shell 入口 | 用 `torchrun` 启动 DMD 训练，指定 `configs/self_forcing_wan22_dmd.yaml`。 |
| `configs/self_forcing_wan22_dmd.yaml` | DMD 训练配置 | 指定三模型名称、4 步 timestep、Wan2.2 TI2V 形状、trainer、loss、学习率、更新比例等。 |
| `configs/default_config.yaml` | 默认训练配置 | 提供 `backward_simulation`、`same_step_across_blocks`、`last_step_only`、`context_noise` 等默认值。 |
| `train.py` | `main` | 加载配置；根据 `config.trainer` 选择 `Wan22ScoreDistillationTrainer`。 |
| `trainer/wan22_distillation.py` | `Wan22ScoreDistillationTrainer` | Wan2.2 DMD 的分布式训练主循环、FSDP 包装、数据加载、双 optimizer、checkpoint 保存。 |
| `model/base.py` | `BaseModel` | 初始化 `generator`、`real_score`、`fake_score`、VAE、text encoder、inference pipeline。 |
| `model/dmd.py` | `DMD` | 实现 DMD 的 generator loss、fake_score / critic loss、KL gradient 估计。 |
| `pipeline/bidirectional_training.py` | `BidirectionalTrainingPipeline.inference_with_trajectory` | bidirectional generator 的多步 backward simulation / random exit 训练轨迹。 |
| `utils/wan_wrapper.py` | `WanDiffusionWrapper` | 统一封装 Wan / Wan2.2 / CausalWan 的 forward、scheduler、flow-to-x0 转换。 |
| `utils/loss.py` | `X0PredLoss`、`FlowPredLoss`、`NoisePredLoss` | fake_score denoising loss 的不同预测目标。 |
| `utils/dataset.py` | `ODERegressionCSVDataset`、`masks_like` | 读取视频 CSV 数据；构造 Wan2.2 first-frame mask。 |

### 2.1 当前训练入口

Wan2.2 DMD 的启动脚本是：

```bash
bash running_scripts/train/Wan2.2/dmd.sh
```

脚本核心命令：

```bash
torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    train.py \
    --config_path configs/self_forcing_wan22_dmd.yaml \
    --logdir $LOGDIR \
    --data_path data/MagicData.csv \
    --no_visualize \
    --disable-wandb
```

`train.py` 根据配置选择 trainer：

```python
elif config.trainer == "score_distillation_wan22":
    from trainer.wan22_distillation import Wan22ScoreDistillationTrainer
    trainer = Wan22ScoreDistillationTrainer(config)
```

因此 Wan2.2 DMD 主训练逻辑全部在 `trainer/wan22_distillation.py` 和 `model/dmd.py`。

### 2.2 三模型如何初始化

三模型在 `model/base.py` 的 `BaseModel._initialize_models` 中初始化：

```python
self.generator = WanDiffusionWrapper(
    model_name=args.generator_name,
    is_causal=args.generator_type == "causal",
    ...
)
self.real_score = WanDiffusionWrapper(
    model_name=args.real_score_name,
    is_causal=False,
    ...
).requires_grad_(False)
self.fake_score = WanDiffusionWrapper(
    model_name=args.fake_score_name,
    is_causal=False,
    ...
).requires_grad_(True)
```

对应配置在 `configs/self_forcing_wan22_dmd.yaml`：

```yaml
generator_name: Wan2.2-TI2V-5B
real_score_name: Wan2.2-TI2V-5B
fake_score_name: Wan2.2-TI2V-5B
generator_type: bidirectional
```

含义：

- `real_score` 和 `fake_score` 固定使用 `is_causal=False`，因此是双向 full attention Wan2.2。
- `generator_type: bidirectional` 时，`generator` 也走非因果 / 双向 full attention Wan2.2。
- 三者名称相同，因此都会从同一个 pretrained 权重源加载。这个设计可以让 fake_score “隐式” 从 real teacher 同源初始化。

### 2.3 WanDiffusionWrapper 如何区分 full attention / causal

`utils/wan_wrapper.py` 中：

```python
if is_causal:
    self.model = CausalWanModel.from_pretrained(...)
elif "2.2" in model_name:
    self.model = Wan22Model.from_pretrained(...)
else:
    self.model = WanModel.from_pretrained(...)
```

这意味着：

- causal generator 使用 `CausalWanModel`，支持 `local_attn_size`、`sink_size`。
- bidirectional Wan2.2 使用 `Wan22Model`，当前是 full attention。
- 现有代码没有 `attention_type=block_sparse` 或类似配置。

所以当前仓库不能直接表达 “双向 Block Sparse Attention 的 Wan generator”。

## 3. DMD 原理：对照代码的精简说明

DMD 的目标是让 student/generator 生成的样本分布靠近 frozen real_score teacher 所代表的真实数据分布。当前实现不是直接对 teacher 输出做普通 MSE，而是用 real_score 与 fake_score 的 score / x0 预测差值构造分布匹配梯度。

当前代码里三个角色分别是：

- `generator`：产生当前 student 的样本 `x_gen`。
- `real_score`：冻结 teacher，估计真实数据分布在噪声点 `x_t` 上的去噪方向。
- `fake_score`：训练中的 critic / fake score，估计 generator 分布在同一个 `x_t` 上的去噪方向。

### 1. generator 先生成一个样本

`model/dmd.py` 的 `DMD.generator_loss` 会调用：

```python
image_or_video, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(...)
```

`_run_generator` 来自 `model/base.py`，bidirectional 情况下最终进入：

```python
BidirectionalTrainingPipeline.inference_with_trajectory(...)
```

这个 pipeline 做的是 4-step denoising trajectory 中的 random exit：

1. 从纯噪声 latent 开始。
2. 对 `denoising_step_list` 里的 timestep 逐步 denoise。
3. 在随机选中的 exit step 之前，用 `torch.no_grad()` 推进轨迹。
4. 到 exit step 时，保留 generator 的梯度。
5. 返回该 exit step 的 `x0` 预测，以及当前 step 的 timestep 区间。

配置中：

```yaml
denoising_step_list: [1000, 750, 500, 250]
```

因此目标是 4-step 生成器。

### 2. 对 generator 样本重新加噪

`DMD.compute_distribution_matching_loss` 会从生成样本 `image_or_video` 出发，采样一个 DMD 训练 timestep：

```python
timestep = self._get_timestep(...)
noisy_image_or_video = self.scheduler.add_noise(
    image_or_video.flatten(0, 1),
    noise.flatten(0, 1),
    timestep.flatten(0, 1)
).unflatten(0, image_or_video.shape[:2])
```

这个 timestep 会受 `denoised_timestep_from` / `denoised_timestep_to` 限制，使 DMD 梯度落在当前 generator exit step 对应的时间区间内。

### 3. real_score 和 fake_score 在同一个 noisy point 上预测 x0

核心在 `DMD._compute_kl_grad`：

```python
_, pred_fake_image = self.fake_score(...)
_, pred_real_image_cond = self.real_score(...)
_, pred_real_image_uncond = self.real_score(...)
pred_real_image = pred_real_image_cond + (
    pred_real_image_cond - pred_real_image_uncond
) * self.real_guidance_scale
```

默认配置下：

- real_score 使用 CFG，`real_guidance_scale: guidance_scale`，Wan2.2 DMD 配置里是 `6.0`。
- fake_score 默认 `fake_guidance_scale = 0`，即通常只用 conditional fake_score。

`WanDiffusionWrapper.forward` 返回：

```python
return flow_pred, pred_x0
```

其中 `pred_x0` 是从 flow prediction 转换得到的 clean latent 预测。DMD 主要用的是 `pred_x0`。

### 4. 用 fake_score 和 real_score 的 x0 差值作为分布匹配梯度

代码：

```python
grad = pred_fake_image - pred_real_image
```

直观理解：

- 如果 fake_score 和 real_score 对同一个 noisy point 的 clean sample 预测一致，说明 generator 分布和真实 teacher 分布在这个区域的 score 接近。
- 如果两者不同，就把 generator 往 real_score 指向的方向推。

代码还做了归一化：

```python
normalizer = torch.abs(image_or_video - pred_real_image).mean(dim=[1, 2, 3, 4], keepdim=True)
grad = grad / normalizer
```

这样可以避免不同 timestep / 样本尺度导致梯度过大或过小。

### 5. 用 stop-gradient MSE trick 把手工梯度传给 generator

`compute_distribution_matching_loss` 最后构造：

```python
target = (image_or_video - grad).detach()
loss = 0.5 * F.mse_loss(image_or_video, target, reduction="mean")
```

因为 `target` 被 detach，反向传播时这个 loss 对 `image_or_video` 的梯度就是 `grad` 的缩放版本。这样可以在 PyTorch autograd 中稳定地把 DMD 手工估计的分布匹配梯度传给 generator。

如果 `gradient_mask` 存在，会只在可训练的生成帧 / latent 区域上计算 loss。Wan2.2 TI2V 会固定首帧 latent，不让 DMD loss 推动首帧。

### 6. fake_score / critic 如何训练

`DMD.critic_loss` 中，先用当前 generator 在 `torch.no_grad()` 下生成样本：

```python
with torch.no_grad():
    generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(...)
```

然后对生成样本加噪，让 fake_score 学会 denoise 回 generator 样本：

```python
flow_pred, pred_fake_image = self.fake_score(...)
loss = self.denoising_loss_func(
    pred_fake_image,
    flow_pred,
    x0,
    noise,
    timestep,
)
```

当前 Wan2.2 DMD 配置：

```yaml
denoising_loss_type: x0
```

因此 fake_score 的训练目标是 x0 prediction loss。它让 fake_score 近似当前 generator 分布的 score / denoising function。随后 generator 更新时，`pred_fake - pred_real` 就能形成 DMD 的分布差异方向。

## 4. Wan2.2 TI2V first-frame 逻辑

在 `trainer/wan22_distillation.py` 的 `fwdbwd_one_step` 中，训练 batch 的视频首帧会被 VAE 编码：

```python
first_frame = video_tensor[:, :, :1, :, :]
wan22_image_latent = self.model.vae.encode_to_latent(first_frame).to(device)
```

后续 generator、real_score、fake_score 都会收到 `wan22_image_latent`。

在 bidirectional pipeline 和 DMD loss 中，会通过 mask 固定首帧 latent：

```python
mask2 = masks_like(noisy_image_or_video, zero=True)
noisy_image_or_video = noisy_image_or_video * mask2 + (1 - mask2) * wan22_image_latent
```

`zero=True` 的含义是首帧 mask 为 0，其他帧为 1。也就是说：

- 首帧 latent 来自输入图像 / 视频首帧。
- 后续帧由 generator 生成。
- real_score / fake_score 的 timestep 会为首帧 token 设置特殊值，Wan2.2 使用 separated timestep 逻辑处理 TI2V。

这是迁移 DiffSynth-Studio 时必须严格保留的部分。

## 5. 当前训练循环与更新比例

`trainer/wan22_distillation.py` 的 `train` 中：

```python
TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

if TRAIN_GENERATOR:
    # generator update
    ...

# fake_score / critic update always runs
...
```

所以当：

```yaml
dfake_gen_update_ratio: 5
```

实际行为是：

- step 0：更新 generator，也更新 fake_score。
- step 1-4：只更新 fake_score。
- step 5：更新 generator，也更新 fake_score。

因此当前语义是 fake_score 更新更频繁，generator 更新更少。这个方向符合 DMD2 的 Two Time-scale Update Rule 设计：fake_score / critic 必须更快追踪当前 generator 分布，否则 `pred_fake_x0 - pred_real_x0` 会变成 stale gradient。

迁移时不建议继续使用 `dfake_gen_update_ratio` 这个容易歧义的名字，建议显式使用：

```yaml
fake_score_updates_per_generator_update: 5
generator_updates_per_fake_score_update: 1
```

或：

```yaml
update_schedule:
  fake_score: 5
  generator: 1
```

## Two Time-scale Update Rule 当前实现

Two Time-scale Update Rule，简称 TTUR，指 generator 和 critic/fake_score 用不同更新速度训练。在 DMD 中，fake_score 估计当前 generator 分布的 score；generator 的 DMD 梯度又直接依赖：

```python
grad = pred_fake_x0 - pred_real_x0
```

所以 fake_score 不能落后太多。当前框架通过两种方式实现 TTUR。

第一，generator 和 fake_score 使用不同 optimizer、不同学习率：

```python
self.generator_optimizer = torch.optim.AdamW(
    self.model.generator.parameters(),
    lr=config.lr,
    betas=(config.beta1, config.beta2),
)

self.critic_optimizer = torch.optim.AdamW(
    self.model.fake_score.parameters(),
    lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
    betas=(config.beta1_critic, config.beta2_critic),
)
```

Wan2.2 DMD 配置：

```yaml
lr: 5.0e-07
lr_critic: 1.0e-07
beta1: 0.0
beta2: 0.999
beta1_critic: 0.0
beta2_critic: 0.999
```

第二，fake_score 更新频率高于 generator：

```python
TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
```

随后 fake_score / critic 每个 outer step 都更新。因此 `dfake_gen_update_ratio: 5` 等价于：

```text
fake_score 更新 5 次，generator 更新 1 次
```

迁移到 DiffSynth-Studio 时，TTUR 是必须保留的训练结构，不是可选优化。最低要求：

- `optimizer_generator` 和 `optimizer_fake_score` 分开。
- `lr_generator` 和 `lr_fake_score` 分开。
- `betas_generator` 和 `betas_fake_score` 分开。
- 支持 `fake_score_updates_per_generator_update`，默认建议 5。
- checkpoint 中保存两个 optimizer state 和两个 update counter。
- 日志中分开记录 `generator_loss`、`fake_score_loss`、`generator_grad_norm`、`fake_score_grad_norm`。

不要把 TTUR 简化成“fake_score 学习率调大”。DMD2 的经验是多次 fake_score update 更稳定，因为 fake_score 每次都在新的 generator 样本上拟合当前 fake distribution。

## 6. 当前 training-time rollout / backward simulation 实现

当前代码没有使用 `rollout` 这个名字，等价实现叫：

- `model/base.py::_run_generator`
- `model/base.py::_consistency_backward_simulation`
- `pipeline/bidirectional_training.py::BidirectionalTrainingPipeline.inference_with_trajectory`

它解决的问题是 multi-step student 的 training-inference input mismatch。推理时第 2 步以后的输入不是“真实视频加噪”，而是“generator 上一步输出再加噪”。如果训练时仍然只喂真实样本加噪，student 训练输入和推理输入分布不一致。

### 训练时输入数据

Wan2.2 DMD 训练 batch 中读取：

```python
text_prompts = batch["prompts"]
video_tensor = batch["video"]
first_frame = video_tensor[:, :, :1, :, :]
wan22_image_latent = self.model.vae.encode_to_latent(first_frame)
```

训练数据实际提供：

- prompt：作为文本条件。
- 视频首帧：作为 Wan2.2 TI2V 的 first-frame latent 条件。

当前 DMD 路径中，数据集完整视频不是 generator 的逐像素监督 target。generator 主体样本来自自己的 rollout：

```python
noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
```

Wan2.2 当前 latent shape 是：

```text
[B, 31, 48, 44, 80]
```

### 训练时 rollout 流程

训练 rollout 的流程：

```text
1. 从 Gaussian noise 开始。
2. 随机选择一个 exit step。
3. 按 denoising_step_list 逐步运行 generator。
4. exit step 之前用 torch.no_grad()，只模拟推理输入。
5. 每个非 exit step 后，将 pred_x0 重新加噪到下一 timestep。
6. 到 exit step 时保留 generator 梯度，返回 pred_x0。
```

伪代码：

```python
noisy = torch.randn(...)
exit_step = random_choice([0, 1, 2, 3])

for i, t in enumerate([1000, 750, 500, 250]):
    noisy = replace_first_frame(noisy, wan22_image_latent)

    if i != exit_step:
        with torch.no_grad():
            pred_x0 = generator(noisy, t)
        noisy = scheduler.add_noise(pred_x0, torch.randn_like(pred_x0), next_t)
    else:
        pred_x0 = generator(noisy, t)
        return pred_x0
```

rollout 内部的加噪使用当前仓库的 `FlowMatchScheduler.add_noise`：

```python
sample = (1 - sigma) * original_samples + sigma * noise
```

也就是：

```text
x_t = (1 - sigma_t) * x0 + sigma_t * eps
```

### 训练时还有第二层加噪

rollout 返回 `pred_x0` 后，DMD loss 会再次采样一个 score timestep，并对生成样本加噪：

```python
noise = torch.randn_like(image_or_video)
noisy_latent = self.scheduler.add_noise(
    image_or_video.flatten(0, 1),
    noise.flatten(0, 1),
    timestep.flatten(0, 1)
)
```

这一步不是继续 rollout，而是构造 real_score/fake_score 的 score matching 输入：

```text
generated_x0 -> noisy_latent
real_score(noisy_latent)
fake_score(noisy_latent)
grad = pred_fake_x0 - pred_real_x0
```

fake_score / critic loss 也会先 `no_grad` rollout 得到 generator 样本，再对该样本加噪，让 fake_score 学会 denoise 回 generator 样本。

### 推理时 rollout

Wan2.2 few-step 推理在 `pipeline/wan22_fewstep_inference.py::Wan22FewstepInferencePipeline.inference`。

推理输入：

```text
text prompt + Gaussian noise + 可选 first-frame latent
```

推理不会随机 exit，而是完整跑完所有 denoising steps：

```text
x_1000 = Gaussian noise, 首帧固定为 first-frame latent
pred_x0_1000 = generator(x_1000, 1000)

x_750 = add_noise(pred_x0_1000, fresh_noise, 750)
pred_x0_750 = generator(x_750, 750)

x_500 = add_noise(pred_x0_750, fresh_noise, 500)
pred_x0_500 = generator(x_500, 500)

x_250 = add_noise(pred_x0_500, fresh_noise, 250)
pred_x0_250 = generator(x_250, 250)

decode pred_x0_250
```

训练 rollout 的设计目标就是让 exit step 的 generator 输入分布尽量接近这条推理链路中的真实输入分布。

## BSA-DMD训练方案在当前框架中的支持情况

| 目标 | 当前仓库是否支持 | 说明 |
| --- | --- | --- |
| real_score teacher 使用双向 full attention Wan | 支持 | `real_score` 固定 `is_causal=False`，Wan2.2 路径是 full attention。 |
| fake_score 使用双向 full attention Wan | 支持 | `fake_score` 固定 `is_causal=False`。 |
| fake_score 用 real_score teacher 权重初始化 | 部分支持 | 当 `real_score_name == fake_score_name` 时二者从同源 pretrained 权重加载；但没有显式 copy / custom teacher checkpoint 初始化机制。 |
| student/generator 使用双向 full attention Wan | 支持 | `generator_type: bidirectional`。 |
| student/generator 使用双向 Block Sparse Attention Wan | 不支持 | 当前 bidirectional Wan2.2 只有 full attention；block/local attention 不在该路径。 |
| 4-step DMD 目标 | 支持 | `denoising_step_list: [1000, 750, 500, 250]`。 |
| fake_score 更新快于 generator 的 TTUR | 支持 | 当前 `dfake_gen_update_ratio: 5` 表示 fake_score 每步更新，generator 每 5 个 outer step 更新一次。 |
|                                                       |                  |                                                              |
|                                                       |                  |                                                              |
| training-time rollout / backward simulation | 支持 | 训练时按推理流程模拟 generator 输入，随机 exit step，只在 exit step 反传。 |

