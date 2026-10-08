# 测试图像 0.995 灰度分位数归一化与 PSNR 修改规划

日期：2026-10-07
状态：已按规划完成代码修改及相关验证，包括真实 CUDA 小规模评估。
验证记录：`shs/plan/test_render_quantile_psnr_validation.md`。

## 1. 目标与计算口径

将“0.995 灰度阈值”解释为每张渲染图像全部像素灰度的 99.5% 分位数。
每次测试渲染结束后，直接从该图的 `raw_intensity` 计算阈值；用同一个
缩放系数处理整张图像，使阈值映射到 1，并把超过 1 的值裁剪到 1。
处理后的同一个浮点张量用于测试图像保存、TensorBoard 和 PSNR/L1 计算。

设一张图像的非负、有限灰度为 I，定义：

```text
q = 0.995
t = quantile(flatten(I), q, interpolation="linear")
d = max(t, eps)，eps = 1e-8
I_out = clamp(I / d, 0, 1)
```

当 t >= eps 时，阈值 t 映射到 1，所有大于 t 的像素均饱和到 1。
这里采用纯比例缩放，不减去图像最小值。

具体约定：

- 每张图独立计算阈值，不能把整个测试集或多张图拼在一起计算。
- 全图像素参与，包括零背景；不按前景、有效掩码或非零像素筛选。
- 不按距离行分别计算；测试输出始终使用一个全图阈值，即使相机配置为 `range`。
- 阈值在裁剪到 [0, 1] 之前计算，也不先做现有的逐行最大值归一化。
- 数据集当前按 `convert("L")` 读取，处理对象是单通道声纳灰度图；
  支持 `[H, W]`、`[1, H, W]`，其他形状应显式报错，避免误把批次或颜色通道一起归约。
- GT 使用现有 `viewpoint.original_image` 并保持 [0, 1] 范围。
  本规划按需求对渲染预测图做分位数缩放，GT 不独立做分位数归一化，
  测试 PSNR 也不使用 `anti_noise_image` 替代 GT。

## 2. 当前实现与修改入口

### 2.1 渲染与归一化

`gaussian_renderer/__init__.py`：

- `render()` 返回原始灰度 `raw_intensity`，以及深度、法线、alpha 等辅助结果。
- `normalize_sonar_intensity()` 当前支持 `all` 全图最大值与 `range` 逐行最大值归一化。
- `normalize_render_intensity_for_output()` 根据相机的 `normalizing_method`
  生成 `r_intensity`，训练中的 GUI 显示也调用这个函数。
- `prepare_sonar_intensity_for_loss()` 的 `normalized` 损失分支复用
  `normalize_sonar_intensity()`；直接改此基础函数会同时改变训练损失。

### 2.2 测试评估与图像导出

`train.py::training_report()` 当前处理顺序为：

```text
render() -> normalize_render_intensity_for_output() -> clamp(r_intensity)
         -> 测试 PNG / TensorBoard / L1 / PSNR
```

此函数同时评估完整测试集和前 5 个训练诊断视角。
普通测试迭代、质量时间节点和最终评估均复用这个入口。
当一个节点同时向普通目录和质量目录导出时，已有实现复用同一次渲染和同一个图像张量。

测试导出位置：

- `<model_path>/test_renders/iteration_<iteration>/renders/`
- 开启质量模式图片导出时：
  `<model_path>/quality_time_tests/run_<id>/evaluations/eval_<id>/renders/`

`train.py::save_tensor_image()` 最终以 `round(I_out * 255)` 保存 8 位 PNG。
PSNR 当前直接使用内存浮点张量，因此无需从 PNG 读回计算。

### 2.3 质量日志与离线指标

`utils/quality_time_monitor.py` 在配置中记录 `psnr_protocol`；
`utils/quality_curve_logger.py` 将配置纳入签名并验证检查点恢复口径。
新归一化规则需要更新协议描述，否则日志无法区分旧、新 PSNR。

`metrics.py` 是从 `<model_path>/test/<method>/renders` 和 `gt` 读取 PNG
计算指标的旧离线入口，与当前 `test_renders` 目录并不直接对应。
本次优先修改实际在线测试路径；离线计算应使用已处理的输出，不在 PNG 上再次求分位数。
在线浮点 PSNR 与 8 位 PNG 离线 PSNR 可能存在量化差异，不能要求严格相等。

## 3. 实施方案

### 步骤 1：新增独立的灰度分位数处理工具

建议新增 `utils/intensity_normalization.py`，只依赖 PyTorch，提供：

```python
normalize_test_intensity(intensity, quantile=0.995, eps=1e-8)
# 返回：处理后的灰度张量、裁剪前计算出的原始分位数阈值
```

实现要求：

1. 检查图像形状、非空输入、浮点类型、0 < quantile <= 1 和 eps > 0。
2. 将 NaN、正负 Infinity 置零，沿用当前输出归一化的异常值清理习惯；
   将负灰度置零，然后统计全图分位数。
3. 使用 `torch.quantile(..., interpolation="linear")`，固定插值规则。
4. 用 `clamp_min(t, eps)` 防止除零，再对整图缩放并裁剪到 [0, 1]。
5. 不原地修改输入，保持图像形状和设备；若输入为半精度，先转 float32
   做分位数及缩放，返回 float32，避免不支持的 quantile 类型与低精度误差。
6. 原始 t 与安全分母 d 分开表述，尤其在 t = 0 时不要宣称“阈值已经映射到 1”。

异常及退化图像约定：

- 全黑图：t = 0，输出仍为全零，结果有限。
- 超过 99.5% 的像素为零、仅有少量亮点：t 可能为 0，使用 eps 分母，
  亮点按公式放大并裁剪；不改为统计非零像素或自动回退到最大值归一化。
- 极小的正阈值：同样使用 eps 下限，这是数值保护的明确例外。
- 空图像或不支持的形状：显式报错，不生成貌似正常的指标。

### 步骤 2：在测试分支统一生成输出图像

修改 `train.py::training_report()`：

1. 对 `config["name"] == "test"`，直接将 `render_pkg["raw_intensity"]`
   送入新工具，得到 `image`；替代此分支现有最大值归一化调用。
2. PNG 保存、TensorBoard、L1 和 PSNR 全部复用这个 `image`。
3. GT、文件命名、图像目录、重复目录去重和额外去噪 GT 导出沿用现有流程。
4. 即使 `save_test_images=False`，依然执行相同处理再计算指标。
5. train 诊断分支继续走原归一化流程，确保本次改动范围落在测试图像。
6. 普通迭代、质量时间节点、最终触发共用该逻辑；重合触发只渲染、处理一次。

基础归一化、训练 loss 和训练 GUI 均无需为此修改。
`render_pkg["raw_intensity"]` 保留原值，原始强度统计仍反映实际渲染结果。

### 步骤 3：保持 PSNR 公式并更新评估协议

继续使用 `utils/image_utils.py::psnr()`，数据范围固定为 1：

```text
MSE_view = mean((I_out - GT)^2)
PSNR_view = -10 * log10(MSE_view)
PSNR_test = 各测试视角 PSNR_view 的算术平均
```

不将所有测试图像的 MSE 合并后再取对数。
保留空测试集、完美匹配正无穷及非有限值的已有状态返回方式。
同步 L1 输入，确保同一测试报告中的各项指标对应同一张图像。

在 `utils/quality_time_monitor.py` 的配置中记录：

```text
psnr_protocol = mean_of_view_psnr_original_gt_full_image_quantile_clamped_render_v1
render_normalization = full_image_quantile
render_quantile = 0.995
render_quantile_interpolation = linear
render_normalization_eps = 1e-8
render_quantile_scope = all_pixels_including_zero_background
```

配置与工具共用常量，避免分位数或 eps 与日志描述不一致。
新协议会改变配置签名。现有日志恢复会拒绝携带旧质量协议状态的检查点，
应保留这一保护并明确报告不兼容原因；不能直接修改旧签名或把新结果追加到旧曲线。
本次不新增检查点迁移或质量历史重置功能，若后续需要继续旧训练，需另行设计新质量 run 的恢复策略。

在 README 的测试导出/质量评估说明中补充归一化公式、GT 口径、PNG 量化差异和恢复限制。

## 4. 预计涉及文件

| 文件 | 计划修改 |
| --- | --- |
| `utils/intensity_normalization.py` | 新增纯 PyTorch 的测试灰度处理工具及协议常量 |
| `train.py` | 测试分支使用分位数处理后的统一图像 |
| `utils/quality_time_monitor.py` | 更新 PSNR 协议和归一化配置 |
| `tests/test_test_intensity_normalization.py` | 新增算法与退化输入验证 |
| `tests/test_quality_psnr.py` | 使用真实灰度处理工具验证实际 report、指标及导出 |
| `tests/test_quality_time_logs.py` | 验证协议变化拒绝混用历史且不改写旧日志 |
| `README.md` | 说明新的测试输出和评估口径 |

## 5. 验证计划

### 5.1 数值处理

- 构造已知排序的灰度数组，独立按索引 `(N - 1) * 0.995` 和线性插值
  求参考阈值；核对阈值、缩放比例和超过阈值后的裁剪。
- 构造主体灰度加少量极亮离群点，验证主体不再被全图最大值压暗。
- 构造不同距离行强度的图像，验证全图只有一个缩放系数。
- 正常阈值下，验证正比例放大原始输入后得到相同归一化输出。
- 验证全黑、常量非零、稀疏亮点、负值、NaN/Infinity、空输入等边界行为。
- 检查输入未被修改，输出有限且在 [0, 1]，CPU/CUDA 正常输入结果在浮点容差内一致。

### 5.2 实际测试报告

现有 `tests/test_quality_psnr.py` 通过 AST 提取实际 report 函数，
并把旧归一化函数替换为恒等函数；单靠它无法验证新灰度处理。
需要在测试 namespace 中注入真实的新工具，并让模拟渲染返回 `raw_intensity`。

- 用非均匀预测图与已知原始 GT，独立手算归一化、MSE 与 PSNR；
  验证分位数处理在 PSNR 前执行，GT 未额外缩放且未替换为去噪图。
- 用两张误差不同的测试图验证逐视角 PSNR 算术平均。
- 验证相机 `all` / `range` 配置对相同 raw 测试图产生相同结果。
- 用同一份参考 `I_out` 验证 PNG 像素等于 `round(I_out * 255)`，
  捕获 TensorBoard 输入并验证其与 PSNR 输入一致。
- 验证关闭图片保存仍得出相同 PSNR，双目录导出只渲染一次且 PNG 完全相同。
- 保留空测试集、完美预测状态等现有覆盖；补充 train 诊断沿用旧规则的验证。

### 5.3 集成与小规模真实评估

- 执行新增数值测试、`test_quality_psnr.py`、`test_quality_time_logs.py`、
  `test_quality_time_integration.py` 和 `test_training_time_integration.py`。
- 验证新增分位数计算仍处于评估阶段，不计入累计训练时间。
- 验证配置签名变化在任何历史日志写入前被拒绝。
- CUDA 环境可用时，用现有小数据集做一次测试节点评估，核对两个导出目录、
  质量日志和 TensorBoard；必要时临时记录每张图阈值检查数值。
- 对同一组原始渲染张量比较旧、新输出与 PSNR，说明变化来源；
  新方法是否提高 PSNR 取决于 GT，不将“PSNR 必须上升”作为通过条件。

## 6. 完成标准

- 每张测试图从原始渲染灰度计算全图 0.995 分位数，按统一比例缩放并裁剪。
- 测试 PNG、TensorBoard 和 PSNR/L1 使用同一处理结果。
- GT 与逐视角指标平均口径明确，数值边界可复现。
- 普通测试与质量时间测试使用同一规则，日志可识别新协议。
- 相关数值、报告和集成验证通过；真实 CUDA 验证若暂时不可用，单独记录限制。

实施完成：已新增归一化工具，接入测试报告，更新评估协议和 README，并通过相关验证。
