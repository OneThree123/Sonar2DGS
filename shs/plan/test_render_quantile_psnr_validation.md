# 测试图像分位数归一化与 PSNR 验证记录

日期：2026-10-07
对应规划：`test_render_quantile_psnr_plan.md`

## 实现结果

- `utils/intensity_normalization.py` 提供纯 PyTorch 的 `normalize_test_intensity()`。
  对非负、有限的原始渲染灰度计算全图 0.995 线性分位数 t，
  再计算 `clamp(I / max(t, 1e-8), 0, 1)`。
- 全图包括零背景，不使用 mask，不按距离行分别计算。
  输入支持 `[H, W]` 或 `[1, H, W]`，半精度提升为 float32；输入不被原地修改。
- `train.py::training_report()` 的测试分支直接处理 `raw_intensity`，
  PNG、TensorBoard、PSNR 和 L1 共用处理结果；GT 沿用 `original_image`。
  测试集指标仍是逐视角 PSNR 的算术平均。
- 训练损失、train 诊断视角和 GUI 继续使用原流程。
- 质量模式配置记录分位数、插值、eps、统计范围及新版 `psnr_protocol`。
  旧质量协议检查点在恢复质量历史时被现有签名校验拒绝，原历史文件保留。
- README 已补充计算口径、PNG 量化差异及旧质量协议的恢复限制。

## 环境与执行命令

环境：`/home/onethree/miniconda3/envs/surfel_splatting`，
PyTorch `2.10.0+cu128`，CUDA 可用，1 张可见 GPU。

以下测试均在仓库根目录执行，使用该环境的 Python，并设置：

```bash
export LD_LIBRARY_PATH=/home/onethree/miniconda3/envs/surfel_splatting/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

```bash
/home/onethree/miniconda3/envs/surfel_splatting/bin/python -m unittest discover -s tests -p test_test_intensity_normalization.py -v
/home/onethree/miniconda3/envs/surfel_splatting/bin/python -m unittest discover -s tests -p test_quality_psnr.py -v
PYTHONPATH=tests /home/onethree/miniconda3/envs/surfel_splatting/bin/python -m unittest test_quality_time_logs test_quality_time_integration test_training_time_integration -v
RUN_QUALITY_TIME_GPU_TESTS=1 /home/onethree/miniconda3/envs/surfel_splatting/bin/python -m unittest discover -s tests -p test_quality_time_gpu.py -v
```

## 验证结果

| 验证范围 | 结果 |
| --- | --- |
| 分位数算法及边界值，含 CPU/CUDA 一致性 | 9 项通过 |
| 实际 training_report、指标与图像导出 | 8 项通过 |
| 质量日志、质量模式及训练计时集成 | 37 项通过 |
| 真实 CUDA 渲染、Poisson/P2S、保存及恢复 | 1 项通过，耗时 85.494 秒 |

合计 55 项通过，无跳过项。`git diff --check` 通过；实际 `train` 模块与 CUDA 渲染器导入成功。

关键核对：

- 手算线性分位数、缩放和裁剪与工具一致；全黑、极小阈值、稀疏亮点、
  非有限值、负值、半精度、非连续张量和错误输入得到预期结果。
- 相同 raw 测试图在 `all` / `range` 相机设置下得到相同输出。
  train 诊断视角仍采用实际旧版逐行最大值归一化。
- PNG 像素等于 `round(I_out * 255)`；TensorBoard、L1、PSNR 的输入一致。
  开关图片保存不改变 PSNR，双目录保存只渲染一次且字节相同。
- 两个视角的 PSNR 独立手算后取平均，与测试报告吻合；GT 不做额外缩放或去噪替换。
- 更换 PSNR 协议后，旧质量配置、JSONL、CSV 和 HTML 均未被恢复操作修改。
- CUDA 烟测使用临时小数据集与真实渲染，独立以 NumPy 分位数核对阈值及输出；
  从处理后的浮点图独立计算 PSNR，再与真实报告比较。
  在分位数处理回调中注入延迟，训练时钟保持冻结。
  同时验证质量节点合并、双目录图像一致及新版协议检查点恢复。

## 同一原始图像的新旧处理示例

构造 20×20 的图像：399 个像素灰度为 1，1 个像素为 1000，GT 全图为 0.8。
用同一原始张量比较，结果为：

| 处理方式 | 普通像素输出灰度 | PSNR（dB） |
| --- | --- | --- |
| 旧全图最大值归一化 | 0.001000 | 1.959253 |
| 新全图 0.995 分位数归一化 | 1.000000 | 13.979400 |

此例分位数阈值为 1，演示少量极亮离群点不再压暗主体。
它仅用于说明数值变化，不代表所有真实数据的 PSNR 都会上升。

本次完成了数值、集成及临时小数据集 CUDA 验证；未对正式数据集执行完整训练实验。
