# SonarSplat 3DGS 声纳渲染器提取规划

日期：2026-10-07。

参考项目：`/home/onethree/ws/SonarSplat`。

目标项目：`/home/onethree/ws/2d-gaussian-splatting`。

本文根据两个项目的本地代码制定，仅规划第一阶段的提取和独立验证；尚未复制代码、安装依赖、编译或修改训练流程。

## 1. 目标与阶段边界

最终实验目标：使用本项目的 Python 侧点云初始化、损失计算、优化参数与调度、分裂/裁剪判断、输出检测与评估逻辑，搭配 SonarSplat 的 3DGS 声纳渲染器开展对比实验。

当前阶段目标：将参考项目的渲染实现作为独立源码模块整理到本项目的 `submodules`，保留参考项目的 Python 调用接口和渲染计算行为，使它能脱离参考项目目录独立安装、完成前向与反向计算。

本阶段不接入 `train.py`，不修改 `GaussianModel`，不统一两种渲染器的返回字典，也不迁入 SonarSplat 的初始化、损失、优化器配置、增密策略或评估流程。后续接入中的难点在本文中登记，但不在提取时提前解决。

## 2. 已确认的代码边界

本项目当前入口：

- `gaussian_renderer/__init__.py::render`。
- 实际后端为 `submodules/sonar-rasterization` 中的 `sonar_rasterization` 包。
- 模型初始化的尺度为 `[N, 2]`，与二维面元表示一致。

参考项目当前入口：

- `examples/sonar_simple_trainer.py::Runner.rasterize_splats` 调用 `gsplat._sonar_rasterization`。
- 渲染主入口位于 `gsplat/rendering.py::_sonar_rasterization`。
- 输入尺度为 `[N, 3]`，由四元数和三个尺度构造三维协方差，这是后续 3DGS 对比需要保留的表示特征。

实际渲染链可概括为：

```text
gsplat._sonar_rasterization
  → 三维协方差构造、坐标转换
  → 笛卡尔坐标到球坐标的均值和协方差转换
  → AE（方位角/俯仰角）投影与透射率计算
  → AR（距离/方位角）投影与瓦片相交计算
  → 声纳专用高斯/像素相交索引
  → Python/PyTorch 回波累加、饱和处理
  → saturated_returns, meta
```

主要文件及职责：

| 文件或目录 | 提取理由 |
| --- | --- |
| `gsplat/rendering.py` | 声纳主入口，以及共享的渲染入口和工具引用 |
| `gsplat/cuda/_torch_impl_sonar.py` | 透射率、回波累加、饱和处理、球坐标转换 |
| `gsplat/cuda/_torch_impl.py` | 三维协方差构造等 PyTorch 工具 |
| `gsplat/cuda/_wrapper.py` | CUDA 调用与自动求导封装 |
| `gsplat/cuda/_backend.py` | 预编译扩展加载及 JIT 编译 |
| `gsplat/cuda/csrc/`、`gsplat/cuda/include/` | C++/CUDA 实现、绑定、头文件及 GLM 依赖 |
| `gsplat/distributed.py`、`gsplat/utils.py` | 主入口直接导入的共享工具 |
| `gsplat/__init__.py`、`gsplat/version.py` | 原包导出接口和版本标识 |
| `setup.py`、`pyproject.toml`、`LICENSE` | 包构建、安装与来源许可证 |

声纳专用 CUDA 索引绑定名为 `rasterize_to_indices_in_range_sonargs`；对应源文件名是 `rasterize_to_indices_in_range_radargs.cu`。提取清单应按绑定和调用关系识别，不能只按文件名中的 sonar 字样筛选。

## 3. 提取方式与建议目录

建议目录名：`submodules/sonarsplat-renderer`。目录名用于区分后端，内部 Python 包名仍为 `gsplat`，CUDA 扩展名仍保留原有名称。

第一阶段采用“完整 gsplat 包快照 + 独立说明与验证”，不做逐函数裁剪。

原因是：`gsplat/__init__.py` 会导入 compression、strategy 等包，而 `setup.py` 和 JIT 后端会扫描全部顶层 C++/CUDA 源文件，`ext.cpp` 也统一注册多种算子。现在直接删去非声纳文件，容易造成导入失败、链接失败或公开接口变化。先保留完整包，可让源码提取与算法修改分别验证。

保留包内 strategy、optimizers、compression 等代码只是为了维持包的原有结构和接口；未来主项目训练仍使用自身逻辑，不调用这些训练策略。若以后需要精简，作为单独阶段处理，并重新验证依赖与行为。

```text
submodules/sonarsplat-renderer/
├── gsplat/                         # 参考项目完整包源码
│   ├── rendering.py
│   ├── cuda/
│   │   ├── _torch_impl_sonar.py
│   │   ├── _wrapper.py
│   │   ├── _backend.py
│   │   ├── csrc/third_party/glm/
│   │   └── include/
│   └── ...
├── setup.py                        # 原构建入口
├── pyproject.toml                  # 原构建配置
├── LICENSE                         # 原许可证
├── UPSTREAM_README.md              # 原 README，保存来源说明
├── README.md                       # 本次提取的用途、安装及接口说明
├── SOURCE_MANIFEST.json            # 来源路径、逐文件 SHA256、增补/修改清单
├── requirements-renderer.txt       # 声纳路径额外依赖，版本在验证后固定
├── .gitignore                      # 排除本地编译与测试产物
└── validation/
    ├── make_fixture.py             # 独立、小规模的固定渲染输入
    ├── capture_reference.py        # 渲染输出与梯度采集
    ├── compare_results.py          # 原始版/提取版结果对照
    └── VALIDATION.md               # 环境、命令、结果和已知限制
```

复制时排除 `__pycache__`、`.pyc`、`.so`、build、dist、egg-info 等生成物。保留 CUDA/include/GLM 的源文件和许可证，不使用已有二进制代替源码。

不复制项目级 `examples`、`sonar`、`scripts`、`out`、`wandb`、数据集、训练产物或项目级质量评估测试。原训练器和数据读取器可用于查明调用约定，但独立验证脚本不导入它们。

### 源码模块与 Git submodule 的区别

当前目标项目已有的 `submodules` 目录是普通文件跟踪，参考项目本地目录也没有 Git 仓库元数据。因此本阶段建议沿用现有做法，将提取源码作为主仓库直接管理的模块。

若确实需要 Git 意义上的 submodule，应在后续建立可访问的独立仓库，再添加 `.gitmodules` 和固定提交。不要把 `/home/onethree/ws/SonarSplat` 作为仅本机有效的 submodule 地址。此次来源用逐文件 SHA256 固定；参考包中的 `1.4.0` 版本号不足以唯一标识本地修改版。

## 4. 原接口需要记录的内容

保留调用入口：

```python
from gsplat import _sonar_rasterization

returns, meta = _sonar_rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    sat_probability=sat_probability,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=width,
    height=height,
    max_range=max_range,
    near_plane=near_plane,
    far_plane=far_plane,
    sph=True,
    camera_model="ortho",
    packed=False,
    # 其余选项按原接口传递
)
```

必须写明以下约定：

- `means: [N, 3]`；`quats: [N, 4]`，共享工具采用 wxyz 顺序；`scales: [N, 3]` 为激活后的尺度。
- `opacities` 和 `sat_probability` 为 `[N]` 的激活后值；模型存储的 logits 不应直接当作渲染输入。
- `colors` 可采用原入口支持的直接特征或 SH 系数格式；SH 模式还需要 `sh_degree`。后续与本项目 reflection 的对应方式另行确定。
- `viewmats: [C, 4, 4]` 为世界到传感器变换；原训练器由 camtoworlds 求逆得到。
- `Ks: [C, 3, 3]` 在声纳模式中承担距离、角度到像素的映射，不能默认套用相机透视内参解释。
- 原训练器明确传入 `sph=True` 和 `camera_model="ortho"`。函数默认的 `sph=False` 路径中，后续会使用未在该分支定义的 `rae_means`；本阶段说明有效调用方式并保留原实现。
- 原数据读取器给 near/far 传入俯仰角边界 `±vfov/2` 的弧度值；独立输入应明确传入，不能使用函数默认的巨大 far 值。
- 返回图像为 `[C, height, width, 1]`。距离/方位角与行列的对应关系须从原数据路径取样确认，记录翻转、转置和像素中心约定。
- 实际返回是二元组 `(saturated_returns, meta)`，原函数类型注解为三元组。说明以实现为准；提取时不据注解改变接口。
- `meta` 包括 `unsaturated_returns`、`opacity_returns`、`radii`、`means2d`、`depths`、`conics`、`transmittances` 和相交索引等。逐项记录其形状和实际语义。

第一阶段以原训练调用使用的单视角、非 packed 模式建立基线。多视角、distributed 和其他可选路径尚未验证，不因签名中存在这些选项就宣称可用。

## 5. 实施步骤与交付物

### P0：固定来源和原始基线

1. 生成待提取文件清单及 SHA256，记录参考路径、包版本和提取日期。
2. 查明可运行的 Python/PyTorch/CUDA 环境，记录 GPU、驱动、nvcc、编译器和必要依赖版本，不根据 requirements 中所有包推断渲染依赖。
3. 构造固定单视角、小规模 3D 高斯输入：位置、四元数、三个正尺度、不透明度、反射特征、饱和概率及声纳投影参数。
4. 用原实现采集渲染输出、关键 meta 和梯度作为基线。使用固定的标量目标触发反向，仅用于迁移验证，不引入任何训练损失策略。

交付：来源清单、固定输入、原始输出与梯度、运行环境记录。较大的二进制测试产物放入忽略目录，仅将生成方法和结果摘要提交到源码中。

### P1：复制和组织源码

1. 新增 `submodules/sonarsplat-renderer`，按第 3 节复制源码与许可证。
2. 保持原包名、函数名、参数默认值、返回结构、投影/累加/饱和计算和 CUDA 源码。
3. 添加模块 README、来源清单、额外依赖说明和验证脚本。
4. 将复制文件与清单逐项对比，任何必要的构建或打包修改必须单独记录原因与差异。

交付：可独立构建的源码目录。除新增模块和说明外，本项目训练代码保持现状。

### P2：独立构建与导入

1. 在隔离环境安装，优先用 `setup.py` 对应的预编译扩展路径验证；通过固定目标环境使用 `python -m pip install --no-build-isolation ...`，避免构建阶段隐式下载另一套 PyTorch。
2. 原 `setup.py` 声明了 torch/ninja/numpy/jaxtyping/rich 等依赖，但声纳路径还直接依赖 `nerfacc` 和 `matplotlib`，且使用 `typing_extensions`。根据实际环境验证并补充依赖说明，不整份迁入参考项目的 requirements。
3. 检查 `gsplat.__file__` 和 CUDA 扩展加载位置，证明导入的是提取版，而不是其他安装的 gsplat 或参考目录。
4. 在目标模块目录之外运行验证，确认不依赖参考项目绝对路径、不从参考项目加载 Python 文件或旧二进制。
5. 固定源码打包范围，检查可构建的包中包含 CUDA 源码、头文件和 GLM。JIT 路径需要这些资源，源码目录内可用不等于打包后可用。

原始版与提取版保留相同包名，必须分别在独立进程/环境运行；不能在同一解释器中通过切换 sys.path 比较。若验证 JIT，分别指定独立扩展缓存目录，避免共用 `gsplat_cuda` 缓存。记录构建路径及 fast-math、GPU 架构、并行编译配置，使两次运行可比。

交付：成功的构建/导入记录及可复用安装命令。JIT 可作为后续独立验证项，不代替预编译路径的验收。

### P3：渲染与梯度等价验证

至少验证三类输入：

1. 单高斯：检查投影位置、图像尺寸、三个尺度影响及反向路径。
2. 多个重叠高斯：覆盖 AE 透射率、AR 累加、非零饱和概率及可见性。
3. 高斯位于视野外：检查无有效相交时的返回行为；若原实现存在缺陷，应在报告中记录原始版和提取版的共同表现。

对照项目：

- 主返回图像以及 `unsaturated_returns`、`opacity_returns`、`transmittances`。
- `radii`、`means2d`、`conics` 与关键相交索引。
- 同一个标量目标下 means、quats、scales、opacities、colors、sat_probability 的梯度。
- 对 `meta['means2d']` 保留梯度，记录它是否可供后续分裂判断使用。

浮点结果记录最大绝对误差和相对误差，先用 `atol=1e-6, rtol=1e-5` 作为候选阈值；若原实现重复运行存在 CUDA 累加差异，先测得重复运行误差，再解释采用的阈值。离散索引在非边界、无排序并列的固定输入上应一致。

有有效相交且未被裁剪的输入应覆盖有限梯度和实际梯度传递。被遮挡、饱和或 clamp 截断的参数可能合法地产生零梯度，不能把“所有元素梯度非零”设为要求。无相交输入若没有可反传图，也应按原行为记录。

交付：`validation/VALIDATION.md`，明确哪些检查通过、哪些因环境未运行、哪些属于原实现限制。不能仅凭成功 import 宣称完成验证。

### P4：完成第一阶段并登记后续适配事项

汇总文件、构建环境、接口说明与验证结论，确认主项目没有引入 SonarSplat 训练逻辑。之后再进入主项目适配阶段。

## 6. 后续接入必须解决的差异

这些差异影响最终对比实验的解释，但本阶段只记录：

| 差异 | 后续处理要求 |
| --- | --- |
| 本项目 `[N, 2]` 尺度，参考接口 `[N, 3]` | 在本项目模型层定义三维尺度的初始化与优化；保留本项目初始化方法及超参数原则，明确新增维度的规则。不能仅补零或极薄固定尺度，就宣称采用了完整可优化的 3DGS 表示。 |
| 本项目 reflection，参考入口 colors/SH | 明确对应的反射语义、激活方式和裁剪；不直接带入参考训练器的 SH 初始化和优化规则。 |
| 参考渲染器额外要求 sat_probability | 明确固定还是优化、使用何值及其对回波的作用；属于额外实验因素，不能默认为无影响。 |
| 原参考输出含饱和处理和 clamp | 后续明确使用哪一个输出作为本项目 raw_intensity；保留本项目损失归一化、输出归一化逻辑，同时记录两种后端原生回波定义的差异。 |
| 投影梯度不同 | 本项目增密读取 screenspace_points.grad；参考返回的是 AR means2d，AE 路径也参与回波计算。需验证梯度单位、尺度和路径，再映射到本项目分裂统计，保留既有判断规则。 |
| 本项目分裂方向使用局部前两维切向梯度 | 三维表示下需明确该规则的含义和必要的维度适配，不能直接换成参考项目增密策略。 |
| 辅助输出不等价 | 参考 meta 没有与本项目同义的 rend_normal、rend_dist、surf_normal。本项目法线和距离正则、可视化及几何输出依赖它们，后续必须给出语义明确的构造或实验设置，不能补零来维持字典。 |
| opacity_returns、depths 的语义 | 不直接当作本项目 rend_alpha 或像素深度；前者是参考回波累加项，后者是投影阶段的逐高斯量，需另行分析。 |
| 相机矩阵和图像坐标约定 | 对齐矩阵转置、旋转、距离/角度轴、像素中心及 FOV 单位，并用可解释的高斯位置验证。 |
| 来源包含额外声纳成像机制 | 最终实验准确描述为“本项目训练逻辑 + SonarSplat 3DGS 声纳渲染器”；若要单独归因于二维/三维表示，还需要控制其他渲染差异或补充消融。 |

## 7. 第一阶段验收标准

- [ ] 源码位于 `submodules/sonarsplat-renderer`，来源及逐文件校验信息完整。
- [ ] 原 `_sonar_rasterization` 接口、有效调用约定及实际返回值均已记录。
- [ ] 保留 `[N, 3]` 尺度构造三维协方差的渲染路径。
- [ ] 模块可在脱离 SonarSplat 目录的情况下独立构建、导入和运行。
- [ ] 固定输入的前向结果、关键 meta 和反向梯度完成原始版/提取版对照。
- [ ] 包名冲突、扩展缓存及额外依赖的处理方法已记录。
- [ ] 原实现缺陷和未验证选项有明确记录，提取时没有混入算法修正。
- [ ] 本项目初始化、损失、优化、分裂和输出评估代码未因提取而改变。
- [ ] 后续接口适配事项形成清单，可据此制定第二阶段计划。

推荐推进顺序：**固定原始基线 → 完整源码提取 → 独立构建 → 前向/梯度对照 → 主项目接口适配**。第一阶段完成的标志是获得行为可核对的独立渲染模块；完整对比实验属于后续阶段。
