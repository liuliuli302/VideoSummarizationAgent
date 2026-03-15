# VideoSummarizationAgent

一个面向研究与实验的视频摘要项目，核心思路是把视频摘要拆成“全局规划 + 多专家打分 + 帧级映射 + 官方协议评测”四个阶段。当前仓库的主执行路径基于多 Agent 推理流水线：先对视频分段并生成片段描述，再由 Planner 生成视频主题与专家权重，随后由多个专家 Agent 对各片段进行重要性判断，最后把片段分数映射回帧级分数，并在 SumMe / TVSum 协议下完成评测与聚合。

项目同时保留了一批面向研究迭代的配置、消融和历史测试代码，因此仓库中会同时出现两类内容：

- 当前主流程：以 `main.py` / `src/main.py` 为入口的推理与评测系统。
- 研究与归档内容：`configs/`、`scripts/run_ablation.py`、`archive/` 以及部分历史测试与兼容接口。

如果你的目标是直接运行当前版本，请优先参考本文的“快速开始”和“运行方式”部分。

## 1. 项目目标

这个项目尝试解决长视频摘要中的几个核心问题：

1. 不同视频类型的“重要片段”定义不同，不能只靠单一统一打分器。
2. 长视频难以一次完成细粒度全局建模，需要先有全局理解，再做局部判断。
3. 摘要不仅要保留高光，还要兼顾叙事主线、信息密度和情绪变化。
4. 连续片段之间存在冗余与上下文关系，片段评分应支持记忆增强。

当前实现采用一种 training-free 的多 Agent 推理范式，更强调可解释性、模块化和快速实验，而不是训练一个端到端监督模型。

## 2. 当前主流程

当前可运行的主流程由 `main.py` 调用 `src/main.py`，核心类是 `src/pipeline/video_pipeline.py` 中的 `VideoSummarizationPipeline`。

完整流程如下：

1. 读取视频基本信息，包括 `video_id`、FPS 和总帧数。
2. 按设定方式进行分段，支持按段数切分、固定帧窗口、滑动窗口。
3. 对每个片段抽取若干代表帧，并使用视觉语言模型生成片段 caption。
4. 将所有片段 caption 交给 Planner Agent，生成：
   - 视频主题 `video_theme`
   - 全局摘要 `global_summary`
   - 各专家权重 `expert_weights`
5. 顺序遍历片段，Planner 和多个专家分别给出该片段的重要性评分。
6. 可选地读取历史 caption 记忆，作为当前片段评分时的上下文。
7. 聚合为片段最终得分，并映射到原始采样帧或完整帧序列。
8. 保存推理结果；如果开启评测，则进一步输出指标、曲线图和聚合报告。

## 3. 模块说明

### 3.1 Agents

当前主流程中实际参与打分的 Agent 位于 `src/agents/`：

- `PlannerAgent`：从全部片段 caption 中推断视频主题、全局摘要和专家权重，并对当前片段给出 Planner 分数。
- `StoryAgent`：偏重叙事推进与主线信息。
- `VisualAgent`：偏重视觉变化、场景和显著视觉内容。
- `EmotionAgent`：偏重情绪表达、冲突和感染力。
- `InformationAgent`：偏重信息密度、讲解价值和事实性内容。

评分聚合逻辑是：

- Planner 单独给出一个分数。
- 各专家按 Planner 生成的动态权重进行加权求和。
- 最终片段分数为 `planner_score + weighted_expert_scores`。

### 3.2 Caption

`src/caption/segment_captioner.py` 负责片段描述生成。当前默认使用本地视觉语言模型：

- 默认模型：`Qwen/Qwen3-VL-8B-Instruct`
- 默认设备：自动选择 CUDA 或 CPU
- 默认提示词：强调持续动作、场景上下文和主要事件

这一部分与 `--llm_mode` 无关。`--llm_mode mock` 只会替换 Planner / Expert 的文本生成逻辑，不会关闭片段 caption 模型加载。

### 3.3 LLM

`src/llm/client.py` 提供两种后端：

- `api`：通过 OpenAI 兼容接口调用真实大模型。
- `mock`：使用确定性规则返回 JSON，方便调试 Planner / Agent 行为。

API 模式默认读取：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`，默认值为 `https://api.openai.com/v1`

因此如果你使用 DeepSeek 或其他 OpenAI 兼容服务，只需要替换 `OPENAI_BASE_URL` 和对应密钥即可。

### 3.4 Memory

`src/memory/memory_manager.py` 提供一个轻量记忆机制：

- 关闭时，不向 Agent 提供历史上下文。
- 开启时，会把之前片段的 caption 作为“Previous segment N: ...”的文本上下文注入到后续打分过程中。
- 可以通过 `--max_history_segments` 限制历史窗口长度。

### 3.5 Evaluation

`src/evaluation/` 提供官方视频摘要评测相关能力：

- `VsumEvaluator`：计算 F1、Precision、Recall、Spearman rho、Kendall tau。
- `EvaluationReporter`：保存逐视频评测文件、曲线图和整体 overview。
- `SplitEvaluationAggregator`：按官方 split 信息做聚合统计。

当前评测阶段会对帧分数生成两个变体：

- `normalized_raw`
- `normalized_smoothed`

再分别计算指标并保存结果。

## 4. 仓库结构

下面是对当前仓库中主要目录的功能解释：

```text
.
├── main.py                        # 根目录快捷入口
├── src/
│   ├── main.py                    # 主 CLI
│   ├── agents/                    # Planner 与专家 Agent
│   ├── caption/                   # 片段 caption 生成
│   ├── config/                    # YAML 配置加载与兼容层
│   ├── data/                      # 数据结构与数据集读取
│   ├── evaluation/                # 官方协议评测与报告
│   ├── io/                        # JSON 输出
│   ├── llm/                       # OpenAI-compatible / mock LLM 客户端
│   ├── memory/                    # 记忆模块
│   ├── models/                    # 历史模型与研究代码
│   ├── pipeline/                  # 当前主流水线
│   ├── preprocessing/             # 视频读取、分段、映射等预处理
│   └── utils/                     # 通用工具
├── configs/                       # 研究配置与消融配置
├── scripts/                       # 常用运行脚本
├── tests/                         # 单元测试与历史实验测试
├── archive/                       # 算法设计与历史文档
└── data/                          # 数据与元信息目录
```

## 5. 环境要求

### 5.1 Python 版本

`setup.py` 要求 Python 版本不低于 3.10。

### 5.2 主要依赖

项目核心依赖包括：

- `torch`
- `torchvision`
- `transformers`
- `openai`
- `decord`
- `opencv_python`
- `Pillow`
- `numpy`
- `scipy`
- `matplotlib`
- `h5py`
- `python-dotenv`
- `PyYAML`

安装方式：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

说明：

- `requirements.txt` 中同时出现了两个 `PyYAML` 版本声明，安装时通常以后者为准；如果你的环境对依赖解析较严格，建议手动统一版本。
- 视觉 caption 模型默认较大，推荐准备可用的 GPU 环境。

## 6. 环境变量

建议在仓库根目录放置 `.env`，或直接在 shell 中导出以下变量：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

如果你需要调整片段 caption 模型，也可以设置：

```bash
export SEGMENT_CAPTION_MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
export SEGMENT_CAPTION_DEVICE="auto"
export SEGMENT_CAPTION_MAX_NEW_TOKENS="96"
export SEGMENT_CAPTION_PROMPT="Describe the sampled frames as one concise segment caption. Focus on persistent actions, scene context, and the main event."
```

## 7. 快速开始

### 7.1 单视频推理

最直接的方式是运行：

```bash
python main.py \
  --task run \
  --video_path path/to/video.mp4 \
  --llm_mode api \
  --llm_model gpt-4o-mini \
  --segment_mode count \
  --segment_value 8 \
  --caption_frames_per_segment 4 \
  --enable_memory \
  --max_history_segments 8
```

仓库中也提供了对应脚本：

```bash
bash scripts/run_inference.sh
```

### 7.2 使用 mock 模式调试 Agent

如果你只想调试 Planner 和 Expert 的评分逻辑，可以将 `--llm_mode` 设置为 `mock`：

```bash
python main.py \
  --task run \
  --video_path path/to/video.mp4 \
  --llm_mode mock \
  --llm_model mock \
  --segment_mode count \
  --segment_value 8
```

注意：这不会跳过片段 caption 生成。只要视频帧被正常读取，系统仍会尝试加载本地视觉语言模型。

### 7.3 单视频评测

如果你已经准备好 SumMe / TVSum 数据集及其映射文件，可以直接运行：

```bash
python main.py \
  --task run_eval \
  --dataset_root /path/to/datasets \
  --dataset_name summe \
  --video_id Jumps \
  --llm_mode api \
  --llm_model gpt-4o-mini \
  --segment_mode count \
  --segment_value 8 \
  --caption_frames_per_segment 4 \
  --enable_memory \
  --max_history_segments 8
```

对应脚本：

```bash
bash scripts/run_real_summe.sh
bash scripts/run_real_tvsum.sh
```

### 7.4 整数据集 split 聚合评测

当指定 `--dataset_name` 但不指定 `--video_id`，且任务为 `eval` 或 `run_eval` 时，系统会自动遍历该数据集的全部视频，并基于 split 信息生成聚合结果：

```bash
python main.py \
  --task run_eval \
  --dataset_root /path/to/datasets \
  --split_root /path/to/datasets/splits \
  --dataset_name tvsum \
  --llm_mode api \
  --llm_model DeepSeek-V3.2 \
  --segment_mode count \
  --segment_value 8 \
  --caption_frames_per_segment 4 \
  --enable_memory \
  --max_history_segments 8 \
  --split_count 5
```

对应脚本：

```bash
bash scripts/run_full_datasets.sh
```

## 8. CLI 参数说明

当前主入口 `src/main.py` 支持以下关键参数：

| 参数                           | 说明                                              |
| ------------------------------ | ------------------------------------------------- |
| `--task`                       | `run`、`eval`、`run_eval` 三选一                  |
| `--video_path`                 | 单视频推理时的视频路径                            |
| `--dataset_root`               | SumMe / TVSum 数据根目录                          |
| `--dataset_name`               | `summe` 或 `tvsum`                                |
| `--video_id`                   | 指定数据集视频 ID；不传时可触发整集评测           |
| `--llm_mode`                   | `api` 或 `mock`                                   |
| `--llm_model`                  | 文本大模型名称，如 `gpt-4o-mini`、`DeepSeek-V3.2` |
| `--segment_mode`               | `count`、`fixed_frames`、`sliding_window`         |
| `--segment_value`              | 段数或每段帧数                                    |
| `--segment_overlap`            | 固定帧 / 滑窗模式下的重叠帧数                     |
| `--caption_frames_per_segment` | 每段抽取多少帧用于 caption                        |
| `--enable_memory`              | 是否启用历史记忆                                  |
| `--max_history_segments`       | 记忆回看长度                                      |
| `--output_root`                | 推理输出目录                                      |
| `--eval_output_root`           | 评测输出目录                                      |
| `--eval_experiment_name`       | 评测实验名                                        |
| `--eval_smooth_window`         | 平滑窗口大小                                      |
| `--split_root`                 | split 文件目录                                    |
| `--split_count`                | 支持 `5` 或 `50`                                  |

## 9. 数据集要求

当前评测代码默认面向 SumMe 和 TVSum，依赖以下工件：

### 9.1 H5 标注文件

- `eccv16_dataset_summe_google_pool5.h5`
- `eccv16_dataset_tvsum_google_pool5.h5`

### 9.2 映射文件

- `summe_mapping.json`
- `tvsum_mapping.json`

### 9.3 视频目录

`DatasetLoader` 默认按以下目录解析原始视频：

- SumMe：`<dataset_root>/SumMe/videos`
- TVSum：`<dataset_root>/TVSum/ydata-tvsum50-v1_1/video`

### 9.4 split 文件

做整数据集协议评测时，默认读取：

- `--split_root /root/autodl-tmp/datasets/splits`

如果你的数据目录不同，需要显式传入 `--dataset_root` 和 `--split_root`。

## 10. 输出结果说明

### 10.1 推理输出

单个视频推理完成后，默认会在 `outputs/inference_results/<video_id>/` 下生成：

```text
captions.json
planner_plan.json
segment_scores.json
frame_scores.json
inference_result.json
```

其中：

- `captions.json`：每个片段的 caption 与帧范围。
- `planner_plan.json`：全局主题、全局摘要、专家权重与原因。
- `segment_scores.json`：每个片段的 Planner 分数、专家分数和最终分数。
- `frame_scores.json`：映射后的帧级分数与 picks。
- `inference_result.json`：完整推理结果汇总。

### 10.2 评测输出

评测完成后，默认会在 `outputs/evaluation/exam_<timestamp>/` 下生成：

```text
overview.json
overview.md
overview_records.json
split_overview.json           # 整数据集评测时生成
<dataset>/<video>/
  frame_scores_variants.json
  eval_normalized_raw.json
  eval_normalized_smoothed.json
  eval_<video>.json
  frame_scores_vs_gt.png
```

这部分输出适合做实验记录、横向比较和后续汇报。

## 11. scripts 目录说明

仓库中已有若干便捷脚本：

- `scripts/run_inference.sh`：单视频推理示例。
- `scripts/run_experiment.sh`：对 SumMe 单视频执行推理和评测。
- `scripts/run_eval.sh`：对 TVSum 单视频执行推理和评测。
- `scripts/run_real_summe.sh`：SumMe 单视频真实评测示例。
- `scripts/run_real_tvsum.sh`：TVSum 单视频真实评测示例。
- `scripts/run_full_datasets.sh`：对 SumMe 和 TVSum 全量运行并聚合 split 结果。
- `scripts/run_ablation.py`：基于 YAML 配置的 training-free ablation 入口。
- `scripts/create_dummy_data.py`：生成简单测试视频。
- `scripts/download_datasets.py`：数据下载占位脚本，目前未实现实际下载逻辑。

## 12. configs 与主流程的关系

仓库中存在一套较完整的 YAML 配置系统，入口位于 `src/config/config_loader.py`，支持：

- `defaults` 递归继承
- 配置深合并
- 旧版字段兼容与运行时规范化

但需要注意：

- 当前主 CLI `src/main.py` 主要使用命令行参数驱动，不直接依赖 `configs/default.yaml`。
- `configs/` 更常见于 ablation、历史实验和兼容路径。
- 如果你只是想跑当前主流程，不必先理解整套 YAML 配置。

## 13. 测试与代码现状说明

`tests/` 中既包含当前模块的测试，也包含较多研究阶段遗留测试。阅读和使用时建议注意：

1. 部分测试文件引用的是历史类名或旧版流水线接口，不一定与当前主入口完全同步。
2. 仓库保留了较多实验演进痕迹，例如 `configs/` 中的旧别名兼容、`archive/` 中的算法设计文档、`src/models/` 中的历史模型代码。
3. 如果你的目标是理解当前可运行系统，优先从 `main.py`、`src/main.py`、`src/pipeline/`、`src/agents/`、`src/evaluation/` 开始。

## 14. 研究文档

如果你希望进一步理解项目背后的设计动机和完整算法草图，可以阅读：

- `archive/算法落地实现.md`
- `archive/算法梳理.md`

这些文档更接近研究设计说明，而不是当前代码入口文档。README 的目标是帮助你快速运行和定位主流程，归档文档则适合深入理解算法演进。

## 15. 常见问题

### Q1. 为什么设置 `--llm_mode mock` 后依然会下载或加载视觉模型？

因为 mock 只作用于 Planner / Expert 的文本推理。片段 caption 由 `SegmentCaptioner` 负责，默认仍会加载本地视觉语言模型。

### Q2. 为什么评测找不到视频或标注文件？

通常是以下原因之一：

- `--dataset_root` 不正确。
- 映射文件 `summe_mapping.json` / `tvsum_mapping.json` 缺失。
- H5 文件不在预期位置。
- 原始视频目录不符合 `DatasetLoader` 约定结构。

### Q3. 为什么仓库里既有命令行参数入口，又有 YAML 配置系统？

这是因为项目同时承载当前主流程和历史研究实验。命令行入口适合直接运行当前版本，YAML 系统更偏向实验配置与兼容。

## 16. 建议的阅读顺序

如果你是第一次接触这个仓库，推荐按以下顺序阅读：

1. `main.py`
2. `src/main.py`
3. `src/pipeline/video_pipeline.py`
4. `src/agents/`
5. `src/caption/segment_captioner.py`
6. `src/evaluation/`
7. `archive/算法落地实现.md`

这样可以先掌握当前系统如何运行，再回过头理解项目的研究设计背景。