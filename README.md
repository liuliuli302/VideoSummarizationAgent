# 基于 LLM Agent 的视频摘要工程实现

这是一个面向工程落地的最小视频摘要系统。

目标输入：一个视频。

目标输出：数据集原始采样帧序列上的重要性分数。

当前仓库已经删掉旧的规则式多阶段 pipeline，只保留一条最小主链路：

1. 视频分段
2. 对每段抽帧并生成多模态 caption
3. Planner Agent 分析全局 caption，输出视频主题和专家权重
4. 4 个 Expert Agent 对每段分别打分
5. Planner 对当前段再给一个整体分数
6. 计算段分数并映射回原始采样帧序列
7. 调用评估模块输出 F1、Precision、Recall、Spearman、Kendall

推荐真实运行参数：

- `--llm_mode api`
- `--llm_model gpt-4o-mini`
- `--segment_mode count`
- `--segment_value 8`
- `--caption_frames_per_segment 4`
- `--enable_memory`
- `--max_history_segments 8`

## 1. 当前代码结构

```text
src/
	agents/
		planner_agent.py
		story_agent.py
		visual_agent.py
		emotion_agent.py
		information_agent.py
	caption/
		segment_captioner.py
	data/
		dataset_loader.py
		schemas.py
	evaluation/
		vsum_evaluation.py
		vsum_runner.py
		vsum_utils.py
	io/
		json_saver.py
	llm/
		client.py
		parser.py
		prompts.py
	memory/
		memory_manager.py
	pipeline/
		video_pipeline.py
	preprocessing/
		frame_mapper.py
		segmenter.py
		video_reader.py
	main.py
```

入口文件：

- [main.py](main.py)
- [src/main.py](src/main.py)

## 2. 环境要求

- Python 3.10
- Linux
- 可访问的 LLM API
- 环境变量 `OPENAI_API_KEY`
- 如使用非默认 OpenAI 兼容接口，可额外设置 `OPENAI_BASE_URL`

安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 数据目录

当前默认读取：

```text
/root/autodl-tmp/datasets/
	eccv16_dataset_summe_google_pool5.h5
	eccv16_dataset_tvsum_google_pool5.h5
	summe_mapping.json
	tvsum_mapping.json
	SumMe/
		videos/
	TVSum/
		ydata-tvsum50-v1_1/
			video/
```

## 4. 运行方式

### 4.1 只跑推理

```bash
python main.py \
	--task run \
	--video_path /path/to/video.mp4 \
	--llm_model gpt-4o-mini \
	--segment_mode count \
	--segment_value 8 \
	--caption_frames_per_segment 5
```

### 4.2 跑数据集单视频推理并评估

```bash
python main.py \
	--task run_eval \
	--dataset_root /root/autodl-tmp/datasets \
	--dataset_name tvsum \
	--video_id EE-bNr36nyA \
	--llm_mode api \
	--llm_model gpt-4o-mini \
	--segment_mode count \
	--segment_value 8 \
	--caption_frames_per_segment 4 \
	--enable_memory \
	--max_history_segments 8
```

### 4.3 开启 Memory Strategy

```bash
python main.py \
	--task run_eval \
	--dataset_name summe \
	--video_id Jumps \
	--llm_mode api \
	--enable_memory \
	--max_history_segments 10
```

参数说明：

- `--task`: `run`、`eval`、`run_eval`
- `--video_path`: 直接输入视频路径时使用
- `--dataset_root`: 数据集根目录
- `--dataset_name`: `summe` 或 `tvsum`
- `--video_id`: 支持 h5 key 对应名或 mapping 后的视频名；若省略且 task 为 `eval`/`run_eval`，则按整个数据集逐视频跑完后再按 split 聚合
- `--llm_mode`: `api` 或 `mock`，无 API key 调试时可用 `mock`
- `--llm_model`: 所有 agent 和 caption 共用的模型名
- `--segment_mode`: `count`、`fixed_frames` 或 `sliding_window`
- `--segment_value`: 分段数量或每段帧数
- `--segment_overlap`: `fixed_frames`/`sliding_window` 下的重叠帧数
- `--caption_frames_per_segment`: 每段用于 caption 的抽帧数
- `--enable_memory`: 开启前序 caption memory
- `--max_history_segments`: memory 最大历史段数
- `--eval_experiment_name`: 评估实验名，对应 `outputs/evaluation/exam_xxx`
- `--split_root`: split 文件目录，默认 `/root/autodl-tmp/datasets/splits`
- `--split_count`: 使用 5 或 50 split 文件聚合

## 5. 输出目录

推理输出默认保存在：

```text
outputs/inference_results/<video_id>/
	captions.json
	planner_plan.json
	segment_scores.json
	frame_scores.json
	inference_result.json
```

评估输出默认保存在：

```text
outputs/evaluation/exam_xxx/
	overview.json
	overview.md
	split_overview.json
	split_overview.md
	<dataset_name>/
		<video_name>/
			eval_<video_name>.json
			eval_normalized_raw.json
			eval_normalized_smoothed.json
			frame_scores_variants.json
			frame_scores_vs_gt.png
```

## 6. 中间结果说明

- `captions.json`: 每个分段的 caption 和抽帧索引
- `planner_plan.json`: 视频主题、全局摘要、专家权重
- `segment_scores.json`: planner 分数、专家分数、最终段分数
- `frame_scores.json`: 原始采样帧序列上的分数
- `inference_result.json`: 汇总结果

## 7. 当前实现约束

- 所有 agent 均通过 LLM API 调用实现
- 不引入 langchain 等额外框架
- Caption 与 Agent 复用同一个 OpenAI 兼容客户端
- 当前只保留对本方法真正有用的最小模块
- 评估复用了 [src/evaluation/vsum_evaluation.py](src/evaluation/vsum_evaluation.py)

## 8. 常见问题

### 8.1 没有设置 API Key

默认 `--llm_mode api` 会在启动时抛出 `OPENAI_API_KEY is not set.`。

如果你只是想联调工程流程，可以使用：

```bash
python main.py --task run_eval --dataset_name tvsum --video_id EE-bNr36nyA --llm_mode mock --llm_model mock
```

### 8.2 `scipy` 缺失

评估模块依赖 `scipy` 计算 Spearman 和 Kendall，需要先安装依赖。

### 8.3 评估时报长度不一致

说明输出的 `frame_scores` 长度没有对齐到数据集的 `picks`。当前实现已经默认按 `picks` 回填，如果你自行改了 mapping 逻辑，需要重新检查 [src/preprocessing/frame_mapper.py](src/preprocessing/frame_mapper.py)。
