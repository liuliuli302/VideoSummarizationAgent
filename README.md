# 基于 Agent 架构的视频摘要科研工程

这是一个面向科研实验的视频摘要系统。项目包含两条可运行路径：

1. **推荐主链路**：多 Agent 视频摘要管线
	- 入口：`python main.py --task summary`
	- 输出：窗口级决策、帧级打分、最终摘要 JSON

2. **基线链路**：`VideoAgent` + 时序策略网络
	- 入口：`python main.py --task experiment|inference|eval`
	- 用于快速训练/推理/评估基线模型

## 1. 环境要求

- Python 3.10
- PyTorch 2.0+
- Linux / CUDA 可选

安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 推荐数据目录

```text
dataset/
	SumMe/
	TVSum/

data/
	raw/
		demo.mp4
	metadata/
	processed/
```

## 3. 配置系统

项目已改为配置驱动模式。主配置入口是 [configs/default.yaml](configs/default.yaml)，
并通过 `defaults` 字段递归合并以下子配置：

```text
configs/
	default.yaml
	datasets/default.yaml
	models/default.yaml
	agents/default.yaml
	summarization/default.yaml
	experiment/default.yaml
	evaluation/default.yaml
	experiment/baseline.yaml
	model/video_encoder.yaml
```

代码中统一使用如下访问方式：

- `config.dataset.data_root`
- `config.model.embed_dim`
- `config.agent.hidden_dim`
- `config.summarization.window.length_sec`
- `config.experiment.batch_size`
- `config.evaluation.num_samples`

## 4. 运行方式

### 3.1 生成视频摘要

```bash
python main.py --task summary --video_path data/raw/demo.mp4
```

可选参数：

- `--title`
- `--category`
- `--asr_path`
- `--output_path`

### 3.2 运行基线实验

```bash
python main.py --task experiment --config configs/experiment/baseline.yaml
```

### 3.3 单视频基线推理

```bash
python main.py --task inference --video_path data/raw/demo.mp4
```

### 3.4 评估基线模型

```bash
python main.py --task eval --config configs/default.yaml --num_samples 10
```

也可以仅通过修改 YAML 切换实验设置：

```bash
python main.py --config configs/default.yaml --task summary
```

## 5. 输出目录

```text
outputs/
	inference_results/
	evaluation/
	generated_videos/
```

## 6. 核心流程

```text
main.py
  -> 加载配置
  -> 读取视频 / 数据集
  -> 特征提取
  -> Agent 推理 / 基线策略网络
  -> 摘要生成或评估
  -> 结果保存
```

## 7. 常见问题

### 6.1 没有 GPU

项目会自动回退到 CPU，但速度会更慢。

### 6.2 `decord` 安装失败

代码已提供 OpenCV 回退路径，仍可运行；如需高性能读取，可单独通过 pip 安装 `decord`。

### 6.3 `matplotlib` 缺失

评估模块仍会保存 JSON 指标，但不会生成曲线图。

### 6.4 权重下载失败

基线视觉编码器默认不强制下载预训练权重，适合离线环境。
