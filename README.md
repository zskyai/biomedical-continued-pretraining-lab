# Biomedical Continued Pretraining Lab

一个面向生物医学持续预训练（continued pretraining, CPT）的可复现数据与评测实验仓库。
项目问题是：领域语料能降低 PubMed perplexity，但纯领域训练可能损伤通用能力。仓库把问题拆成
`质量过滤 -> 去重 -> 文档级切分 -> 领域/通用回放配比 -> 污染审计 -> 领域增益/通用遗忘评测`，
并提供可在没有 GPU 和外网的情况下运行的确定性 smoke。

## 当前真实状态

已完成：

- 规范化、质量规则、精确/近重复去重和稳定哈希分片；
- PubMed 与 FineWeb-Edu 的数据契约、流式下载入口和混合比例配置；
- 评测集污染的 n-gram 审计与领域增益/通用遗忘指标；
- 纯标准库的离线 smoke、样例数据和单元测试。

尚未声称完成：

- Qwen 0.5B 的正式 GPU CPT；
- PubMedQA、MedMCQA 或 WikiText 的最终 PPL/准确率；
- 任何“提升百分比”。这些数字必须在固定数据版本、随机种子和 token 预算下真实运行后再写入报告。

## 真实数据

主实验建议使用：

- 领域：NCBI PubMed abstracts（Hugging Face 数据集 `ncbi/pubmed`，或 NCBI 官方 XML）；
- 通用回放：`HuggingFaceFW/fineweb-edu`；
- 领域评测：PubMedQA、MedMCQA、MMLU medical；
- 通用回归：WikiText-2、ARC、HellaSwag。

不要把评测题或其答案混入训练语料。脚本按 PMID/文档 ID 做稳定分片，并可对评测文本做 8-gram
污染检查。原始数据不随仓库提交，使用者需要遵守各数据集许可证和访问条款。

## 快速开始（离线）

```bash
python scripts/run_smoke.py --output-dir outputs/smoke
python -m unittest discover -s tests -v
```

smoke 只验证数据协议、去重、混配和指标计算，不代表模型能力结果。

## 准备真实语料

先安装可选依赖：

```bash
pip install -e ".[data]"
```

从 Hugging Face 流式读取 PubMed 的示例（字段名可能随数据版本变化，脚本会在日志中打印实际字段）：

```bash
python scripts/prepare_corpus.py \
  --pubmed-dataset ncbi/pubmed \
  --pubmed-config pubmed \
  --replay-dataset HuggingFaceFW/fineweb-edu \
  --max-domain-records 200000 \
  --max-replay-records 50000 \
  --domain-ratio 0.8 \
  --output-dir data/processed/pubmed_80_replay20
```

也可以传入已经审计过的 JSONL（每行至少包含 `id` 与 `text`）：

```bash
python scripts/prepare_corpus.py \
  --domain-jsonl data/raw/pubmed.jsonl \
  --replay-jsonl data/raw/fineweb_edu.jsonl \
  --domain-ratio 0.8 \
  --output-dir data/processed/pubmed_80_replay20
```

输出包括 `train.jsonl`、`validation.jsonl`、`test.jsonl` 和 `metadata.json`。metadata 记录数据源、
记录数、字符/token 近似量、过滤原因、去重率、混合比例和 SHA-256，便于复现实验。

## CPT 参考命令

安装训练依赖后，可用 Transformers 参考脚本做小规模验证：

```bash
pip install -e ".[train]"
python scripts/train_cpt.py \
  --model Qwen/Qwen2.5-0.5B \
  --train-file data/processed/pubmed_80_replay20/train.jsonl \
  --validation-file data/processed/pubmed_80_replay20/validation.jsonl \
  --output-dir outputs/qwen05b_pubmed_cpt \
  --max-steps 1000
```

正式实验至少比较：Base、100% PubMed、80/20 PubMed+通用回放、回放+logit anchoring，以及全参与
LoRA CPT。所有变体固定训练 token、batch、学习率和评测生成配置。建议记录领域 held-out PPL、
通用 PPL、医疗问答准确率、gradient norm、loss spike、tokens/s、峰值显存和训练成本。

## 面试可验证表述

在正式 GPU 实验完成前，准确表述是：

> 我实现了生物医学 CPT 的数据清洗、近重复去重、领域/通用回放混配、文档级切分和污染审计，
> 并用离线 smoke 验证了数据与评测闭环；Qwen 0.5B 的正式 GPU 训练和遗忘曲线仍按固定 token
> 预算执行中。

## 目录

```text
src/biomedical_cpt/     清洗、去重、混配、污染和评测函数
scripts/                真实数据准备、CPT 参考训练、离线 smoke
configs/                smoke 与 Qwen 0.5B 实验配置
data/                   数据契约和许可证说明（不提交原始语料）
tests/                  可重复单元测试
```

