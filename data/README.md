# Data contract

原始 PubMed、FineWeb-Edu、PubMedQA 和 MedMCQA 数据不提交到 Git 仓库。请遵守各数据集许可证、
访问条款和隐私要求。

预处理脚本接受 JSONL，每行至少包含：

```json
{"id": "pmid-123", "text": "normalized document text", "title": "optional"}
```

输出会额外写入 `source`、`split`、`text_sha256`，并在 `metadata.json` 中记录来源、过滤原因、
去重统计、混合比例和文件 SHA-256。训练/验证/测试必须按文档 ID 稳定切分，不能随机拆同一篇摘要。

