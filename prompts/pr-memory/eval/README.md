# pr-memory 合成对话评测集

`synthetic_conversations.jsonl`:30 条合成的「一天内人机对话」,模仿真实样本的口吻(用户简短、随意、有错字和犹豫;助手长、引用活动数据、每轮以提问结尾)。15 个虚构用户,部分用户跨多天,后一天带着前一天应有的已有条目,用来测 CONFIRM / UPDATE / REMOVE / RESOLVED / ABANDONED。

每行字段:

| 字段 | 说明 |
|------|------|
| `id`, `user_id`, `session_date` | 记录 id、虚构用户、对话日期 |
| `tests` | 这条对话考察的点,例如 `assistant_echo`、`weak_ack`、`privacy_high`、`double_write`、`noop` |
| `profile_items` / `episodic_items` / `intent_items` | 已有记忆,按 v5 模板的输入格式 |
| `turns` | `[{"role": "user" \| "assistant", "text": ...}]` |
| `expected` | 参考标注:`action`、`memory_type`、`target_id`、`fact`、`evidence`、`privacy_level`,以及日期字段(若有) |

**参考标注是按 v5 规则写的草稿,需要人工复核**,不能代替 S2-3 的双人标注集。

## 运行

```bash
pip install openai jinja2
export OPENAI_API_KEY=sk-...           # 只放在环境变量里,不要写进文件
python run_eval.py --dry-run           # 检查数据集并打印第一条渲染后的 prompt,不调用 API
python run_eval.py --out runs/v5-1.jsonl
python run_eval.py --prompt v4 --out runs/v4-1.jsonl
```

常用参数:`--model`(默认 `gpt-4.1`)、`--ids c001 c007` 只跑部分记录、`--workers` 并发数、`--temperature -1` 不传 temperature(部分推理模型不接受该参数)。输出文件必须是新路径,已存在会报错。

## 输出

- `runs/xxx.jsonl`:每条记录的原始输出,v5 还会带上校验拒绝项(`rejects`)、命中(`matched`)、漏掉(`missed`)、多出(`extra`)。
- `runs/xxx.jsonl.summary.json`(仅 v5):precision / recall、HIGH 隐私召回、日期准确率、noop 对话的误提取数、各动作的命中统计。

v4 的输出格式和参考标注不同,只保存原始输出,不打分。

## 打分规则与局限

- 先用 `../v5/extract_validator.py` 校验;被拒的动作不计入命中,单独统计。
- 预测与参考按 `(action, memory_type, target_id)` 匹配。NEW 没有 target_id,只按类型数量匹配,**不比较事实内容**,事实是否正确需要看结果文件人工判断。
- HIGH 隐私召回和日期准确率只在已匹配的动作上计算。
