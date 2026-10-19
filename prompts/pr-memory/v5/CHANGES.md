# pr-memory v5 修改说明

依据 `healix_memory_plan.md`(v1.0, 2026-10-19)修订 v4。v4 原文保留在 `../v4/pr-template.jinja`。

## 采用的 P0 建议值(待 M1 拍板,若结论不同需同步修改 prompt 与 schema)

| P0 | 本版采用 |
|----|---------|
| P0-4 动作命名 | `NEW / CONFIRM / UPDATE / REMOVE / RESOLVED / ABANDONED`;UPDATE 带 `reason: change \| correction`;`MERGE` 仅 merge LLM 使用,extractor 禁止输出 |
| P0-6 pillar | `energy / sleep / activity / nutrition_food / stress / physiological / lifestyle / other` |
| P0-7 stability | PROFILE、EPISODIC 输出 `null`;仅 INTENT 填写 |
| P0-9 EXPIRED 复活 | 无 due date 用 CONFIRM,有 due date 用 UPDATE 带新日期 |
| P0-10 PROFILE CONFIRM | 照常输出 CONFIRM,由 APPLY 记录 `last_confirmed_at` |

## 本版新增的约定(设计文档未写明,需确认)

- **RESOLVED 携带事件**:INTENT 被 RESOLVED 时,若用户描述了完成事件,事件写在同一条动作的 `fact / anchor_date / granularity` 中,由 APPLY 生成 EPISODIC(对应 S3-6),extractor 不再另发 NEW,避免重复。
- **INTENT 暂停/恢复**:统一动作集里没有 PAUSE,本版用 `UPDATE` + `intent_status: paused | active` 表达。
- **REMOVE 原因**:`no_longer_true / user_requested / incorrect`;`user_requested` 对接 tombstone 与 `forget()`(S1-5、S7-6)。
- **日期**:日期一律 `YYYY-MM-DD`;month 粒度写当月 1 日,year 粒度写 1 月 1 日;"上周"、"几天前"这类不确定的说法归为 month 粒度。设计里没有 week 粒度。
- **模板变量**:`session_date`、`profile_items`、`episodic_items`、`intent_items`、`conversation`、可选的 `is_partial`;条目以 `tojson` 渲染,需带 `id`。详见模板头部注释。

## 逐项修改

| # | v4 问题 | v5 处理 |
|---|--------|--------|
| 1 | 输出完整画像 + `removed_facts`,漏写即丢事实 | 只输出变化动作 `actions`;未提及的条目不输出;空数组即 NOOP(S2-1) |
| 2 | 动作名称与 P0-4 不符,缺 RESOLVED/ABANDONED | 换成统一动作集,并写明每个动作的适用类型和必填字段 |
| 3 | 无条目 ID,只能用字符串指代 | 输入条目带 `id`,有目标的动作必须填 `target_id`(S2-5 校验) |
| 4 | 只输入 PROFILE | 同时输入 EPISODIC 与 INTENT(含 7 天内过期的 INTENT),支持去重、RESOLVED、复活 |
| 5 | 日期写在 fact 文本里,无 session 日期,示例年份 2025,FLAG 无字段 | 新增 `session_date`;结构化 `anchor_date / granularity / due_date / due_granularity`;fact 文本禁止写日期;anchor_date 不晚于 session_date;删除 FLAG |
| 6 | pillar 7 个且 `stress/mood` 与 `stress_mood` 不一致 | 按 P0-6 改为 8 个,全文统一 |
| 7 | 所有类型都填 stability | 仅 INTENT 填写,按打分/淘汰的权重语义重新定义 |
| 8 | 每条只能一类,诊断判为 EPISODIC | 新增 DOUBLE_WRITE:诊断、手术同时输出 EPISODIC + PROFILE(S3-5) |
| 9 | 预约没有归类 | INTENT 明确包含已排期的未来事件,带 due_date |
| 10 | 隐私规则偏向 LOW | "拿不准就判 HIGH";补全 HIGH 类别与边界示例;按内容而非 pillar 判断;UPDATE 不得把 HIGH 降为 LOW |
| 11 | "100% 确定才提取" | 改为可执行标准:必须有用户原话作为 `evidence`;同时说明漏记换药、过敏同样有害 |
| 12 | "主要任务是 PROFILE" | 三类型同等对待 |
| 13 | 不处理"忘掉这件事" | REMOVE + `user_requested`,且不在别处另存 |
| 14 | 不约束事实粒度 | 一条只写一件事(两种过敏分两条),降低 merge 压力与误合并风险 |
| 15 | SCRATCHPAD 与 reasoning 重复 | 只保留简短 scratchpad,删除每条 reasoning;新增 `evidence` 原话;提供 `output-schema.json` 供结构化输出 |
| 16 | 对话放在最后,无说话人格式 | 输入(含对话)放在指令前;模板约定 `User:` / `Assistant:` 前缀;支持 `is_partial` 分段(S2-7) |
| 17 | 第 500 行多余 `</ROLE>`,措辞不通 | 删除;重写开头 |

## 规则冲突的统一结论

| 冲突 | v5 结论 |
|------|--------|
| "I'll try" 拒绝 vs "I'll try to..." 算 INTENT | 单独一句 "I'll try" 回应建议 = 拒绝;"I'll try to + 具体行为" = INTENT |
| 求建议算 INTENT vs "找动力技巧"拒绝 | "帮我 X / 给我做 X 的计划" = INTENT;"How do I / tips for X" 需要对话中有用户本人关切的上下文才算 |
| "eat healthy" 是 INTENT vs 模糊愿望拒绝 | 必须有具体行为、结果或排期事件:"eat healthier" 可以,"stay motivated"、"feel motivated to exercise" 不可以 |
| 稳定性门槛 6 个月 / 1 个月 / 1 周 | 统一为 "3 个月后大概率仍成立" |
| 用户答 "maybe" 标 user_explicit=false | 弱回应不存任何东西 |
| 用户答 "no" 标 user_explicit=false | 否认只能触发对已有条目的 REMOVE/UPDATE,绝不 NEW |
| 膝伤 REMOVE vs UPDATE or REMOVE | 康复 = REMOVE,reason `no_longer_true` |
| 健身车是 STABLE vs 设备排除 | 支撑健康行为的器材(健身车、CPAP)可存;可穿戴品牌与 app 设置不存 |
| "trying to" 触发 INTENT vs "trying the fan tonight" | 当天开始当天结束的计划不存 |

## schema 之外需要校验层(S2-5)检查的规则

JSON Schema 只约束字段与枚举。以下跨字段规则需在校验层实现:
- `target_id` 存在于输入条目,且 `memory_type` 与目标一致
- 各动作的必填/置空字段(见 prompt 的 "Fields by action")
- `anchor_date <= session_date`,`due_date >= session_date`
- RESOLVED/ABANDONED 只能作用于 INTENT;INTENT 的 REMOVE 只允许 `user_requested` / `incorrect`
- UPDATE 不得把 `privacy_level` 从 HIGH 降为 LOW
- `evidence` 是某个 User 轮次的子串

上述规则已在 `extract_validator.py` 中实现(仅用标准库),单测见 `test_extract_validator.py`。
运行方式:`pytest -q prompts/pr-memory/v5`(该目录不在仓库默认 `testpaths` 中,需显式指定路径)。
`validate_output()` 返回可交给 APPLY 的 `valid` 与写入 `extract_rejects` 的 `rejects`;整体无法解析时抛出 `OutputFormatError`。同一条目被多个动作指向时,这些动作全部拒绝。
