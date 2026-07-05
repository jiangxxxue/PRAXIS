# optimize_graph_knowledge.py 完整 Prompt 和流程

本文档记录 `memory/optimize_graph_knowledge.py` 当前实际发送给 LLM 的完整 prompt，以及 optimize graph knowledge 的完整执行流程。

## 1. LLM 调用包装

所有 LLM 调用都走 `LLMClient.complete_json(task, payload)`，默认请求 OpenRouter 的
OpenAI-compatible `/chat/completions`：

```text
base_url = https://openrouter.ai/api/v1
api_key  = OPENROUTER_API_KEY
```

请求使用标准鉴权头：

```text
Authorization: Bearer <OPENROUTER_API_KEY>
```

### System Prompt

```text
You are a strict knowledge-graph optimizer. Return only valid JSON. Do not include markdown.
```

### User Message

User message 是一个 JSON object 的字符串：

```json
{
  "task": "<下面某一个完整 task prompt>",
  "...": "task-specific payload"
}
```

其中 `task` 由 `structured_task_prompt(...)` 生成，固定包含以下五个 section：

```text
## Task Background
## Goal
## Input Fields
## Judgment Pipeline
## Output
```

如果模型返回的内容不是 JSON object，或者 JSON schema 不符合当前 prompt 的校验规则，代码不会再发送 JSON 修复 prompt；而是删除本次 cache entry，用同一个原始 `task + payload` 重新生成一次。

## 2. 完整 Prompt: Propagation

用途：判断一个 practice knowledge 是否应该沿一条 caller/callee 调用边，从 `current_node` 传播到 `target_node`。

```text
## Task Background
- The input graph is a dependency graph of functions and methods. Nodes represent code entities, call edges represent caller/callee relationships, and practice-knowledge items are implementation rules mounted on graph nodes.

## Goal
- Decide whether the practice-knowledge item should propagate across this one call-graph edge to target_node.

## Input Fields
- `knowledge`: the practice-knowledge item being evaluated, including id, trigger, content, evidence, and confidence.
- `is_propagated_to_current_node`: true when this knowledge was propagated onto current_node from another node before this decision; false when this knowledge is current_node's direct knowledge.
- `current_node`: the node currently expanding the propagation frontier, including source_code.
- `target_node`: the candidate node reached by this one call edge, including source_code.
- `edge`: direction and meaning for the candidate caller/callee relationship.

## Judgment Pipeline
- Read the knowledge trigger/content/evidence and identify the concrete implementation rule it states.
- Use current_node and target_node source code plus edge direction to decide whether target_node likely needs this rule.
- Use is_propagated_to_current_node only as provenance context; still judge the current edge on its own.
- Propagate only when target_node's implementation likely needs the rule because it calls, is called by, wraps, delegates to, validates for, or consumes behavior from current_node.
- Reject local implementation details that do not constrain target_node.
- Provide a concise reason.

## Output
- Return only JSON: {"propagate": boolean, "reason": string}.
```

### Propagation Payload

```json
{
  "knowledge": {
    "id": "knowledge id",
    "trigger": "trigger",
    "content": "content",
    "evidence": "evidence object/string/null",
    "confidence": "practice knowledge confidence object/number/null"
  },
  "is_propagated_to_current_node": false,
  "current_node": {
    "node_key": "current node key",
    "source_code": "source code slice"
  },
  "target_node": {
    "node_key": "target node key",
    "source_code": "source code slice"
  },
  "edge": {
    "direction": "callee",
    "meaning": "target is a callee/helper used by current"
  }
}
```

`edge.direction=caller` 时，`edge.meaning` 是：

```text
target is a caller/wrapper of current
```

`source_code` 不在 optimize 阶段读取。`knowledge_mount.py` 构造 `dep_graph.with_knowledge.json` 时已经把每个 graph node 的源码切片写入 node。Propagation prompt 只通过 `node_payload(...)` 转发 current/target graph node 上已有的 `source_code` 和可选 `source_code_error`，不再向 LLM 传入 origin node、path 或 hop count。

## 3. 完整 Prompt: Same-Node Knowledge Relation

用途：把同一个 node 下的 `knowledge.direct + knowledge.propagated` 两两比较，一次完成去重和冲突判断。

```text
## Task Background
- The optimizer is canonicalizing and resolving practice knowledge attached to one function or method node in the dependency graph. Direct and propagated knowledge may appear together on the same node.

## Goal
- Classify how two same-node practice-knowledge items relate for implementation.

## Input Fields
- `node`: the graph node whose attached knowledge items are being canonicalized, including source_code.
- `knowledge_a`: first candidate knowledge item, including id, trigger, content, evidence, and confidence.
- `knowledge_b`: second candidate knowledge item, including id, trigger, content, evidence, and confidence.

## Judgment Pipeline
- Identify the concrete implementation rule stated by each knowledge item.
- Use node source code to ground whether the rules apply to the same implementation condition.
- Choose `duplicate` only for surface paraphrases, restatements, or one item being a strict subset of the other when merging would not lose any concrete constraint.
- Choose `conflict` only when both rules cannot be true under the same relevant condition; if so, choose exactly one rule to keep: `a` or `b`.
- Choose `independent` when the items cover complementary rules, different scenarios, different inputs, or different implementation aspects.
- For duplicate or independent, set keep to null.
- Provide a concise reason.

## Output
- Return only JSON: {"relationship": "duplicate"|"conflict"|"independent", "keep": "a"|"b"|null, "reason": string}.
```

### Same-Node Relation Payload

```json
{
  "node": {
    "node_key": "node key",
    "source_code": "source code slice"
  },
  "knowledge_a": {
    "id": "knowledge id",
    "trigger": "trigger",
    "content": "content",
    "evidence": "evidence object/string/null",
    "confidence": "practice knowledge confidence object/number/null"
  },
  "knowledge_b": {
    "id": "knowledge id",
    "trigger": "trigger",
    "content": "content",
    "evidence": "evidence object/string/null",
    "confidence": "practice knowledge confidence object/number/null"
  }
}
```

Schema 校验规则：

- `relationship` 必须是 `duplicate`、`conflict`、`independent` 之一。
- `relationship=conflict` 时，`keep` 必须是 `a` 或 `b`。
- `relationship=duplicate` 或 `relationship=independent` 时，`keep` 必须是 `null`。

冲突检测只检测同一个 node 下的 knowledge pair，不比较相邻调用节点，也不比较跨节点 knowledge。

## 4. 完整 Prompt: Duplicate Cluster Merge

用途：当同一个 node 上的一组 knowledge 被 relation prompt 判定为 duplicate 后，合并为一条 canonical implementation rule。

```text
## Task Background
- The optimizer has grouped duplicate practice knowledge attached to one dependency-graph node and is creating one canonical knowledge item.

## Goal
- Merge the duplicate cluster into one concise canonical implementation rule.

## Input Fields
- `node`: the graph node whose duplicate cluster is being merged, including source_code.
- `knowledge_items`: semantically duplicate practice-knowledge items to merge, each with id, trigger, content, evidence, and confidence.

## Judgment Pipeline
- Read every item in the cluster and identify all concrete constraints that must be preserved.
- Use node source code to keep the merged rule grounded in the function implementation.
- Write one concise rule for this node that covers the duplicate cluster.
- Preserve all concrete constraints from the inputs.
- Do not invent new constraints, examples, APIs, or behavior not present in the inputs.
## Output
- Return only JSON: {"trigger": string, "content": string}.
```

### Duplicate Merge Payload

```json
{
  "node": {
    "node_key": "node key",
    "source_code": "source code slice"
  },
  "knowledge_items": [
    {
      "id": "knowledge id",
      "trigger": "trigger",
      "content": "content",
      "evidence": "evidence object/string/null",
      "confidence": "practice knowledge confidence object/number/null"
    }
  ]
}
```

如果 duplicate cluster 只有一个 item，不调用 LLM，直接沿用该 item 的 `trigger/content`，并把来源 `evidence/confidence` 作为集合保留到 canonical knowledge。

## 5. 完整 Optimize 流程

入口是 `build_optimized_graph(graph_path, llm, propagate=True, version="propagation_v1")`。

### Step 0: 读取 mounted graph

读取输入文件 `dep_graph.with_knowledge.json`。输入 graph 已经由 `knowledge_mount.py` 挂载好：

- `nodes[*].source_code`
- `nodes[*].source_code_error`，可选
- `nodes[*].knowledge.direct`，数组中每个元素都是完整 knowledge object
- 顶层 `knowledge_items`，数组中每个元素都是完整 knowledge object

optimize 阶段不会再次读取源码文件。

### Step 1: LLM edge-gated propagation

当 `propagate=True` 时执行；命令行 `--no-propagate` 会跳过。

1. 从 call edges 构造双向邻接表：
   - source -> target 记为 `direction=callee`
   - target -> source 记为 `direction=caller`
2. 对每个 origin node 上的每条 direct knowledge 启动 BFS frontier。
3. 初始 queue item 是 `(origin_key, [origin_key], 0, False)`，其中最后一个 bool 表示 `is_propagated_to_current_node`。
4. 对每条候选边调用 propagation prompt。Prompt 只包含当前待判断的 knowledge、`is_propagated_to_current_node`、current node、target node 和 edge direction；origin/path/hop 只保留在内部报告与 propagated item 元数据中。
5. 模型必须返回：

```json
{
  "propagate": true,
  "reason": "reason"
}
```

6. 只通过 `propagate=true/false` 判断是否传播，不使用 confidence。
7. `propagate=false` 时，记录 `status=rejected_by_llm`。
8. `propagate=true` 时：
   - 生成一条 `format=propagated_llm` 的 propagated knowledge item。
   - 写入 `propagated_knowledge_items`。
   - 把完整 propagated knowledge object 追加到 target node 的 `knowledge.propagated`。
   - 记录 `status=accepted`。
   - 把 target node 加入 BFS queue，下一跳继续由模型逐边判断是否传播。
9. 每条 direct knowledge 的默认上限：
   - `DEFAULT_PROPAGATION_MAX_HOPS = 4`
   - `DEFAULT_PROPAGATION_MAX_TARGETS_PER_KNOWLEDGE = 12`
10. 下列情况不调用或不接受传播：
    - target 不存在
    - target 已在当前 path 中，避免 cycle
    - target 已经有同一条 direct knowledge
    - 已达到 max hops
    - 已达到 max targets
    - LLM 调用失败
    - LLM 返回 schema 无效且重试后仍无效

所有传播判断都会写入 `reports.propagation_decisions`。

### Step 2: Same-node relation

对每个 node 取：

```text
knowledge.direct + knowledge.propagated
```

去重后，在同一个 node 内做两两 relation 判断。

1. 每个 pair 调用 same-node knowledge relation prompt。
2. `relationship=duplicate`：
   - 用 UnionFind 合并两个 source knowledge object 的 `id`。
   - 后续会进入同一个 duplicate cluster。
3. `relationship=independent`：
   - 不合并。
   - 不删除。
4. `relationship=conflict`：
   - 先记录为 conflict candidate。
   - 等 duplicate clusters 构造完成后，再把 conflict 两端映射到 canonical cluster。

所有 relation 判断都会写入 `reports.knowledge_relation_decisions`。

### Step 3: Same-node conflict resolution

conflict 只在同 node 内处理。只要 relation prompt 返回 `relationship=conflict` 且 `keep=a/b` 有效，就直接按模型选择保留一侧、移除另一侧。

对每个 conflict candidate：

1. 找到两端 knowledge 当前所属的 duplicate cluster root。
2. 如果两端已经在同一个 cluster，跳过。
3. 如果任一 cluster 已经被删除，跳过。
4. 根据 `keep=a/b` 选择 kept root 和 removed root。
5. 如果 `keep` 无效，记录：

```text
resolution_status=unresolved_invalid_keep
```

6. `keep` 有效时，把 removed root 加入 `removed_roots`，并记录：

```text
resolution_status=removed_same_node_conflict
```

conflict relation 不使用 confidence，也不再用 reason 文本做严格/非严格冲突二次判断。

### Step 4: Duplicate cluster merge

对每个 node 的 duplicate cluster 生成 canonical item。

1. 如果 cluster root 在 `removed_roots` 中：
   - 不调用 merge prompt。
   - 用 cluster 第一个 item 的 trigger/content。
   - 写入 `status=removed_conflict`。
   - 把来源 `evidence/confidence` 作为集合保留。
   - 完整 canonical object 进入 node 的 `knowledge.removed`。
2. 如果 cluster 只有一个 item：
   - 不调用 merge prompt。
   - 直接沿用该 item 的 trigger/content。
   - 写入 `status=active`。
   - 把来源 `evidence/confidence` 作为集合保留。
   - 完整 canonical object 进入 node 的 `knowledge.canonical`。
3. 如果 cluster 有多个 item：
   - 调用 duplicate cluster merge prompt。
   - 模型返回 `trigger/content`。
   - `trigger` 为空时回退到 cluster 第一个 item 的 trigger。
   - `content` 为空时回退到 cluster 第一个 item 的 content。
   - 把来源 `evidence/confidence` 作为集合保留。
   - 完整 canonical object 进入 node 的 `knowledge.canonical`。

每个 cluster 的处理结果都会写入 `reports.merge_report`。

### Step 5: 写出 optimized artifact

输出 JSON 顶层结构：

```json
{
  "$schema": "DEP_GRAPH_KNOWLEDGE_OPTIMIZED_V1",
  "meta": {
    "base_graph": "input graph path",
    "generated_at": "UTC iso timestamp",
    "optimization_version": "propagation_v1",
    "propagation_enabled": true,
    "propagation_config": {
      "mode": "llm_edge_gated_bidirectional_call",
      "max_hops": 4,
      "max_targets_per_knowledge": 12
    },
    "implemented_steps": [
      "llm_edge_gated_bidirectional_call_propagation",
      "llm_pairwise_same_node_relation_with_llm_merge"
    ]
  },
  "stats": {
    "num_nodes": 0,
    "num_edges": 0,
    "num_raw_knowledge_items": 0,
    "num_propagated_knowledge_items": 0,
    "num_propagation_decisions": 0,
    "num_accepted_propagation_decisions": 0,
    "num_rejected_propagation_decisions": 0,
    "num_capped_propagation_decisions": 0,
    "num_canonical_knowledge_items": 0,
    "num_active_canonical_knowledge_items": 0,
    "num_conflicts": 0
  },
  "nodes": [],
  "edges": [],
  "knowledge_items": [],
  "propagated_knowledge_items": [],
  "canonical_knowledge_items": [],
  "conflicts": [],
  "reports": {
    "propagation_decisions": [],
    "knowledge_relation_decisions": [],
    "merge_report": []
  }
}
```

默认输出位置：

```text
<graph_dir>/dep_graph.with_knowledge.optimized.json
```
