# KGRec 高级推荐实验：从 Hybrid 到 Retrieve-and-Rank

## 1. 这轮实验回答什么问题

目标不是简单地把 ALS 分数和 TF-IDF 分数相加，而是依次回答四个问题：

1. ALS 与 TF-IDF 的候选集合是否互补？
2. 如果候选已经包含了用户真正喜欢的物品，能否通过更好的排序把它们推到 Top-K？
3. 在当前 KGRec 数据规模和密度下，哪些协同过滤模型比 ALS 更合适？
4. 当一个复杂模型没有提高指标时，问题出在召回、特征、目标函数，还是训练设置？

最终结论很清楚：**当前数据上最值得保留的主模型是 EASE(lambda=100)**。三路 RRF 能提高候选召回率和少量 Precision/Recall，但没有提高 NDCG；ALS+TF-IDF 的 RRF 比线性分数融合略强；当前 Logistic、LambdaMART、BPR 和 LightGCN 配置都没有超过 EASE。

## 2. 实验协议：为什么要做 inner split

原始数据已经有 `train / validation / test`。如果直接用 validation 训练融合权重或排序器，再用同一个 validation 报分，结果会有信息泄漏。因此本轮采用两层验证结构：

```text
原始 train（596,545 条）
├── inner base-train（507,100 条）── 训练召回模型
└── inner holdout（89,445 条）────── 调参、训练排序器

原始 validation（77,493 条）────── 外层模型比较
原始 test（77,493 条）──────────── 全程封存
```

inner split 是逐用户完成的：每位用户约 15% 的训练交互进入 inner holdout，并确保 base-train 至少保留一个交互。这样排序器看到的标签来自“未来交互”，但不会看到原始 validation，更不会看到 test。

这一区别非常重要：

- **召回模型训练数据**：inner base-train。
- **排序器监督标签**：inner holdout。
- **超参数选择**：inner holdout 或其用户级子划分。
- **最终 validation 报分**：召回模型用完整 train 重训；排序器和超参数冻结。
- **test**：只有选定最终方案后才允许评估一次。

## 3. 第一层：ALS + TF-IDF 多路召回

### 3.1 为什么分别 retrieve 再 rank

推荐系统常拆成两个阶段：

```text
全量 8,640 个物品
        │
        ├─ ALS Top-500：行为相似
        └─ TF-IDF Top-100：内容相似
                    │
                    ▼
            候选并集，平均约 570 个
                    │
                    ▼
             融合/学习排序
                    │
                    ▼
               Top-10 / Top-20
```

这比对两个原始分数直接加权更稳健，因为 ALS 的内积分数和 TF-IDF 的余弦相似度不在同一尺度，且二者的分布随用户变化。分别召回后使用 rank 特征，重点变为“一个物品在每个通道排第几”，而不是比较不可直接对齐的数值。

### 3.2 候选召回率与 oracle ceiling

原始 validation 上的候选诊断结果：

| 候选来源 | Candidate Recall |
|---|---:|
| ALS Top-100 | 0.434751 |
| ALS Top-300 | 0.675907 |
| ALS Top-500 | 0.771698 |
| TF-IDF Top-100 | 0.126546 |
| ALS Top-500 ∪ TF-IDF Top-100 | **0.778544** |

并集平均有 569.59 个物品。TF-IDF 的独立贡献不大，但确实把总体候选召回从 0.771698 提到 0.778544。

为了区分“召回问题”和“排序问题”，还计算了 oracle：假设在候选集合内部有一个完美排序器，把所有 relevant items 放在最前面。

| Oracle 指标 | 结果 |
|---|---:|
| Recall@10 | 0.649607 |
| NDCG@10 | 0.951626 |
| Recall@20 | 0.774431 |
| NDCG@20 | 0.858625 |

这说明候选层已经提供了很高的理论上限，**主要瓶颈是排序，不是候选数量**。注意 oracle NDCG 很高不代表真实模型接近该水平；它只是诊断上限。

## 4. 不需要训练的排序：Weighted RRF

### 4.1 公式

实验采用按候选空间归一化的 reciprocal rank fusion：

\[
d_m(i)=\frac{rank_m(i)-1}{N-1}
\]

\[
score(i)=\sum_m \frac{w_m}{c+d_m(i)}
\]

其中 `m` 是 ALS 或 TF-IDF 通道，`w_m` 是通道权重，`c` 控制头部名次的陡峭程度。它与经典 `1/(k+rank)` 的思想相同，但先把不同通道的 rank 映射到统一的 0 到 1 区间。

inner sweep 选择：

- `c = 0.025`
- `w_ALS = 0.80`
- `w_TFIDF = 0.20`

validation 结果：

| 模型 | P@10 | R@10 | NDCG@10 | P@20 | R@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| 线性 Hybrid | 0.133737 | 0.094686 | 0.143667 | 0.112608 | 0.157604 | 0.155224 |
| Weighted RRF | **0.136103** | **0.096235** | **0.146797** | **0.114368** | **0.159993** | **0.158047** |

RRF 小幅但稳定地超过原线性 Hybrid。原因是 rank 比原始分数更容易跨模型比较，而且 ALS 的主导权重符合本数据集上的实证表现。

## 5. 有监督排序：Logistic 与 LambdaMART

### 5.1 候选级特征

每一个 `(user, candidate item)` 样本构建 21 个特征：

- 原始分数：ALS score、TF-IDF score。
- 名次特征：两个通道的 normalized reciprocal-rank、是否进入各自 Top-20。
- 交叉特征：min、max、product、绝对分差。
- 来源特征：是否由 ALS/TF-IDF 召回、是否两个通道同时召回。
- 物品特征：流行度、是否有 tag、tag 数量。
- 用户特征：历史长度、历史物品 tag 覆盖率。
- 分歧特征：TF-IDF 缺失、TF-IDF 高而 ALS 低、ALS 高而 TF-IDF 低。

标签为 inner holdout 中是否发生过交互。排序器训练完成后冻结，再对由完整 train 重训的召回器所产生的 validation 候选打分。

### 5.2 Logistic ranker

技术实现是 `StandardScaler + SGDClassifier(loss="log_loss", class_weight="balanced")`。它训练的是 pointwise 二分类概率：

\[
P(y=1\mid x)=\sigma(w^T x+b)
\]

结果 NDCG@10 只有 **0.079996**。主要原因不是 Logistic “不能排序”，而是当前训练目标与最终指标不一致：

- 它逐物品优化分类损失，不直接优化每个用户内部的 Top-K 顺序。
- 正负样本极不平衡，`class_weight` 只能修正全局比例，不能表达不同用户的 query group。
- 训练候选来自 inner 模型，最终候选来自完整 train 重训后的模型，存在分布漂移。
- 大量易负样本会主导损失，而 NDCG 最关心列表头部少量难例。

### 5.3 LambdaMART

LambdaMART 使用 XGBoost `XGBRanker`，目标是 `rank:ndcg`。它按用户组织 query group，通过树模型学习非线性交互，early stopping 后最佳轮次为 14。

结果 NDCG@10 为 **0.130487**，明显强于 Logistic，但仍弱于 RRF 和线性 Hybrid。特征重要性显示：

- `als_rank_score` 最重要，说明 ALS 名次仍是最可靠信号。
- `retrieved_by_als`、`score_min` 排名靠前，说明“被 ALS 召回”以及“两路都不差”很关键。
- TF-IDF 原始分数的重要性有限。
- 两个手工 disagreement flag 几乎没有被树使用。

这与先前误差分析一致：ALS 高/TF-IDF 低的强分歧样本命中率约 7.37%，而 TF-IDF 高/ALS 低的命中率约 0.009%。因此不能对两种分歧对称处理；TF-IDF 更适合作为弱证据或冷启动补充，而不是覆盖强 ALS 信号。

LambdaMART 没有超过 RRF 的可能原因包括：监督标签仍稀疏、只有一次 inner holdout、候选选择带来 selection bias、特征有限，以及数据规模不足以稳定学习复杂的用户条件规则。这里的正确结论是“本配置没有获益”，而不是“LTR 不适用于推荐”。

## 6. 关键突破：EASE

### 6.1 模型是什么

EASE 是一个线性 item-item 协同过滤模型。令二值用户物品矩阵为 `X`，它求解：

\[
\min_B \|X-XB\|_F^2+\lambda\|B\|_F^2,\quad diag(B)=0
\]

闭式解为：

\[
P=(X^TX+\lambda I)^{-1},\qquad B_{ij}=-\frac{P_{ij}}{P_{jj}},\qquad B_{jj}=0
\]

用户的预测分数是：

\[
S=XB
\]

直觉上，`B[i,j]` 学的是“用户喜欢物品 i 时，对物品 j 的条件性证据”。`diag(B)=0` 防止模型仅复制已经看过的物品。

### 6.2 为什么它在 KGRec 上特别强

本数据集有 5,199 个用户、8,640 个物品、596,545 条训练交互，平均每个用户约 115 条历史。这样的数据并不极端稀疏：

- ALS 用 64 维 latent factors 压缩整个偏好空间，可能损失细粒度 item-item 共现结构。
- EASE 直接学习 8,640 × 8,640 的稠密 item-item 权重，表达能力更贴近当前数据。
- 它有闭式解，不依赖负采样、epoch、初始化或复杂优化器，训练稳定。

代价是 `O(I^2)` 内存和接近 `O(I^3)` 的矩阵求逆，因此非常适合当前 8,640 个物品，但物品规模达到数十万时不可直接使用。

### 6.3 正则化调参

inner validation：

| lambda | NDCG@10 | NDCG@20 |
|---:|---:|---:|
| 100 | **0.250639** | **0.248940** |
| 300 | 0.248297 | 0.244953 |
| 500 | 0.243563 | 0.238287 |
| 1000 | 0.230682 | 0.223815 |

所以冻结 `lambda=100`，在完整 train 上重训并评估原始 validation：

| 模型 | P@10 | R@10 | NDCG@10 | P@20 | R@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| EASE(lambda=100) | **0.278073** | **0.193442** | **0.304735** | **0.217090** | **0.296595** | **0.309646** |

相对 ALS，NDCG@10 提升约 **160.8%**；相对原线性 Hybrid，提升约 **112.1%**。这是本轮最重要的实验结果。

## 7. 三路召回：EASE + ALS + TF-IDF

为了验证 EASE 是否还能从 ALS/TF-IDF 的互补候选获益，构建：

```text
EASE Top-500 ∪ ALS Top-500 ∪ TF-IDF Top-100
                    ↓
              Weighted RRF
```

inner sweep 选择：

- `c = 0.005`
- `w_EASE = 0.90`
- `w_ALS = 0.075`
- `w_TFIDF = 0.025`

候选 Recall 从单独 EASE Top-500 的 0.776409 提高到 **0.858025**，平均候选数约 810.83；但最终结果是：

| 模型 | P@10 | R@10 | NDCG@10 | P@20 | R@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| EASE | 0.278073 | 0.193442 | **0.304735** | 0.217090 | 0.296595 | **0.309646** |
| 三路 RRF | **0.278515** | **0.193936** | 0.303888 | **0.217109** | **0.296727** | 0.308835 |

这揭示了一个经典现象：**候选召回提高，不等于最终 NDCG 提高**。三路并集找到了更多 relevant items，但 RRF 对 EASE 已经很好的头部次序产生了扰动。Precision/Recall 略增，NDCG 略降，说明新增命中没有被放在足够靠前的位置。

因此下一代 ranker 如果继续开发，应把 EASE score/rank 作为主特征，并只在模型有充分证据时改动它，而不是重新对三个通道平权排序。

## 8. BPR-MF 与 LightGCN

### 8.1 BPR-MF

BPR 用隐式反馈的 pairwise objective。对用户 `u`、正样本 `i`、未观察负样本 `j`：

\[
\mathcal{L}_{BPR}=-\log\sigma(s_{ui}-s_{uj})+\lambda\|\Theta\|^2
\]

实现使用 PyTorch、64 维 embedding、Adam、每个正样本动态采一个未见物品作为负样本、batch size 8192。inner sweep 在 10/30 epoch 中选择 30。

validation NDCG@10 为 **0.120457**，低于 ALS；但 Coverage@20 为 **0.690162**，高于 EASE 的 0.415856。这说明它更愿意推荐长尾物品，准确率与覆盖率之间存在明显 trade-off。

### 8.2 LightGCN

LightGCN 在用户—物品二部图上传播 embedding：

\[
E^{(l+1)}=D^{-1/2}AD^{-1/2}E^{(l)}
\]

最终 embedding 是 0 到 L 层的平均，再使用 BPR 损失训练。本实现采用 3 层、64 维、full-batch 图传播和每个正样本一个负样本；inner sweep 在 30/100 epoch 中选择 100。

validation NDCG@10 为 **0.070055**，Coverage@20 仅 0.075231。训练损失在 100 epoch 时仍下降，所以这更准确地说明：

- 当前 full-batch、单负样本、固定学习率的配置未调到合适区域；
- 可能存在 popularity collapse；
- 需要更好的负采样、学习率/正则化搜索和更长训练；
- 不能据此断言 LightGCN 本身不适合 KGRec。

不过从实验资源分配角度看，EASE 已显著领先，继续深调 LightGCN 的优先级低于开发以 EASE 为 anchor 的融合排序器。

## 9. validation 统一结果

按 NDCG@10 排序：

| 排名 | 模型 | P@10 | R@10 | NDCG@10 | P@20 | R@20 | NDCG@20 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | EASE(lambda=100) | 0.278073 | 0.193442 | **0.304735** | 0.217090 | 0.296595 | **0.309646** |
| 2 | ThreeChannel-RRF | 0.278515 | 0.193936 | 0.303888 | 0.217109 | 0.296727 | 0.308835 |
| 3 | RetrieveRank-RRF | 0.136103 | 0.096235 | 0.146797 | 0.114368 | 0.159993 | 0.158047 |
| 4 | Linear Hybrid | 0.133737 | 0.094686 | 0.143667 | 0.112608 | 0.157604 | 0.155224 |
| 5 | LambdaMART | 0.117869 | 0.082256 | 0.130487 | 0.097259 | 0.134367 | 0.137486 |
| 6 | BPR-MF | 0.109444 | 0.074532 | 0.120457 | 0.090719 | 0.123050 | 0.125731 |
| 7 | ALS | 0.112002 | 0.079324 | 0.116859 | 0.103703 | 0.146013 | 0.135650 |
| 8 | Logistic ranker | 0.076996 | 0.051917 | 0.079996 | 0.072091 | 0.097164 | 0.091550 |
| 9 | LightGCN | 0.062050 | 0.041478 | 0.070055 | 0.052481 | 0.070352 | 0.073035 |
| 10 | Popularity | 0.033795 | 0.022661 | 0.036673 | 0.027419 | 0.036977 | 0.037490 |
| 11 | TF-IDF | 0.026909 | 0.019753 | 0.027323 | 0.024466 | 0.035340 | 0.032127 |

指标各自回答不同问题：

- `Precision@K`：推荐的 K 个物品里有多少命中。
- `Recall@K`：用户真实喜欢的物品中有多少被找回。
- `NDCG@K`：既看命中，也更奖励靠前的命中；本轮主要选择指标。
- `HitRate@K`：用户是否至少命中一个。
- `Coverage@K`：全体推荐结果覆盖了多少物品。
- `Novelty@K`：推荐物品是否偏长尾。
- `Diversity@K`：同一列表的内容向量是否多样。

因此“最佳”必须和产品目标绑定。若主要目标是离线准确性，选 EASE；若必须显著增加 catalogue exposure，BPR 的 coverage 值值得作为 reranking 或多目标优化信号研究。

## 10. 技术栈与每个组件负责什么

| 层 | 技术 | 用途 |
|---|---|---|
| 数据 | pandas + Parquet | 交互表、物品元数据、实验结果 |
| 稀疏计算 | SciPy CSR | 用户物品矩阵、TF-IDF、图邻接矩阵 |
| 数值线代 | NumPy / OpenBLAS | EASE Gram matrix 与矩阵求逆 |
| 内容模型 | scikit-learn TF-IDF / normalize | tag 向量与 cosine-style 相似度 |
| 线性学习 | scikit-learn SGDClassifier | 可扩展 pointwise Logistic ranker |
| 树排序 | XGBoost XGBRanker | LambdaMART / `rank:ndcg` |
| 深度模型 | PyTorch CPU | BPR embedding、LightGCN sparse propagation |
| 评估 | 项目统一 evaluator | P/R/NDCG/HR/Coverage/Novelty/Diversity |

本机运行使用 CPU；PyTorch 没有 CUDA。EASE 的主要计算由底层 BLAS 加速，8,640 个物品的规模仍可在内存中完成。

## 11. 如何复现

从项目根目录依次运行：

```powershell
python run_retrieve_rank_experiments.py
python run_ease_eval.py
python run_three_channel_rrf.py
python run_torch_cf_eval.py
python run_advanced_report.py
```

主要实现文件：

- `src/recommenders/hybrid.py`：ALS + TF-IDF 线性 hybrid 与组件分数。
- `run_retrieve_rank_experiments.py`：inner split、多路召回、候选诊断、RRF、Logistic、LambdaMART。
- `src/recommenders/ease.py`：EASE 闭式解与推荐接口。
- `run_ease_eval.py`：EASE inner 调参与 validation 评估。
- `run_three_channel_rrf.py`：EASE + ALS + TF-IDF 三路召回与 RRF。
- `src/recommenders/torch_cf.py`：BPR-MF 与 LightGCN。
- `run_torch_cf_eval.py`：深度协同过滤的 inner epoch sweep。
- `run_advanced_report.py`：统一合并 validation 指标，不访问 test。

主要结果文件在 `artifacts/results/`：

- `advanced_model_comparison_val.csv/json/md`：统一模型排名。
- `retrieve_rank_candidate_analysis.json`：两路候选 recall 和 oracle。
- `three_channel_candidate_analysis.json`：三路候选 recall。
- `retrieve_rank_feature_importance.csv`：Logistic 系数和 LambdaMART 重要性。
- `rrf_inner_sweep.csv`、`ease_inner_sweep.csv`、`three_channel_rrf_inner_sweep.csv`、`torch_cf_inner_sweep.csv`：所有 inner 调参记录。
- `retrieve_rank_val.json`、`ease_val.json`、`three_channel_rrf_val.json`、`torch_cf_val.json`：外层 validation 结果。

## 12. 下一步的科学实验顺序

当前最合理的后续顺序不是继续堆模型，而是围绕 EASE 缩小问题：

1. **冻结 EASE(lambda=100) 作为主 baseline**，不要再根据 test 改参数。
2. 构建 `EASE Top-500 ∪ ALS Top-500 ∪ TF-IDF Top-100` 候选，但让排序器以 EASE rank 为 anchor。
3. 使用 out-of-fold 方式生成 ranker 训练特征，减少 inner/full retriever 的分布漂移。
4. 为 EASE 与 ALS/TF-IDF 的分歧建立四象限分析，并加入用户历史长度、tag coverage、item popularity 等 gating 特征。
5. 优先尝试 residual reranking：学习“相对 EASE 应上调或下调多少”，而不是从零预测 relevance。
6. 若优化多样性或覆盖率，再在 EASE Top-N 后增加 MMR/xQuAD 或 BPR-aware reranking，并显式报告准确率损失。
7. 所有设计决策在 validation 完成并冻结后，对 test 只跑一次，作为最终无偏报告。

最有希望的公式形态是：

\[
score_{final}=score_{EASE}+\alpha(u,i)\cdot residual(ALS,TFIDF,context)
\]

其中 `alpha(u,i)` 是一个保守 gate：只有在 TF-IDF 有 tag、用户历史 tag coverage 足够、且多个协同信号不冲突时，才允许内容模型明显调整 EASE 顺序。这个方向直接吸收了本轮最重要的发现：**EASE 的头部排序很强，新增候选有价值，但融合器必须学会少动、只在有证据时动。**
