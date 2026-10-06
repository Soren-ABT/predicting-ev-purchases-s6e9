# 第三方来源与归属

本仓库只有**原创代码**适用 MIT 许可（见 `LICENSE`）。本文件说明这份提交依赖了哪些他人的工作，以及我们不主张什么。

## 一、融合成员 M 与 C（占 95% 权重）

这两个文件**不随本仓库分发**。`src/fetch_sources.py` 按固定版本取回，并逐个校验 SHA256；版本或内容不符即中止。

### M — 62.5%

| 项 | 值 |
|---|---|
| 文件 | `r8_m00_mega_verbatim.csv` |
| 作者 | **megayak** |
| 托管 | Kaggle 数据集 `defiaudit/s6e9-realmlp-oof`，version 15 |
| 数据集页 | https://www.kaggle.com/datasets/defiaudit/s6e9-realmlp-oof/versions/15 |
| 原始出处 | https://www.kaggle.com/code/megayak/s6e9-0-94656-reading-the-public-split （script version 350941977） |
| SHA256 | `23052891f418c5b146693f5d811ceb81be4b560defe669972cdb8c3ddc664f78` |
| 作者报告分数 | 0.94656 |

该文件在 defiaudit 的数据集中以"逐字节保留"的形式收录，其哈希与 megayak 原始 notebook 输出一致。

### C — 32.5%

| 项 | 值 |
|---|---|
| 文件 | `r6_rebuilt_94651.csv` |
| 作者 | **defiaudit** |
| 来源 | Kaggle 数据集 `defiaudit/s6e9-realmlp-oof`，version 15 |
| 数据集页 | https://www.kaggle.com/datasets/defiaudit/s6e9-realmlp-oof/versions/15 |
| SHA256 | `7ee8a5698217b3b5783a07526dc887ec5f47d42f667e487d19d79e00f3e52457` |
| 作者报告分数 | 0.94651 |

**已知冲突**：该数据集 notebook 正文把这一基线写成 0.94652，而它的结果 CSV 写成 0.94651。本仓库采用逐文件表的值，并且**不解决这个冲突**。

## 二、O 内部的公开列（占 5% 权重中的大部分）

`inputs/submission_round28.csv` 是本项目的产出物，但它的 19 个输入列里有 15 列是公开预测：

| 贡献者 | 来源 | 版本 | 用到的列数 |
|---|---|---|---|
| **najiama** | https://www.kaggle.com/datasets/najiama/s6e9-oof | v5 | 6 |
| **megayak** | https://www.kaggle.com/datasets/megayak/s6e9-six-feature-views-oof-library | v3 | 7（六种特征视角 + ensemble） |
| **megayak** | 同上（RealMLP 部分） | v3 | 2 |

这些列的 OOF 与测试预测均按 ID 对齐后使用，本项目未执行第三方代码。声明这些列的存在是必要的：它意味着 O **不是**一个纯自研模型。

## 三、本项目自己的贡献

仅限以下两项：

1. **元模型**：把 19 列转为平均百分位排名，用 `StandardScaler + LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=1200, tol=1e-5)` 五折交叉拟合，测试端取五折平均后再次排名。
2. **融合设计**：M/C/O 在排名空间按 0.625 / 0.325 / 0.05 融合，再取一次排名。

**不包含** 0.94657 这个分数。该分数主要由公开预测取得，本项目在其上做的是组合。

## 四、我们明确不主张的

- 不主张 M、C 的任何权利，也不为它们的正确性、可用性或后续版本的稳定性负责。
- 不主张 0.94657 是本项目的模型成绩。
- 不主张本文件曾经或现在优于任何其他提交。它是与既往最佳持平的一份文件。
- 不主张任何排名。项目记录中有一条同日同分的 rank43 反馈，无法确认归属本文件。

## 五、复用前请注意

本仓库的融合代码本身没有争议——它就是几行排名运算。但如果你打算复用**预测文件**，请自行确认：

- 该竞赛关于数据、代码与预测分享的条款；
- 原始数据集各自的使用条款（Kaggle 数据集页面）；
- 本仓库不包含比赛原始数据 `train.csv` / `test.csv`，也未重新分发它们。

如需引用 M、C 或 O 内部的公开列，请引用上表中对应贡献者的原始出处，而不是本仓库。
