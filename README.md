# predicting-ev-purchases-s6e9-repro


## 成绩

| 项 | 值 |
|---|---|
| 公榜分数 | **0.94657** |
| 排名 | 43/2981（巅峰） |
| 提交文件 | `submission_round30_sprint.csv`，286 571 行 |

---

| 符号 | 权重 | 内容 | 归属 |
|---|---:|---|---|
| **M** | 62.5% | `r8_m00_mega_verbatim.csv` | **megayak 的公开预测**（第三方） |
| **C** | 32.5% | `r6_rebuilt_94651.csv` | **defiaudit 的公开预测**（第三方） |
| **O** | 5% | `submission_round28.csv` | 本项目的 19 列 L1 逻辑回归堆叠 |

先在排名空间融合，再对融合结果取一次排名。融合必须在排名空间做：单独对某一列做单调变换不改变它的 AUC，但多列混合后的排序会变。

---

## ⚠️ 三件必须先知道的事

**一、95% 的权重不是我们做的。**
M、C 是 megayak 与 defiaudit 发布的公开预测。本仓库不拥有它们，也不把它们算作本项目成绩。逐版本署名见 `NOTICE.md` 与 `pinned.json`。

**二、那 5% 也不是纯原创。**
O 的 19 列里，15 列同样是公开预测（najiama OOF 库 6 列、megayak 六视角库 7 列、RealMLP 2 列），只有 4 列来自本项目自行训练的模型（Round23/25/26/27）。所以本项目在这份提交中的原创贡献是**元模型的设计与拟合**，以及**融合比例的设计**，而不是 0.94657 这个分数本身。

**三、0.94657 在本地无法验证。**
本地只有历史 OOF 代理：O 的代理 AUC 为 0.9464439573，比它的父模型高 0.0000622173。但这不是嵌套验证——历史 OOF 被多轮复用，基础模型没有在元模型外折内部重训，公开来源也没有配对 OOF。**本地增益不能换算成公榜增益。** 62.5% 这个比例是 50% 与 75% 两个已知点的中点，属于未验证的假设，没有最优性保证；权重插值不等于分数插值。

---

## 快速开始

需要 Python 3.10，以及 `numpy / pandas / scipy`（版本见 `requirements.txt`）。

```bash
pip install -r requirements.txt

# 1. 取回两份第三方输入
python src/fetch_sources.py
#    已有解压好的数据集副本时可以离线走：
#    python src/fetch_sources.py --from-dir <目录>

# 2. 复现
python src/reproduce_champion.py

# 3. 验证
python src/verify.py
```

第 1 步失败时脚本会打印 Kaggle 数据集页面、版本号、文件名与期望 SHA256，让你手工放入 `inputs/`。**它不会静默降级成用别的文件凑数。**

---

## 目录结构

```
predicting-ev-purchases-s6e9-repro/
├─ pinned.json                  所有输入的版本、SHA256、权重、归属（机器可读的唯一事实源）
├─ requirements.txt
├─ src/
│  ├─ common.py                 契约与校验：rank / sha256 / 读取 / 断言 / 写出
│  ├─ fetch_sources.py          取回并校验第三方输入
│  ├─ reproduce_champion.py     唯一入口：融合并校验输出指纹
│  └─ verify.py                 独立重算（用 pandas 而非 scipy 排名）并校验 31 项
├─ inputs/
│  ├─ submission_round28.csv    O，随仓库提供
│  ├─ megayak_r8_m00_mega_verbatim.csv      M，需下载（已被 .gitignore 排除）
│  ├─ defiaudit_r6_rebuilt_94651.csv        C，需下载（已被 .gitignore 排除）
│  └─ PLACE_THIRD_PARTY_HERE.md
├─ output/                      生成物，不随仓库（已被 .gitignore 排除）
└─ reference/
   ├─ original/                 原始脚本原字节归档，仅供追溯
   └─ HOW_O_WAS_BUILT.md        O 的完整依赖链，以及为什么本仓库不能重训它
```

---

## 校验做了什么

`verify.py` 不导入 `reproduce_champion.py`，而是重读原始文件并用**另一套排名实现**（pandas 的 `rank(pct=True)`，而非 scipy 的 `rankdata`）重算，共 31 项检查：

- 三份输入的 SHA256 与行数、列名、ID 唯一性、值域
- 三份输入的 ID 顺序逐元素一致（融合是按位置的，文件顺序错位会静默产出错误提交）
- 产出文件指纹、行数、ID 顺序
- 独立重算的向量与产出文件**逐元素完全相等**（实测最大差 0）
- 结构不变量：所有值落在 `k/(2n)` 网格上（并列会带来半整数排名）；对结果再取一次排名是恒等变换

输入被改动时，`reproduce_champion.py` 以退出码 2 终止，并打印是哪个文件、期望哈希、实际哈希。

---

## 已知限制

- **O 无法在本仓库重训。** 它需要本项目 Round19/22/23/25/26/27 的逐折缓存和比赛原始数据 `train.csv` / `test.csv`。见 `reference/HOW_O_WAS_BUILT.md`。
- **本仓库不包含比赛数据**，也不包含 4 个本地模型列的原始特征或训练代码。
- **只开源这一版。** 本项目其余 31 轮实验的代码不在本仓库内。
- **无联合 OOF。** M、C 没有配对的完整 OOF，因此这份融合没有可计算的联合交叉验证分数。
- **不能据此推断私榜表现。** M、C 继承了按公榜反馈做的排名区间调整，其私榜优势未经验证。

---

## 第三方来源与许可

本仓库的**原创代码**以 MIT 许可发布（见 `LICENSE`）。第三方预测文件**不在** MIT 覆盖范围内，也不随仓库分发；它们各自属于 megayak 与 defiaudit，来源、版本与哈希见 `NOTICE.md`。

`inputs/submission_round28.csv` 是本项目的产出物，但请注意它本身是公开预测与本地模型的堆叠结果，我们不对其中第三方成分主张权利。

如果你打算复用本仓库，请自行确认该竞赛的数据与代码分享条款。

---

## English summary

This repository reproduces, byte for byte, one Kaggle S6E9 submission scored **0.94657** by the author, from three pinned prediction files:

```
output = rank(0.625·rank(M) + 0.325·rank(C) + 0.05·rank(O))
```

**Read this before citing it.** M (62.5%) and C (32.5%) are *third-party public predictions* by megayak and defiaudit, fetched at pinned versions rather than redistributed here. O (5%) is this project's 19-column L1 logistic stacking, which itself contains 15 public columns. This project's own contribution is the meta-model and the blend design — **not** the 0.94657 score. The number ties a previously confirmed best; no improvement is claimed and no rank is claimed. Local OOF (0.9464439573) is a reused historical proxy, not nested validation, and does not transfer to a leaderboard estimate.

Other 31 experiment rounds from this project are intentionally out of scope.
