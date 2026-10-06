# O 是怎么造出来的，以及为什么本仓库重训不了它

`inputs/submission_round28.csv` 是融合公式里的 O（5% 权重）。它是本项目**唯一**原创成分，所以本文件把它交代清楚：它由什么组成、在什么条件下产生的、以及为什么这个仓库刻意不包含重训它所需的一切。

## 一、O 是一个 19 列元模型

产出脚本：`reference/original/round28_run.py`（原字节归档）。

对 19 列预测各取平均百分位排名，然后拟合元模型：

```
StandardScaler
  → LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=1200, tol=1e-5)
  → StratifiedKFold(5, shuffle=True, random_state=42)
  → 测试端取五折概率的平均，再取一次排名
```

19 列的分组（来自 `work/round28/inputs.npz` 的 `columns`，已核对）：

| 来源 | 列数 | 列名 |
|---|---:|---|
| najiama 公开 OOF 库 v5 | 6 | `naji_blend`, `naji_lgb1`, `naji_lgb3`, `sergey_lgb`, `naji_xgb10`, `naji_xgb5` |
| megayak 六视角库 v3 | 7 | `A_lgbm_triple_te_digits_3seed`, `B_xgb_on_A_features`, `C_no_digits_windows_lift_sm2_30_300`, `D_no_exact_key_ladder_windows`, `E_ladder25_250_2500_lift_sm5_50_500`, `F_exact_rate_as_init_score`, `ensemble` |
| megayak RealMLP | 2 | `G_realmlp_10fold`, `G_realmlp_3seed` |
| **本项目自训** | **4** | `own_round25`, `own_reference`, `own_residual`, `own_periodic` |

**15 列公开 / 4 列自研。** 所以 O 不是纯自研模型，本仓库也不这样称呼它。

## 二、4 个自研列的来源

| 列 | 来源 | 具体文件 |
|---|---|---|
| `own_round25` | `work/round25/oof_proxy.npy` 与根目录 `submission_round25.csv`（该轮实际提交，SHA256 `bab79dd4…5c6b81`） | 两者都被使用，且脚本断言二者的映射关系 |
| `own_reference` | `work/round26_reference/runs/exact_te_margin/fold{0..9}.npz` | 逐折 `indices` / `valid_ids` / `test_ids` / `valid` / `test` |
| `own_residual` | `work/round27/runs/residual_keys/fold{0..9}.npz` | 同上 |
| `own_periodic` | `work/round23/runs/tabm_periodic/fold{0..9}.npz` | 同上 |

此外脚本还要读取 `work/round19/oof_candidate.npy` 与 `test_candidate.npy`，以及 `work/round22/runs/tabm_ple/fold{0..9}.npz`——它们不直接作为列，而是用于**校验** `own_round25` 的历史组成关系：

```python
assert np.array_equal(.75*rank(o22mix) + .25*rank(o25), baseline)
```

即 `own_round25` 必须能被历史链条精确重构出来，否则脚本中止。

## 三、本仓库为什么不包含它们

要在这里重训 O，需要同时具备：

- 比赛原始数据 `train.csv`（44.7 MB）、`test.csv`（18.3 MB）、`sample_submission.csv`
- `work/round28/public/` 下 najiama 与 megayak 两份公开库的完整 OOF 与测试预测
- 上表 4 个自研列各自的十折缓存，以及 round19 / round22 的校验数组
- `work/round28/inputs.npz`（197 MB，19 列拼接后的训练 OOF 与测试矩阵）由脚本自行重建

这几项加起来约 **370 MB**，其中绝大多数体积属于第三方公开库和比赛数据。把它们塞进一个"只开源最佳版本"的仓库，既与目标相反，也无权分发其中大部分内容。

因此本仓库的取舍是：**O 作为固定输入随仓库提供，其产出的过程以源码归档 + 本文件说明**。可复现性由哈希保证——`pinned.json` 固定 O 的 SHA256 为 `c1349315bdde776333b8ff3896c8a5ffb6e73884b65d94c5998544c769c60391`，任何替换都会让复现中止。

## 四、如果你确实想重训 O

在原项目工作目录中（而非本仓库），按顺序执行：

```powershell
& 'D:\Miniconda3\envs\amc\python.exe' 'work\round28\run.py'
& 'D:\Miniconda3\envs\amc\python.exe' 'work\round28\export_public.py'
& 'D:\Miniconda3\envs\amc\python.exe' 'work\round28\verify.py'
```

`run.py` 依赖 `__file__` 相对层级定位上游目录与 `train.csv` / `test.csv`，因此**必须保持原目录层级**。缓存命中时会跳过已完成的折；要强制重训需先清空 `work/round28/public15_local4/`。

重训出的 `submission_round28.csv` 若与固定哈希不符，本仓库的复现就会失败——这是预期行为，不是 bug。

## 五、O 的本地指标，及其边界

| 项 | 值 |
|---|---:|
| O 的代理 OOF AUC | 0.9464439573 |
| 父模型（Round25 代理） | 0.9463817399 |
| 增益 | +0.0000622173 |
| 十个 OOF 分组 | 全部为正 |
| O 的公榜成绩 | 0.94648 / rank254 |

**这不是嵌套验证。** 历史 OOF 被本项目多轮复用；基础模型没有在元模型外折内部重训；公开来源没有配对的完整 OOF。因此：

- 不能把这个本地增益换算成公榜增益。事实上 O 的公榜提升只有 0.00001，而本地代理增益是 0.000062。
- 不能由此推断融合比例 0.625 是最优的。它是 50% 与 75% 两个已知点的中点，是一个**未验证的假设**。
- 不能据此判断私榜表现。M、C 继承了按公榜反馈做的排名区间调整，其私榜优势没有得到证明。

这些限制不因十折同向而消失。
