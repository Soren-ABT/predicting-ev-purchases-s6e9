# 这里需要两份第三方预测文件

`inputs/` 下的这两个文件**不随本仓库分发**，因为它们属于其他作者：

| 本地文件名 | 数据集内文件名 | 归属 | SHA256 |
|---|---|---|---|
| `megayak_r8_m00_mega_verbatim.csv` | `r8_m00_mega_verbatim.csv` | megayak | `23052891f418c5b146693f5d811ceb81be4b560defe669972cdb8c3ddc664f78` |
| `defiaudit_r6_rebuilt_94651.csv` | `r6_rebuilt_94651.csv` | defiaudit | `7ee8a5698217b3b5783a07526dc887ec5f47d42f667e487d19d79e00f3e52457` |

两者都来自 Kaggle 数据集 **`defiaudit/s6e9-realmlp-oof` version 15**：
https://www.kaggle.com/datasets/defiaudit/s6e9-realmlp-oof/versions/15

## 取回方式

```bash
python src/fetch_sources.py                 # 走 Kaggle CLI
python src/fetch_sources.py --from-dir DIR  # 从已解压的数据集副本复制
python src/fetch_sources.py --check         # 只看状态，不下载
```

已有该数据集解压副本时用 `--from-dir`，离线也能用。

## 为什么不能随便找个版本

本包的产出文件被固定在 SHA256 `17680b24…dc7bcf` 上。数据集若更新，同名文件的内容可能变化，融合结果就不再是那份提交。因此脚本会核对哈希，**不匹配即中止**，不会退而求其次。

手工放入也可以，但请自行确认哈希一致。

`inputs/submission_round28.csv` 是本项目自己的产出物，随仓库提供，无需下载。
