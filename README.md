# PSS-OptCut

## 项目摘要
PSS-OptCut 面向产品-服务系统（PSS）的配置与交付联合优化问题，将“服务等级选择 + 多团队路径排程 + 时间窗约束”统一建模为混合整数规划（MILP），在库存、工时、路线连续性等硬约束下综合优化客户效用、服务及时性与企业利润。

项目的重点不是只求一个业务实例，而是把求解器本身当作实验对象，系统比较不同 valid cut 在 Branch-and-Bound 不同阶段的使用方式，并沉淀“实例特征 + cut bundle + 求解表现”的数据集，用于后续机器学习预测求解时间和策略选择。

## 当前版本在做什么
当前代码的完整实验流程是：

1. `src/Gurobi_Solver/main.py` 生成 1 个随机实例。
2. `src/Gurobi_Solver/sample_generator.py` 对这个固定实例枚举 64 个两阶段 cut bundle。
3. `src/Gurobi_Solver/Solver_builder.py` 对每个 cut bundle 分别建模并求解。
4. 记录实例参数、统计特征、模型规模、cut 组合和 runtime。
5. 将所有结果追加写入 `data/result.csv`。

当前总实验轮数是 `400` 轮：
- 前 `300` 轮为原始中小规模随机实例，记为 `baseline_300`
- 后 `100` 轮为递增压测段，记为 `stress_100`

每轮都会运行 `64` 个 cut bundle，因此理论上会产生：

`400 × 64 = 25600`

条实验记录。

## 核心创新点
- **联合优化建模**：将服务配置与交付排程统一成一个 MILP。
- **定制 valid cut**：针对路径与时间窗结构设计 3 类 cut。
- **分阶段切包实验**：区分 root 与 non-root 阶段，枚举 6 位 cut bundle。
- **数据驱动策略学习**：记录实例特征、模型规模与 runtime，用于训练回归模型预测求解时间。

## 优化模型（概要）
详细变量、约束与求解逻辑见 `src/Gurobi_Solver/Solver_builder.py`。

### 决策变量
- `R[i,m]`：客户 `i` 是否选择服务等级 `m`
- `X[i,j,k]`：团队 `k` 是否从节点 `i` 前往节点 `j`
- `Y[i,k]`：客户 `i` 是否由团队 `k` 服务
- `Z[k]`：团队 `k` 是否出勤
- `T[i,k]`：团队 `k` 在节点 `i` 的服务开始时间

### 当前目标函数权重
`src/Gurobi_Solver/main.py` 中默认使用：

```python
OBJ_COEF = [1, -1, 1, 0]
```

对应于：
- 提升客户效用/价格表现
- 惩罚开始服务时间
- 提升利润
- 暂不使用第 4 个辅助目标

### 主要约束
- 每客户最多选择一种服务等级
- 产品库存限制
- 团队工作时长限制
- 多团队路径流平衡
- 路径连续性和服务时间递推
- 客户时间窗约束
- 服务选择与团队服务的一致性约束

## Valid Cut 设计
候选 cut 池在 `src/Gurobi_Solver/main.py` 中定义，实际添加逻辑在 `src/Gurobi_Solver/Solver_builder.py` 中实现。当前核心 3 类 cut 为：

1. **反向弧互斥 cut**
   `X[i,j,k] + X[j,i,k] <= 1`

2. **基于服务时长的时间窗 cut**
   `T[i,k] + sum(R[i,m] * Service_Time[m]) + Transfer[i][j] - BigM * (1 - X[i,j,k]) <= LateStart[j]`

3. **基于最早开始时间和最小服务时长的时间窗 cut**
   `EarlyStart[i] + Transfer[i][j] + min(Service_Time) - BigM * (1 - X[i,j,k]) <= LateStart[j]`

### 当前实验主路径
虽然 `src/Gurobi_Solver/Solver_builder.py` 中保留了 `trigger1/trigger2/trigger3/trigger4` 四类机制，但当前主实验只使用：

- `trigger4`：两阶段混合策略

在当前实现中，6 位 cut bundle 的含义为：
- 前 3 位：root 阶段静态添加哪些 cut
- 后 3 位：non-root 阶段 callback 添加哪些 cut

因此会枚举 `2^6 = 64` 种组合。

## 实例生成逻辑
实例生成由 `src/Gurobi_Solver/Pamas_generator.py` 完成，`src/Gurobi_Solver/main.py` 负责为每轮实验随机设置规模和参数范围。

### 规模参数
当前规模参数为：
- `I = Num_Customer`
- `M = Num_Service`
- `N = Num_Product`
- `K = Num_Team`

其中代码始终保持：

`Num_Product = Num_Service`

### 400 轮规模设计
`src/Gurobi_Solver/main.py` 中的 `sample_instance_size(times)` 将实验分成 5 个阶段：

- `S1`，`times = 1..300`
  - `I ∈ [5, 9]`
  - `M = N ∈ [3, 5]`
  - `K ∈ [3, 7]`

- `S2`，`times = 301..325`
  - `I ∈ [10, 12]`
  - `M = N ∈ [5, 6]`
  - `K ∈ [6, 8]`

- `S3`，`times = 326..350`
  - `I ∈ [12, 14]`
  - `M = N ∈ [6, 7]`
  - `K ∈ [7, 9]`

- `S4`，`times = 351..375`
  - `I ∈ [14, 16]`
  - `M = N ∈ [7, 8]`
  - `K ∈ [8, 10]`

- `S5`，`times = 376..400`
  - `I ∈ [16, 18]`
  - `M = N ∈ [8, 10]`
  - `K ∈ [9, 12]`

后 100 轮的作用是大规模递增压测。

### 其他随机参数
每轮还会随机生成：
- 服务时间、价格、成本
- 客户效用
- 产品库存
- 客户权重
- 时间窗上下界
- 转移时间矩阵
- 固定成本与可变成本
- 服务与产品消耗关系

## 数据驱动部分（预测设计）
这是项目的核心实验部分。

### 1. 预测目标是什么
这里不是预测最优目标值，也不是用 ML 直接替代优化器，而是预测：

**在某个实例上，某个 cut bundle 的求解时间 `runtime`。**

监督学习任务可以写成：

- 输入：`实例特征 + cut bundle 编码`
- 输出：`求解时间`

### 2. 训练数据如何生成
训练数据由主程序离线实验自动生成：

1. 随机生成一个实例
2. 对该实例枚举 64 个 cut bundle
3. 对每个 bundle 单独建模求解
4. 将这次求解的上下文和结果写成一条样本

也就是说，一条样本代表：

`一个实例 + 一个 cut bundle + 一次真实求解结果`

### 3. 每条实验数据包含什么
当前记录内容至少包括以下几类：

#### 实例标识与策略信息
- `valid_selection`
- `random_seed`
- `scale_phase`
- `scale_stage`
- `size_signature`

#### 实例规模
- `customer_num`
- `service_num`
- `product_num`
- `team_num`

#### 模型规模
- `base_num_vars`
- `base_num_constrs`
- `root_num_constrs`

其中：
- `base_num_vars`：基础 MILP 变量数
- `base_num_constrs`：基础 MILP 约束数
- `root_num_constrs`：root 阶段加完静态 cuts 后的约束数

#### 原始参数与统计量
- 服务时间、价格、成本
- 利润及其均值/标准差/分位数
- 效用矩阵
- 库存
- 客户权重
- 时间窗
- 转移时间矩阵
- 可变成本矩阵
- 服务产品对应关系
- `cut3_percent`

#### 求解结果
- `obj1`
- `obj2`
- `obj3`
- `obj`
- `runtime`

### 4. 输出数据文件
主程序输出：
- `data/result.csv`

求解日志输出：
- `data/实验记录case3.txt`

ML 预处理脚本导出的汇总数据输出：
- `data/总数据.csv`

### 5. 当前 ML 预处理流程
数据驱动实验现在有两条路径：

- Notebook 原型：`notebooks/整体流程_预测时间 .ipynb`
- 模块化脚本：`src/ML_Predict/`

当前脚本化流程的主要步骤如下：

1. 从 `data/result.csv` 读取原始结果表
2. 清洗重复表头和完全重复记录
3. 将 `求解时间` 与 `random_seed` 转为数值
4. 过滤掉 `求解时间 <= 0.5` 的记录
5. 按 `scale_phase + scale_stage + size_signature + random_seed` 组成实例键
6. 保留完整覆盖 `64` 个 bundle 的实例

### 6. 特征工程怎么做
由于很多原始特征是变长向量或矩阵，当前模块化脚本不会直接使用原始对象，而是先提取统计摘要。

`src/ML_Predict/feature_engineering.py` 会对以下信息做统计压缩：

- 服务时间
- 服务价格
- 服务成本
- 利润
- 库存
- 客户权重
- 时间窗下界
- 时间窗上界
- 客户效用
- 产品服务对应关系
- 转移时间矩阵
- 可变成本矩阵

典型统计量包括：
- 均值
- 标准差
- 中位数
- 众数
- 上四分位 / 下四分位
- 偏度
- 峰度

### 7. cut bundle 如何编码进模型
`src/ML_Predict/feature_engineering.py` 会将 `有效不等式选择` 从字符串元组解析成真正的 6 维二进制向量，并拆成：

- `Feature1`
- `Feature2`
- `Feature3`
- `Feature4`
- `Feature5`
- `Feature6`

这 6 个特征就是“策略输入”。

### 8. 训练标签如何定义
最终训练标签是：

- `求解时间`

模块化脚本中会构造：
- `train_x`：去除无关列后的输入特征
- `train_y`：`求解时间`

### 9. 当前尝试过的模型
当前脚本中已经整理并支持多种回归模型：

- `LGBMRegressor`
- `XGBRegressor`
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `SVR`
- `LinearRegression`

其中树模型，尤其是 LightGBM，是当前最主要的方向。

### 10. 当前评估方式
当前脚本主要使用：
- `GroupShuffleSplit`
- `train_test_split`
- `mean_squared_error`
- `mean_absolute_error`
- `r2_score`
- `spearmanr`

### 11. 当前是否已经实现在线闭环
还没有完全闭环。

目前已经实现的是：
- 大量离线数据生成
- 特征工程
- 多种回归模型训练与比较

但还没有完全实现：
- 对新实例自动提特征
- 对 64 个 bundle 全部预测 runtime
- 自动选择预测最短的 bundle 再回到求解器执行

也就是说，当前数据驱动部分主要完成了**离线建模与验证**，而不是完整的**在线策略选择系统**。

## 项目结构
```text
PSS-OptCut/
├─ src/
│  ├─ Gurobi_Solver/
│  │  ├─ main.py             # 主实验入口：400 轮实例生成 + 结果汇总
│  │  ├─ Pamas_generator.py  # 实例与参数生成器
│  │  ├─ Solver_builder.py   # Gurobi 模型构建与分阶段 cut 求解
│  │  └─ sample_generator.py # 枚举 64 个 cut bundle 并生成训练样本
│  └─ ML_Predict/
│     ├─ main.py             # ML 流程入口：清洗、特征工程、训练与评估
│     ├─ data_loading.py     # result.csv 清洗与实例筛选
│     ├─ feature_engineering.py
│     ├─ dataset_builder.py  # 训练集构造与编码
│     ├─ models.py           # LightGBM/XGBoost/RF/GBDT/SVR/LinearRegression
│     ├─ evaluation.py       # 回归指标与实例级任务指标
│     └─ experiments.py      # 基线实验与手工删列实验
├─ data/
│  ├─ result.csv             # 主程序输出的实验结果
│  ├─ 总数据.csv              # ML 脚本导出的衍生训练数据
│  └─ 实验记录case3.txt       # 求解日志
├─ notebooks/
│  └─ 整体流程_预测时间 .ipynb # 原始实验 Notebook（保留）
├─ document/
│  ├─ 业务需求.md
│  ├─ 优化模型.md
│  └─ 算法设计.md
└─ AGENTS.md
```

## 快速开始
### 1. 安装依赖
```bash
/pub/data/caohy/miniconda/envs/PSS/bin/python -m pip install numpy pandas openpyxl scikit-learn xgboost lightgbm matplotlib scipy imbalanced-learn
```

说明：
- 当前推荐直接使用现有环境 `/pub/data/caohy/miniconda/envs/PSS/bin/python`
- 该环境已验证可用，`gurobipy` 已预装；运行主程序至少需要 `gurobipy numpy pandas openpyxl`
- 使用 Notebook 还需要 `scikit-learn xgboost lightgbm matplotlib scipy imbalanced-learn`
- 需配置有效的 Gurobi 许可证

### 2. 运行完整实验
```bash
/pub/data/caohy/miniconda/envs/PSS/bin/python -m src.Gurobi_Solver.main
```

或使用仓库内包装脚本：

```bash
bash run_rsc.sh
```

输出：
- `data/result.csv`
- `data/实验记录case3.txt`

### 3. 做数据驱动分析
```bash
/pub/data/caohy/miniconda/envs/PSS/bin/python -m src.ML_Predict.main
```

输出：
- `data/总数据.csv`
- `data/ml_predict_outputs/`

### 4. 打开原始 Notebook
```bash
/pub/data/caohy/miniconda/envs/PSS/bin/python -m notebook
```

打开：
- `notebooks/整体流程_预测时间 .ipynb`

注意：
- 当前模块化脚本默认直接读取 `data/result.csv`
- 如只想生成衍生表、不跑训练，可执行：

```bash
/pub/data/caohy/miniconda/envs/PSS/bin/python -m src.ML_Predict.main --skip-training
```

- 如果当前环境未安装 `notebook`，先执行 `/pub/data/caohy/miniconda/envs/PSS/bin/python -m pip install notebook`

## 常见调整点
- **实验轮数**：修改 `src/Gurobi_Solver/main.py` 中的 `TOTAL_TIMES`
- **规模增长规则**：修改 `src/Gurobi_Solver/main.py` 中的 `sample_instance_size(times)`
- **目标权重**：修改 `src/Gurobi_Solver/main.py` 中的 `OBJ_COEF`
- **cut 组合策略**：修改 `src/Gurobi_Solver/sample_generator.py` 与 `src/Gurobi_Solver/Solver_builder.py`
- **记录字段**：修改 `src/Gurobi_Solver/sample_generator.py` 中 `current_data`
- **特征工程**：修改 `src/ML_Predict/feature_engineering.py`
- **训练特征筛选**：修改 `src/ML_Predict/constants.py` 中的 `DEFAULT_DROP_COLUMNS`
- **训练与评估流程**：修改 `src/ML_Predict/experiments.py`

## 当前局限
- 当前主实验路径几乎固定为 `trigger4`
- `valid_cut_pool` 更多作为描述池存在，真正加 cut 的逻辑主要手写在 `src/Gurobi_Solver/Solver_builder.py`
- ML 脚本是从 Notebook 逻辑模块化而来，已经可脚本运行，但仍保留明显的实验代码风格
- 在线“预测后自动选 cut bundle 再求解”的闭环尚未完全实现

## 业务需求对齐
业务背景和约束见：

- `document/业务需求.md`
- `document/优化模型.md`
- `document/算法设计.md`

如果你要快速理解项目，建议按这个顺序阅读：

1. `README.md`
2. `src/Gurobi_Solver/main.py`
3. `src/Gurobi_Solver/sample_generator.py`
4. `src/Gurobi_Solver/Solver_builder.py`
5. `src/ML_Predict/data_loading.py`
6. `src/ML_Predict/feature_engineering.py`
7. `src/ML_Predict/main.py`
8. `notebooks/整体流程_预测时间 .ipynb`
