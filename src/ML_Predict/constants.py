"""Shared constants for the ML prediction pipeline."""

from __future__ import annotations

TARGET_COLUMN = "full_runtime"
BUNDLE_COLUMN = "有效不等式选择"
GROUP_COLUMN = "instance_id"
BASELINE_BUNDLE = "(0, 0, 0, 0, 0, 0)"
EXPECTED_BUNDLE_COUNT = 64

INSTANCE_KEY_COLUMNS = [
    "scale_phase",
    "scale_stage",
    "size_signature",
    "random_seed",
]

STAT_FEATURE_SPECS = [
    ("服务时间", "服务时间"),
    ("服务价格", "服务价格"),
    ("服务成本", "服务成本"),
    ("利润", "利润"),
    ("库存", "库存"),
    ("客户权重", "客户权重"),
    ("服务时间窗Early", "时间窗下界"),
    ("服务时间窗Late", "时间窗上界"),
    ("客户效用值", "客户效用"),
    ("产品服务对应关系矩阵", "产品服务对应关系"),
    ("转移时间矩阵", "转移时间矩阵"),
    ("可变成本", "可变成本"),
]

DEFAULT_DROP_COLUMNS = [
    "序号",
    "求解选择",
    "random_seed",
    "gap",
    "客户效用值",
    "客户效用值下界",
    "客户效用值梯度",
    "产品服务对应关系矩阵",
    "产品服务对应关系",
    "产品服务对应关系下界",
    "产品服务对应关系上界",
    "产品服务对应关系均值",
    "产品服务对应关系标准差",
    "产品服务对应关系上四分位",
    "总性价比",
    "总任务开始时间和",
    "总利润",
    "函数目标值",
    "求解时间",
    "legacy_runtime",
    "库存",
    "库存梯度",
    "库存下界",
    "客户权重",
    "利润",
    "服务成本",
    "服务时间",
    "服务价格",
    "服务时间窗Early",
    "服务时间窗Late",
    "转移时间矩阵",
    "可变成本",
    "产品数",
    "服务时间梯度",
    "服务价格梯度",
    "服务成本梯度",
    "客户权重gap",
    "有效不等式选择",
    "instance_id",
]

MANUAL_SPLIT_DROP_FEATURES = [
    "cut3百分比",
    "服务时间窗Early上界",
    "转移时间矩阵下界",
    "客户权重众数",
    "服务数",
    "服务价格下界",
    "服务时间窗Late上界",
    "利润众数",
    "服务时间窗Late下界",
    "时间窗下界众数",
    "时间窗上界中位数",
    "服务时间窗Early上四分位",
    "转移时间矩阵均值",
    "固定成本",
    "客户权重中位数",
    "可变成本中位数",
    "服务成本下界",
    "库存众数",
    "利润下四分位",
    "服务价格中位数",
    "服务成本中位数",
    "转移时间矩阵峰度",
    "求解时间",
    "legacy_runtime",
    "服务时间窗Early标准差",
    "客户效用众数",
]

MANUAL_GAIN_DROP_FEATURES: list[str] = []
