from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .Pamas_generator import Generator
from .sample_generator import TrainSetGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

OBJ_COEF = [1, -1, 1, 0]
TOTAL_TIMES = 400

RESULT_COLUMN_MAP = [
    ("id", "序号"),
    ("valid_selection", "有效不等式选择"),
    ("size_signature", "size_signature"),
    ("base_num_vars", "base_num_vars"),
    ("base_num_constrs", "base_num_constrs"),
    ("full_runtime", "full_runtime"),
    ("root_num_constrs", "root_num_constrs"),
    ("runtime", "求解时间"),
    ("legacy_runtime", "legacy_runtime"),
    ("customer_num", "客户数"),
    ("service_num", "服务数"),
    ("product_num", "产品数"),
    ("team_num", "服务团队数"),
    ("scale_phase", "scale_phase"),
    ("scale_stage", "scale_stage"),
    ("opt_trigger", "求解选择"),
    ("random_seed", "random_seed"),
    ("gap", "gap"),
    ("service_time", "服务时间"),
    ("service_price", "服务价格"),
    ("service_cost", "服务成本"),
    ("profit", "利润"),
    ("profit_mean", "利润均值"),
    ("profit_std", "利润标准差"),
    ("profit_high4", "利润上四分位"),
    ("profit_low4", "利润下四分位"),
    ("service_time_lb", "服务时间下界"),
    ("service_price_lb", "服务价格下界"),
    ("service_cost_lb", "服务成本下界"),
    ("service_time_step", "服务时间梯度"),
    ("service_price_step", "服务价格梯度"),
    ("service_cost_step", "服务成本梯度"),
    ("utility", "客户效用值"),
    ("utility_lb", "客户效用值下界"),
    ("utility_step", "客户效用值梯度"),
    ("inventory", "库存"),
    ("inventory_step", "库存梯度"),
    ("inventory_lb", "库存下界"),
    ("Ser_Pro_corr", "产品服务对应关系矩阵"),
    ("Ser_Pro_coe", "产品服务对应关系"),
    ("ser_pro_coe_lb", "产品服务对应关系下界"),
    ("ser_pro_coe_ub", "产品服务对应关系上界"),
    ("Ser_Pro_coe_mean", "产品服务对应关系均值"),
    ("Ser_Pro_coe_std", "产品服务对应关系标准差"),
    ("Ser_Pro_coe_high4", "产品服务对应关系上四分位"),
    ("weight", "客户权重"),
    ("weight_gap", "客户权重gap"),
    ("weight_mean", "客户权重均值"),
    ("weight_std", "客户权重标准差"),
    ("weight_high4", "客户权重上四分位"),
    ("early_start", "服务时间窗Early"),
    ("early_start_lb", "服务时间窗Early下界"),
    ("early_start_ub", "服务时间窗Early上界"),
    ("late_start", "服务时间窗Late"),
    ("late_start_lb", "服务时间窗Late下界"),
    ("late_start_ub", "服务时间窗Late上界"),
    ("early_start_mean", "服务时间窗Early均值"),
    ("late_start_mean", "服务时间窗Late均值"),
    ("early_start_std", "服务时间窗Early标准差"),
    ("late_start_std", "服务时间窗Late标准差"),
    ("early_start_high4", "服务时间窗Early上四分位"),
    ("late_start_high4", "服务时间窗Late上四分位"),
    ("transfer_matrix", "转移时间矩阵"),
    ("transfer_lb", "转移时间矩阵下界"),
    ("transfer_ub", "转移时间矩阵上界"),
    ("transfer_mean", "转移时间矩阵均值"),
    ("transfer_std", "转移时间矩阵标准差"),
    ("transfer_high4", "转移时间矩阵上四分位"),
    ("duration", "工作时长"),
    ("FixedCost", "固定成本"),
    ("VariableCost", "可变成本"),
    ("variable_cost_lb", "可变成本下界"),
    ("variable_cost_ub", "可变成本上界"),
    ("variable_cost_mean", "可变成本均值"),
    ("variable_cost_std", "可变成本标准差"),
    ("variable_cost_high4", "可变成本上四分位"),
    ("cut3_percent", "cut3百分比"),
    ("obj1", "总性价比"),
    ("obj2", "总任务开始时间和"),
    ("obj3", "总利润"),
    ("obj", "函数目标值"),
]


def sample_instance_size(times: int) -> tuple[int, int, int, int, str, str, str]:
    """按实验阶段生成实例规模。"""

    if times <= 300:
        num_customer = np.random.randint(5, 10)
        num_service = np.random.randint(3, 6)
        num_team = np.random.randint(3, 8)
        scale_phase = "baseline_300"
        scale_stage = "S1"
    elif times <= 325:
        num_customer = np.random.randint(10, 13)
        num_service = np.random.randint(5, 7)
        num_team = np.random.randint(6, 9)
        scale_phase = "stress_100"
        scale_stage = "S2"
    elif times <= 350:
        num_customer = np.random.randint(12, 15)
        num_service = np.random.randint(6, 8)
        num_team = np.random.randint(7, 10)
        scale_phase = "stress_100"
        scale_stage = "S3"
    elif times <= 375:
        num_customer = np.random.randint(14, 17)
        num_service = np.random.randint(7, 9)
        num_team = np.random.randint(8, 11)
        scale_phase = "stress_100"
        scale_stage = "S4"
    else:
        num_customer = np.random.randint(16, 19)
        num_service = np.random.randint(8, 11)
        num_team = np.random.randint(9, 13)
        scale_phase = "stress_100"
        scale_stage = "S5"

    num_product = num_service
    size_signature = f"I{num_customer}_M{num_service}_N{num_product}_K{num_team}"
    return num_customer, num_service, num_product, num_team, scale_phase, scale_stage, size_signature


def build_valid_cut_pool() -> dict[int, str]:
    valid_cut_pool = {}
    valid_cut_pool[
        0
    ] = "self.model.addConstrs(self.X[i, j, k] + self.X[j, i, k] <= 1 for i in range(self.generator.Num_Customer) for j in range(self.generator.Num_Customer) for k in range(self.generator.Num_Team),name=cut_0)"
    valid_cut_pool[
        1
    ] = "self.model.addConstrs(self.T[i, k] + sum(self.R[i,m]*self.generator.Service_Time[m] for m in range(self.generator.Num_Service)) + self.generator.Transfer_Time[i][j] - self.BigM * (1 - self.X[i, j, k]) <= self.generator.Late_Start_Limit[j] for i in range(self.generator.Num_Customer) for j in range(self.generator.Num_Customer) for k in range(self.generator.Num_Team))"
    valid_cut_pool[
        2
    ] = "self.model.addConstrs(self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i][j] + min(self.generator.Service_Time) - self.BigM * (1 - self.X[i, j, k]) <= self.generator.Late_Start_Limit[j] for i in range(self.generator.Num_Customer) for j in range(self.generator.Num_Customer) for k in range(self.generator.Num_Team))"
    return valid_cut_pool


def generate_random_parameter_ranges() -> dict[str, int]:
    return {
        "service_time_step": np.random.randint(2, 6),
        "service_price_step": np.random.randint(5, 12),
        "service_cost_step": np.random.randint(3, 7),
        "service_time_lb": np.random.randint(1, 7),
        "service_price_lb": np.random.randint(10, 25),
        "service_cost_lb": np.random.randint(1, 7),
        "early_start_lb": np.random.randint(0, 5),
        "early_start_ub": np.random.randint(5, 10),
        "late_start_lb": np.random.randint(10, 15),
        "late_start_ub": np.random.randint(15, 35),
        "transfer_lb": np.random.randint(1, 3),
        "transfer_ub": np.random.randint(3, 6),
        "utility_lb": np.random.randint(4, 12),
        "utility_step": np.random.randint(1, 4),
        "inventory_step": np.random.randint(10, 16),
        "inventory_lb": np.random.randint(10, 20),
        "weight_tri": 1,
        "weight_gap": np.random.randint(2, 6),
        "variable_cost_lb": np.random.randint(0, 2),
        "variable_cost_ub": np.random.randint(2, 3),
        "ser_pro_coe": np.random.randint(1, 20),
        "ser_pro_coe_lb": np.random.randint(1, 2),
        "ser_pro_coe_ub": np.random.randint(2, 4),
    }


def build_result_dataframe(rows: list[dict]) -> pd.DataFrame:
    data = {column_name: [row[key] for row in rows] for key, column_name in RESULT_COLUMN_MAP}
    return pd.DataFrame(data)


def build_result_csv_path(run_date: date | None = None) -> Path:
    current_date = run_date or date.today()
    return DATA_DIR / f"result_{current_date:%Y%m%d}.csv"


def run_experiment(*, total_times: int = TOTAL_TIMES) -> None:
    gen = Generator()
    valid_cut_pool = build_valid_cut_pool()
    result_csv_path = build_result_csv_path()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    next_row_id = 1

    # Each run starts from a fresh dated result file instead of appending to prior results.
    if result_csv_path.exists():
        result_csv_path.unlink()

    print(f"Results will be written to: {result_csv_path}")

    times = 1
    while times <= total_times:
        (
            num_customer,
            num_service,
            num_product,
            num_team,
            scale_phase,
            scale_stage,
            size_signature,
        ) = sample_instance_size(times)

        random_seed = np.random.randint(low=10, high=10000)
        parameter_ranges = generate_random_parameter_ranges()

        gen.set_instance(
            num_customer=num_customer,
            num_service=num_service,
            num_product=num_product,
            num_team=num_team,
            random_seed=random_seed,
        )
        gen.set_params(**parameter_ranges)

        gen.gen_customer_weight(weight_tri=gen.weight_tri, weight_gap=gen.weight_gap)
        gen.gen_service_relations()
        gen.gen_ser_pro_coe()
        gen.gen_customer_relations()
        gen.gen_time_relations()
        gen.gen_dis_matrix()
        gen.gen_cost_relations()

        train_set_generator = TrainSetGenerator()
        train_set_generator.update_paras(
            model_obj_coe=OBJ_COEF,
            model_valid_cut_pool=valid_cut_pool,
            model_opt_trigger=1,
            model_gap=0,
            generator=gen,
            instance_id=times,
            row_id_start=next_row_id,
            scale_phase=scale_phase,
            scale_stage=scale_stage,
            size_signature=size_signature,
        )
        train_set_generator.build_optimize()
        result_df = build_result_dataframe(train_set_generator.xl_data)
        next_row_id += len(train_set_generator.xl_data)
        write_mode = "w" if times == 1 else "a"
        write_header = times == 1
        result_df.to_csv(result_csv_path, index=False, mode=write_mode, header=write_header)

        print(f"-----------------------------------------------times = {times}")
        times += 1


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
