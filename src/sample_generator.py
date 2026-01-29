import pandas as pd
import math
import itertools as it
import numpy as np
from Solver_builder import *

np.set_printoptions(threshold=np.inf)


class TrainSetGenerator(object):
    """
    训练样本生成类，输出excel文件
    """
    def __init__(self):
        self.model_obj_coe = None
        self.model_valid_cut_pool = None
        self.model_condition = None
        self.model_opt_trigger = 0
        self.model_gap = 0

        self.generator = None
        self.model = None
        self.valid_cut_selection_permutation = None  # 二进制全排列组合

        # ================================= excel ============================
        self.xl_data = []  # 存储一个组合所有的信息

        self.xl_id = []
        self.xl_valid_selection = []
        self.xl_opt_trigger = []
        self.xl_random_seed = []
        self.xl_gap = []

        self.xl_customer_num = []
        self.xl_service_num = []
        self.xl_product_num = []
        self.xl_team_num = []

        self.xl_service_time = []
        self.xl_service_price = []
        self.xl_service_cost = []

        self.xl_profit = []
        self.xl_profit_mean = []
        self.xl_profit_std = []
        self.xl_profit_high4 = []
        self.xl_profit_low4 = []

        self.xl_service_time_lb = []
        self.xl_service_price_lb = []
        self.xl_service_cost_lb = []

        self.xl_service_time_step = []
        self.xl_service_price_step = []
        self.xl_service_cost_step = []

        self.xl_utility = []
        self.xl_utility_lb = []
        self.xl_utility_step = []
        self.xl_inventory = []
        self.xl_inventory_step = []
        self.xl_inventory_lb = []

        self.xl_Ser_Pro_corr = []
        self.xl_Ser_Pro_coe = []
        self.xl_Ser_Pro_lb = []
        self.xl_Ser_Pro_ub = []
        # mean
        self.xl_Ser_Pro_coe_mean = []
        # std
        self.xl_Ser_Pro_coe_std = []
        # high 4
        self.xl_Ser_Pro_coe_high4 = []

        self.xl_weight = []
        self.xl_weight_gap = []
        self.xl_weight_mean = []
        self.xl_weight_std = []
        self.xl_weight_high4 = []

        self.xl_early_start = []
        self.xl_early_start_lb = []
        self.xl_early_start_ub = []
        self.xl_late_start = []
        self.xl_late_start_lb = []
        self.xl_late_start_ub = []

        # mean
        self.xl_early_start_mean = []
        self.xl_late_start_mean = []
        # std
        self.xl_early_start_std = []
        self.xl_late_start_std = []
        # high4
        self.xl_early_start_high4 = []
        self.xl_late_start_high4 = []

        self.xl_transfer_matrix = []
        self.xl_transfer_lb = []
        self.xl_transfer_ub = []
        self.xl_transfer_mean = []
        self.xl_transfer_std = []
        self.xl_transfer_high4 = []

        self.xl_duration = []

        self.xl_fixed_cost = []
        self.xl_variable_cost = []
        self.xl_variable_cost_lb = []
        self.xl_variable_cost_ub = []
        self.xl_variable_cost_mean = []
        self.xl_variable_cost_std = []
        self.xl_variable_cost_high4 = []

        self.xl_cut3_percent = []
        # =======  results ======
        self.xl_obj1 = []  # 总性价比
        self.xl_obj2 = []  # 总任务开始时间和
        self.xl_obj3 = []  # 总利润
        self.xl_obj = []  # 函数目标值

        self.xl_runtime = []

    def update_paras(self, model_obj_coe, model_valid_cut_pool, model_opt_trigger,
                     model_gap, generator):
        """
        导入更新的模型参数
        :param model_obj_coe: ，目标函数的系数
        :param model_valid_cut_pool: valid cut 组合池
        :param model_opt_trigger: 选择求解类型
        :param model_gap: 如果选择gap类生成样本，模型停止求解的gap阈值
        :param generator: 参数生成类
        :return:
        """
        self.model_obj_coe = model_obj_coe
        self.model_valid_cut_pool = model_valid_cut_pool
        self.model_opt_trigger = model_opt_trigger
        self.model_gap = model_gap

        self.generator = generator

    def add_valid_cut(self, new_cut):
        """
        设计出新的valid cut，加入cut 池
        :param new_cut:
        :return:
        """
        new_id = len(self.model_valid_cut_pool)

        self.model_valid_cut_pool[new_id] = new_cut

    def check_valid_selection(self, valid_cut_selection):
        """
        检查模型是否存在错误： 服务等级与产品等级数量的不对等
        :param valid_cut_selection:
        :return:
        """
        if len(valid_cut_selection) != len(self.model_valid_cut_pool):
            print('有效不等式个数不匹配')
            # exit()

    def check_obj_len(self):
        """
        检查模型是否存在错误：目标函数系数与目标对应不上
        :return:
        """
        if len(self.model_obj_coe) != len(self.model.Model_Obj):
            print('目标个数与权重个数不匹配')
            exit()

    def generate_selection_permutation(self, valid_cut_pool):
        """
        生成有效切所有组合
        :param valid_cut_pool:
        :return:
        """
        # self.valid_cut_selection_permutation = list(it.product(range(2), repeat=len(valid_cut_pool)))
        self.valid_cut_selection_permutation = list(it.product(range(2), repeat=6))


    def build_optimize(self):
        """
        创建Gurobi模型，并且求解和记录
        :return:
        """
        cnt = 0  # iterate times

        # generate valid cut permutation
        self.generate_selection_permutation(valid_cut_pool=self.model_valid_cut_pool)
        # valid_selection=(0,0,0)
        # self.model = GRBModel()
        # self.model.set_data(obj_coe=self.model_obj_coe, valid_cut_pool=self.model_valid_cut_pool,
        #                     generator=self.generator, valid_selection=valid_selection)
        #
        # # create grb model
        # self.model.model_builder()
        #
        # # check obj
        # self.check_obj_len()
        #
        # # optimize
        # self.model.optimize_(trigger1=0, trigger2=1, trigger3=0)


        for valid_selection in self.valid_cut_selection_permutation:

            self.model = GRBModel()
            self.model.set_data(obj_coe=self.model_obj_coe, valid_cut_pool=self.model_valid_cut_pool,
                                generator=self.generator, valid_selection=valid_selection)

            self.model.valid_cut_cur = {i: [] for i in range(len(self.model_valid_cut_pool))}

            # check valid selection
            self.check_valid_selection(valid_selection)

            # create grb model
            self.model.model_builder()

            # check obj
            self.check_obj_len()

            # optimize
            # self.model.optimize_(trigger1=0,trigger2=0,trigger3=0)

            # print log
            # self.model.print_log()

            # optimize
            # self.model.optimize_(trigger1=1,trigger2=1,trigger3=0)

            # print log
            # self.model.print_log()

            # optimize
            # trigger_select=[[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
            # for trigger in trigger_select:
            #     # self.model.optimize_(trigger1=trigger, trigger2=[0, 0, 0], trigger3=[0, 0, 0],trigger4=[0,0,0,0,0,0])
            #     # self.model.optimize_(trigger1=[0, 0, 0], trigger2=trigger, trigger3=[0, 0, 0],trigger4=[0,0,0,0,0,0])
            #     self.model.optimize_(trigger1=[0, 0, 0], trigger2=[0, 0, 0], trigger3=trigger,trigger4=[0,0,0,0,0,0])
            #
            # trigger4_select = [[1, 0, 0,1, 0, 0], [1, 0, 0,0, 1, 0], [1, 0, 0,0, 0, 1], [1, 0, 0,1, 1, 1],
            #                    [0, 1, 0,1, 0, 0],[0, 1, 0,0, 1, 0],[0, 1, 0,0, 0, 1],[0, 1, 0,1, 1, 1],
            #                    [0, 0, 1,1, 0, 0],[0, 0, 1,0, 1, 0],[0, 0, 1,0, 0, 1],[0, 0, 1,1, 1, 1],
            #                    [1, 1, 1,1, 0, 0],[1, 1, 1,0, 1, 0],[1, 1, 1,0, 0, 1],[1, 1, 1,1, 1, 1]]
            # for trigger4 in  trigger4_select:
            #     self.model.optimize_(trigger1=[0, 0, 0], trigger2=[0, 0, 0], trigger3=[0,0,0],trigger4=trigger4)
            self.model.optimize_(trigger1=[0,0,0], trigger2=[0, 0, 0], trigger3=[0, 0, 0], trigger4=valid_selection)
            # print log
            # self.model.print_log()


            # 计算cut切掉的百分比
            cut3_num = 0
            cut3_cons_num = 0

            for k in range(self.generator.Num_Team):
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        if i != j:
                            cut3_cons_num += 1

                            # 注意 depot 0
                            if self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i + 1][
                                j + 1] + np.min(self.generator.Service_Time) > self.generator.Late_Start_Limit[j]:
                                cut3_num += 1
            current_xl_cut3_percent = cut3_num / cut3_cons_num

            # 记录当此求解数据
            current_data = {
                'id': cnt,
                'valid_selection': valid_selection,
                "opt_trigger": self.model_opt_trigger,
                'random_seed': self.generator.random_seed,
                'gap': self.model_gap,

                'customer_num': self.generator.Num_Customer,
                'service_num': self.generator.Num_Service,
                'product_num': self.generator.Num_Product,
                'team_num': self.generator.Num_Team,

                'service_time': self.generator.Service_Time,
                'service_price': self.generator.Service_Price,
                'service_cost': self.generator.Service_Cost,

                'profit': self.generator.Service_Price - self.generator.Service_Cost,
                'profit_mean': np.mean(self.generator.Service_Price - self.generator.Service_Cost),
                'profit_std': np.std(self.generator.Service_Price - self.generator.Service_Cost),
                'profit_high4': np.quantile(self.generator.Service_Price - self.generator.Service_Cost, 0.75),
                'profit_low4': np.quantile(self.generator.Service_Price - self.generator.Service_Cost, 0.25),

                'service_time_lb': self.generator.service_time_lb,
                'service_price_lb': self.generator.service_price_lb,
                'service_cost_lb': self.generator.service_cost_lb,

                'service_time_step': self.generator.service_time_step,
                'service_price_step': self.generator.service_price_step,
                'service_cost_step': self.generator.service_cost_step,

                'utility': self.generator.Utility,
                'utility_lb': self.generator.utility_lb,
                'utility_step': self.generator.utility_step,

                'inventory': self.generator.Inventory,
                'inventory_lb': self.generator.inventory_lb,
                'inventory_step': self.generator.inventory_step,

                'Ser_Pro_corr': self.generator.Ser_Pro_Corr,
                'Ser_Pro_coe': self.generator.ser_pro_coe,
                'ser_pro_coe_lb': self.generator.ser_pro_coe_lb,
                'ser_pro_coe_ub': self.generator.ser_pro_coe_lb,
                'Ser_Pro_coe_mean': np.mean(self.generator.Ser_Pro_Corr),
                'Ser_Pro_coe_std': np.std(self.generator.Ser_Pro_Corr),
                'Ser_Pro_coe_high4': np.quantile(self.generator.Ser_Pro_Corr, 0.75),

                'weight': self.generator.Weight,
                'weight_gap': self.generator.weight_gap,
                'weight_mean': np.mean(self.generator.Weight),
                'weight_std': np.std(self.generator.Weight),
                'weight_high4': np.quantile(self.generator.Weight, 0.75),

                'early_start': self.generator.Early_Start_Limit,
                'early_start_lb': self.generator.early_start_lb,
                'early_start_ub': self.generator.early_start_ub,
                'late_start': self.generator.Late_Start_Limit,
                'late_start_lb': self.generator.late_start_lb,
                'late_start_ub': self.generator.late_start_ub,
                'early_start_mean': np.mean(self.generator.Early_Start_Limit),
                'late_start_mean': np.mean(self.generator.Late_Start_Limit),
                'early_start_std': np.std(self.generator.Early_Start_Limit),
                'late_start_std': np.std(self.generator.Late_Start_Limit),
                'early_start_high4': np.quantile(self.generator.Early_Start_Limit, 0.75),
                'late_start_high4': np.quantile(self.generator.Late_Start_Limit, 0.75),

                'transfer_matrix': self.generator.Transfer_Time,
                'transfer_lb': self.generator.transfer_lb,
                'transfer_ub': self.generator.transfer_ub,
                'transfer_mean': np.mean(self.generator.Transfer_Time),
                'transfer_std': np.std(self.generator.Transfer_Time),
                'transfer_high4': np.quantile(self.generator.Transfer_Time, 0.75),

                'duration': self.generator.Duration,

                'FixedCost': self.generator.FixedCost,

                'VariableCost': self.generator.VariableCost,
                'variable_cost_lb': self.generator.variable_cost_lb,
                'variable_cost_ub': self.generator.variable_cost_ub,
                'variable_cost_mean': np.mean(self.generator.VariableCost),
                'variable_cost_std': np.std(self.generator.VariableCost),
                'variable_cost_high4': np.quantile(self.generator.VariableCost, 0.75),

                'cut3_percent': current_xl_cut3_percent,

                'obj1': self.model.Model_Obj[0].getValue(),
                'obj2': self.model.Model_Obj[1].getValue(),
                'obj3': self.model.Model_Obj[2].getValue(),
                'obj': self.model.model.ObjVal,

                'runtime': self.model.time_final
            }

            self.xl_data.append(current_data)
            cnt += 1

    def pd_to_excel(self, data):
        """
        将样本导出excel
        :param data:
        :return:
        """
        for i in range(len(data)):
            self.xl_id.append(data[i]["id"])

            self.xl_valid_selection.append(data[i]["valid_selection"])
            self.xl_opt_trigger.append(data[i]["opt_trigger"])
            self.xl_random_seed.append(data[i]["random_seed"])
            self.xl_gap.append(data[i]["gap"])

            self.xl_customer_num.append(data[i]["customer_num"])
            self.xl_service_num.append(data[i]["service_num"])
            self.xl_product_num.append(data[i]["product_num"])
            self.xl_team_num.append(data[i]["team_num"])

            self.xl_service_time.append(data[i]["service_time"])
            self.xl_service_price.append(data[i]["service_price"])
            self.xl_service_cost.append(data[i]["service_cost"])

            self.xl_profit.append(data[i]["profit"])
            self.xl_profit_mean.append(data[i]["profit_mean"])
            self.xl_profit_std.append(data[i]["profit_std"])
            self.xl_profit_high4.append(data[i]["profit_high4"])
            self.xl_profit_low4.append(data[i]["profit_low4"])

            self.xl_service_time_lb.append(data[i]["service_time_lb"])
            self.xl_service_price_lb.append(data[i]["service_price_lb"])
            self.xl_service_cost_lb.append(data[i]["service_cost_lb"])

            self.xl_service_time_step.append(data[i]["service_time_step"])
            self.xl_service_price_step.append(data[i]["service_price_step"])
            self.xl_service_cost_step.append(data[i]["service_cost_step"])

            self.xl_utility.append(data[i]["utility"])
            self.xl_utility_lb.append(data[i]["utility_lb"])
            self.xl_utility_step.append(data[i]["utility_step"])
            self.xl_inventory.append(data[i]["inventory"])
            self.xl_inventory_step.append(data[i]["inventory_step"])
            self.xl_inventory_lb.append(data[i]["inventory_lb"])

            self.xl_Ser_Pro_corr.append(data[i]["Ser_Pro_corr"])
            self.xl_Ser_Pro_coe.append(data[i]["Ser_Pro_coe"])
            self.xl_Ser_Pro_lb.append(data[i]["ser_pro_coe_lb"])
            self.xl_Ser_Pro_ub.append(data[i]["ser_pro_coe_ub"])
            # mean
            self.xl_Ser_Pro_coe_mean.append(data[i]["Ser_Pro_coe_mean"])
            # std
            self.xl_Ser_Pro_coe_std.append(data[i]["Ser_Pro_coe_std"])
            # high 4
            self.xl_Ser_Pro_coe_high4.append(data[i]["Ser_Pro_coe_high4"])

            self.xl_weight.append(data[i]["weight"])
            self.xl_weight_gap.append(data[i]["weight_gap"])
            self.xl_weight_mean.append(data[i]["weight_mean"])
            self.xl_weight_std.append(data[i]["weight_std"])
            self.xl_weight_high4.append(data[i]["weight_high4"])

            self.xl_early_start.append(data[i]["early_start"])
            self.xl_early_start_lb.append(data[i]["early_start_lb"])
            self.xl_early_start_ub.append(data[i]["early_start_ub"])
            self.xl_late_start.append(data[i]["late_start"])
            self.xl_late_start_lb.append(data[i]["late_start_lb"])
            self.xl_late_start_ub.append(data[i]["late_start_ub"])

            # mean
            self.xl_early_start_mean.append(data[i]["early_start_mean"])
            self.xl_late_start_mean.append(data[i]["late_start_mean"])
            # std
            self.xl_early_start_std.append(data[i]["early_start_std"])
            self.xl_late_start_std.append(data[i]["late_start_std"])
            # high4
            self.xl_early_start_high4.append(data[i]["early_start_high4"])
            self.xl_late_start_high4.append(data[i]["late_start_high4"])

            self.xl_transfer_matrix.append(data[i]["transfer_matrix"])
            self.xl_transfer_lb.append(data[i]["transfer_lb"])
            self.xl_transfer_ub.append(data[i]["transfer_ub"])
            self.xl_transfer_mean.append(data[i]["transfer_mean"])
            self.xl_transfer_std.append(data[i]["transfer_std"])
            self.xl_transfer_high4.append(data[i]["transfer_high4"])

            self.xl_duration.append(data[i]["duration"])

            self.xl_fixed_cost.append(data[i]["FixedCost"])
            self.xl_variable_cost.append(data[i]["VariableCost"])
            self.xl_variable_cost_lb.append(data[i]["variable_cost_lb"])
            self.xl_variable_cost_ub.append(data[i]["variable_cost_ub"])
            self.xl_variable_cost_mean.append(data[i]["variable_cost_mean"])
            self.xl_variable_cost_std.append(data[i]["variable_cost_std"])
            self.xl_variable_cost_high4.append(data[i]["variable_cost_high4"])

            self.xl_cut3_percent.append(data[i]["cut3_percent"])
            # =======  results ======
            self.xl_obj1.append(data[i]["obj1"])  # 总性价比
            self.xl_obj2.append(data[i]["obj2"])  # 总任务开始时间和
            self.xl_obj3.append(data[i]["obj3"])  # 总利润
            self.xl_obj.append(data[i]["obj"])  # 函数目标值

            self.xl_runtime.append(data[i]["runtime"])

        df_data = {
            '序号': self.xl_id,

            '有效不等式选择': self.xl_valid_selection,
            '求解选择': self.xl_opt_trigger,
            'random_seed': self.xl_random_seed,
            'gap': self.xl_gap,

            '客户数': self.xl_customer_num,
            '服务数': self.xl_service_num.append,
            '产品数': self.xl_product_num,
            '服务团队数': self.xl_team_num,

            '服务时间': self.xl_service_time,
            '服务价格': self.xl_service_price,
            '服务成本': self.xl_service_cost,

            '利润': self.xl_profit,
            '利润均值': self.xl_profit_mean,
            '利润标准差': self.xl_profit_std,
            '利润上四分位': self.xl_profit_high4,
            '利润下四分位': self.xl_profit_low4,

            '服务时间下界': self.xl_service_time_lb,
            '服务价格下界': self.xl_service_price_lb,
            '服务成本下界': self.xl_service_cost_lb,

            '服务时间梯度': self.xl_service_time_step,
            '服务价格梯度': self.xl_service_price_step,
            '服务成本梯度': self.xl_service_cost_step,

            '客户效用值': self.xl_utility,
            '客户效用值下界': self.xl_utility_lb,
            '客户效用值梯度': self.xl_utility_step,
            '库存': self.xl_inventory,
            '库存梯度': self.xl_inventory_step,
            '库存下界': self.xl_inventory_lb,

            '产品服务对应关系矩阵': self.xl_Ser_Pro_corr,
            '产品服务对应关系': self.xl_Ser_Pro_coe,
            '产品服务对应关系下界': self.xl_Ser_Pro_lb,
            '产品服务对应关系上界': self.xl_Ser_Pro_ub,
            # mean
            '产品服务对应关系均值': self.xl_Ser_Pro_coe_mean,
            # std
            '产品服务对应关系标准差': self.xl_Ser_Pro_coe_std,
            # high 4
            '产品服务对应关系上四分位': self.xl_Ser_Pro_coe_high4,

            '客户权重': self.xl_weight,
            '客户权重gap': self.xl_weight_gap,
            '客户权重均值': self.xl_weight_mean,
            '客户权重标准差': self.xl_weight_std,
            '客户权重上四分位': self.xl_weight_high4,

            '服务时间窗Early': self.xl_early_start,
            '服务时间窗Early下界': self.xl_early_start_lb,
            '服务时间窗Early上界': self.xl_early_start_ub,
            '服务时间窗Late': self.xl_late_start,
            '服务时间窗Late下界': self.xl_late_start_lb,
            '服务时间窗Late上界': self.xl_late_start_ub,

            # mean
            '服务时间窗Early均值': self.xl_early_start_mean,
            '服务时间窗Late均值': self.xl_late_start_mean,
            # std
            '服务时间窗Early标准差': self.xl_early_start_std,
            '服务时间窗Late标准差': self.xl_late_start_std,
            # high4
            '服务时间窗Early上四分位': self.xl_early_start_high4,
            '服务时间窗Late上四分位': self.xl_late_start_high4,

            '转移时间矩阵': self.xl_transfer_matrix,
            '转移时间矩阵下界': self.xl_transfer_lb,
            '转移时间矩阵上界': self.xl_transfer_ub,
            '转移时间矩阵均值': self.xl_transfer_mean,
            '转移时间矩阵标准差': self.xl_transfer_std,
            '转移时间矩阵上四分位': self.xl_transfer_high4,

            '工作时长': self.xl_duration,

            '固定成本': self.xl_fixed_cost,
            '可变成本': self.xl_variable_cost,
            '可变成本下界': self.xl_variable_cost_lb,
            '可变成本上界': self.xl_variable_cost_ub,
            '可变成本均值': self.xl_variable_cost_mean,
            '可变成本标准差': self.xl_variable_cost_std,
            '可变成本上四分位': self.xl_variable_cost_high4,

            'cut3百分比': self.xl_cut3_percent,
            # =======  results ======
            '总性价比': self.xl_obj1,
            '总任务开始时间和': self.xl_obj2,
            '总利润': self.xl_obj3,
            '函数目标值': self.xl_obj,

            '求解时间': self.xl_runtime
        }
        df = pd.DataFrame(df_data)
        # df.to_excel(filename, index=False)
        df.sort_values(by="求解时间", inplace=True, ascending=True)
        # index_ = df.iloc[0][0]

        # df.iloc[0, 54] = 1
        # self.xl_data[df.iloc[0, 0]]['target'] = 1
        print()
        # _index = df.iloc[0][0]    # 提取排序后的最优行

        return df