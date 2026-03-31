import itertools as it
from gurobipy import *
from gurobipy import quicksum
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_LOG_PATH = DATA_DIR / "实验记录case3.txt"


class GRBModel(object):
    """
    Gurobi 模型生成器
    """
    def __init__(self):
        self.obj_coe = None
        self.gap = 0
        self.valid_selection = None
        self.valid_cut_pool = None
        self.valid_cut_cur = {}
        # self.valid_cut_selection_permutation = None

        self.BigM = 1000

        self.generator = None

        self.model = None
        self.Model_Obj = {}
        self.base_num_vars = 0
        self.base_num_constrs = 0
        self.root_num_constrs = 0

        # 决策变量
        self.R = {}
        self.X = {}
        self.Y = {}
        self.Z = {}
        self.T = {}

    def set_data(self, obj_coe, valid_cut_pool, generator, valid_selection):
        """
        设置模型的参数
        :param obj_coe: 目标函数的系数  e.g. [1, 1, 1, 1]
        :param valid_cut_pool: valid cut
        :param generator: 参数生成器， class Pamas_generator
        :param valid_selection: valid cut 组合选择
        :return:
        """
        import numpy as np

        self.obj_coe = obj_coe
        self.valid_cut_pool = valid_cut_pool
        self.generator = generator
        self.BigM = np.max(self.generator.Service_Time) * 100
        self.valid_selection = valid_selection

    def model_builder(self):
        """
        创建 Gurobi 模型
        :return:
        """
        self.model = Model()

        # 添加决策变量
        self.flag=0
        for i in range(self.generator.Num_Customer + 2):
            for m in range(self.generator.Num_Service):
                self.R[i, m] = self.model.addVar(vtype=GRB.BINARY, name="r" + "_" + str(i) + "_" + str(m))

        for k in range(self.generator.Num_Team):
            self.Z[k] = self.model.addVar(vtype=GRB.BINARY, name="z" + "_" + str(k))
            for i in range(self.generator.Num_Customer + 2):
                self.Y[i, k] = self.model.addVar(vtype=GRB.BINARY, name="y" + "_" + str(i) + "_" + str(k))
                self.T[i, k] = self.model.addVar(vtype=GRB.INTEGER, name="t" + "_" + str(i) + "_" + str(k))
                for j in range(self.generator.Num_Customer + 2):
                    self.X[i, j, k] = self.model.addVar(vtype=GRB.BINARY,
                                                        name="x" + "_" + str(i) + "_" + str(j) + "_" + str(k))

        # 目标函数
        self.Model_Obj[0] = quicksum(
            self.R[i, m] * self.generator.Utility[i][m] / self.generator.Service_Price[m] * self.generator.Weight[i] for
            i in range(self.generator.Num_Customer) for m in
            range(self.generator.Num_Service))
        self.Model_Obj[1] = quicksum(
            self.T[i, k] * self.generator.Weight[i] for i in range(self.generator.Num_Customer) for k in
            range(self.generator.Num_Team))
        self.Model_Obj[2] = quicksum(
            self.R[i, m] * (self.generator.Service_Price[m] - self.generator.Service_Cost[m]) for i in
            range(self.generator.Num_Customer) for m in
            range(self.generator.Num_Service)) - quicksum(
            self.Z[k] * self.generator.FixedCost for k in range(self.generator.Num_Team)) - quicksum(
            self.X[i, j, k] * self.generator.VariableCost[i][j] for i in range(self.generator.Num_Customer + 2) for j in
            range(self.generator.Num_Customer + 2) for k in
            range(self.generator.Num_Team))
        self.Model_Obj[3] = quicksum(self.T[self.generator.Num_Customer + 1, k] for k in range(self.generator.Num_Team))

        self.model.setObjective(quicksum(self.obj_coe[i] * self.Model_Obj[i] for i in range(len(self.obj_coe))),
                                GRB.MAXIMIZE)

        # 约束
        # 1.服务选配要求
        self.model.addConstrs(
            quicksum(self.R[i, m] for m in range(self.generator.Num_Service)) <= 1 for i in
            range(self.generator.Num_Customer))
        self.model.addConstr(quicksum(
            self.R[self.generator.Num_Customer, m] + self.R[self.generator.Num_Customer + 1, m] for m in
            range(self.generator.Num_Service)) == 0)

        # 2. 产品库存限制
        self.model.addConstrs(
            quicksum(
                self.R[i, m] * self.generator.Ser_Pro_Corr[m][n] for i in range(self.generator.Num_Customer) for m in
                range(self.generator.Num_Service)) <=
            self.generator.Inventory[n] for
            n in range(self.generator.Num_Product))

        # 3. 服务时长限制
        # self.model.addConstrs(
        #     self.T[self.generator.Num_Customer + 1, k] <= self.generator.Duration[k] * self.Z[k] for k in
        #     range(self.generator.Num_Team))
        self.model.addConstrs(
            self.T[self.generator.Num_Customer + 1, k] <= self.generator.Duration * self.Z[k] for k in
            range(self.generator.Num_Team))

        # 4. 履约路径流平衡约束
        self.model.addConstrs(quicksum(
            self.X[self.generator.Num_Customer, j, k] for j in range(self.generator.Num_Customer + 2)) - quicksum(
            self.X[j, self.generator.Num_Customer, k] for j in range(self.generator.Num_Customer + 2)) == self.Z[k] for
                              k in range(self.generator.Num_Team))
        self.model.addConstrs(quicksum(
            self.X[j, self.generator.Num_Customer + 1, k] for j in range(self.generator.Num_Customer + 2)) - quicksum(
            self.X[self.generator.Num_Customer + 1, j, k] for j in range(self.generator.Num_Customer + 2)) == self.Z[k]
                              for k in range(self.generator.Num_Team))
        self.model.addConstrs(
            quicksum(self.X[i, j, k] for i in range(self.generator.Num_Customer + 2)) - quicksum(
                self.X[j, i, k] for i in range(self.generator.Num_Customer + 2)) == 0
            for k in range(self.generator.Num_Team) for j in range(self.generator.Num_Customer))

        # 5. 履约路径连续性约束
        self.model.addConstrs(
            self.T[i, k] + self.generator.Transfer_Time[i][j] + quicksum(
                self.R[i, m] * self.generator.Service_Time[m] for m in range(self.generator.Num_Service)) - self.T[
                j, k] <= self.BigM * (1 - self.X[i, j, k]) for i in range(self.generator.Num_Customer + 2) for j in
            range(self.generator.Num_Customer + 2) for k
            in
            range(self.generator.Num_Team))

        # 6. 服务开始时间窗约束
        self.model.addConstrs(
            self.T[i, k] >= self.generator.Early_Start_Limit[i] - (1 - self.Y[i, k]) * self.BigM for i in
            range(self.generator.Num_Customer) for k in
            range(self.generator.Num_Team))
        self.model.addConstrs(
            self.T[i, k] <= self.generator.Late_Start_Limit[i] + (1 - self.Y[i, k]) * self.BigM for i in
            range(self.generator.Num_Customer) for k in range(self.generator.Num_Team))

        # 7. 变量合理性约束
        self.model.addConstrs(
            quicksum(self.Y[i, k] for k in range(self.generator.Num_Team)) == quicksum(
                self.R[i, m] for m in range(self.generator.Num_Service)) for i in
            range(self.generator.Num_Customer + 2))
        self.model.addConstr(quicksum(self.X[i, i, k] for i in range(self.generator.Num_Customer + 2) for k in
                                      range(self.generator.Num_Team)) == 0)
        self.model.addConstr(
            quicksum(self.X[self.generator.Num_Customer, self.generator.Num_Customer + 1, k] + self.X[
                self.generator.Num_Customer + 1, self.generator.Num_Customer, k] for k in
                     range(self.generator.Num_Team)) == 0)
        self.model.addConstrs(
            self.Y[i, k] == quicksum(self.X[j, i, k] for j in range(self.generator.Num_Customer + 2)) for i in
            range(self.generator.Num_Customer) for k in
            range(self.generator.Num_Team))
        self.model.addConstrs(
            self.Z[k] >= quicksum(self.Y[i, k] for i in range(self.generator.Num_Customer)) / (
                    self.generator.Num_Customer + 3) for k in range(self.generator.Num_Team))
        self.model.addConstrs(
            self.Z[k] <= quicksum(self.Y[i, k] for i in range(self.generator.Num_Customer)) * (
                    self.generator.Num_Customer + 3) for k in range(self.generator.Num_Team))
        self.model.addConstr(
            quicksum(self.T[self.generator.Num_Customer, k] for k in range(self.generator.Num_Team)) == 0)

        # Valid Cut
        # self.valid_cut_cur = {i: [] for i in range(len(self.valid_cut_pool))}
        # for cut_id in range(len(self.valid_selection)):
        #     if self.valid_selection[cut_id] == 1:
        #         self.valid_cut_cur[cut_id].append(eval(self.valid_cut_pool[cut_id], locals()))
        # 初始化一个字典用于存储约束
        # 添加约束并存储到字典中
        self.model.update()
        self.base_num_vars = self.model.NumVars
        self.base_num_constrs = self.model.NumConstrs
        self.root_num_constrs = self.base_num_constrs


    def optimize_(self, trigger1,trigger2,trigger3,trigger4):
        """
        求解模型
        :param trigger:
        :return:
        """

        def my_callback(model, where):
            if where == GRB.Callback.MIPNODE :
                # 获取当前节点的信息
                node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                objbound = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                objbnd= model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                cuent_gap=round(abs(objbnd-objbound)/objbnd,2)
                # print(cuent_gap)
                # 在这里添加 CB cut 的逻辑，示例中添加了一个简单的 cut
                if cuent_gap<=0.15 and self.flag==0:
                    for i in range(self.generator.Num_Customer):
                        for j in range(self.generator.Num_Customer):
                            for k in range(self.generator.Num_Team):
                                # 添加 CB cut
                                # cut_expr = model.getVarByName(f'x_{i}_{j}_{k}') + model.getVarByName(f'x_{j}_{i}_{k}')
                                # model.cbCut(cut_expr<= 1)
                                if trigger3[0]==1:
                                    model.cbCut(self.X[i,j,k]+self.X[j,i,k]<=1)
                                if trigger3[1]==1:
                                    expr = self.T[i, k] + sum(
                                                        self.R[i, m] * self.generator.Service_Time[m] for m in range(self.generator.Num_Service)
                                                        )+self.generator.Transfer_Time[i][j] - self.BigM * (1 - self.X[i, j, k])
                                    model.cbCut(expr <= self.generator.Late_Start_Limit[j])
                                if trigger3[2]==1:
                                    expr = self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i][
                                        j] + min(self.generator.Service_Time) - self.BigM * (1 - self.X[i, j, k])
                                    model.cbCut(expr <= self.generator.Late_Start_Limit[j])

                    # for i in range(self.generator.Num_Customer):
                    #     for j in range(self.generator.Num_Customer):
                    #         for k in range(self.generator.Num_Team):
                    #             expr = self.T[i, k] + sum(
                    #                     self.R[i, m] * self.generator.Service_Time[m] for m in range(self.generator.Num_Service)
                    #                     )+self.generator.Transfer_Time[i][j] - self.BigM * (1 - self.X[i, j, k])
                    #             # expr = model.getVarByName(f't_{i}_{k}') + quicksum(
                    #             #     model.getVarByName(f'r_{i}_{m}') * self.generator.Service_Time[m] for m in
                    #             #     range(self.generator.Num_Service)
                    #             # ) + self.generator.Transfer_Time[i][j] - self.BigM * (1 - model.getVarByName(f'x_{i}_{j}_{k}'))
                    #             model.cbCut(expr <= self.generator.Late_Start_Limit[j])
                    #
                    # for i in range(self.generator.Num_Customer):
                    #     for j in range(self.generator.Num_Customer):
                    #         for k in range(self.generator.Num_Team):
                    #             # 构建约束表达式
                    #             expr = self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i][j] + min(
                    #                     self.generator.Service_Time) - self.BigM * (1 - self.X[i, j, k])
                    #             # expr = self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i][j] + min(
                    #             #     self.generator.Service_Time) - self.BigM * (1 - model.getVarByName(f'x_{i}_{j}_{k}'))
                    #             model.cbCut( expr <= self.generator.Late_Start_Limit[j])
                    self.flag=1
        def my_callback_1(model, where):
            if where == GRB.Callback.MIPNODE :
                # 获取当前节点的信息
                node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                objbound = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                objbnd= model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                cuent_gap=round(abs(objbnd-objbound)/objbnd,2)
                # print(cuent_gap)
                # 在这里添加 CB cut 的逻辑，示例中添加了一个简单的 cut
                if cuent_gap<=0.15 and self.flag==0:
                    for i in range(self.generator.Num_Customer):
                        for j in range(self.generator.Num_Customer):
                            for k in range(self.generator.Num_Team):
                                # 添加 CB cut
                                # cut_expr = model.getVarByName(f'x_{i}_{j}_{k}') + model.getVarByName(f'x_{j}_{i}_{k}')
                                # model.cbCut(cut_expr<= 1)
                                if trigger4[3]==1:
                                    model.cbCut(self.X[i,j,k]+self.X[j,i,k]<=1)
                                if trigger4[4]==1:
                                    expr = self.T[i, k] + sum(
                                                        self.R[i, m] * self.generator.Service_Time[m] for m in range(self.generator.Num_Service)
                                                        )+self.generator.Transfer_Time[i][j] - self.BigM * (1 - self.X[i, j, k])
                                    model.cbCut(expr <= self.generator.Late_Start_Limit[j])
                                if trigger4[5]==1:
                                    expr = self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i][
                                        j] + min(self.generator.Service_Time) - self.BigM * (1 - self.X[i, j, k])
                                    model.cbCut(expr <= self.generator.Late_Start_Limit[j])
                    self.flag = 1



        # self.model.Params.Heuristics = 0  # 禁用所有启发式算法
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path = EXPERIMENT_LOG_PATH
        #全局优化
        if trigger1 != [0,0,0] and trigger1 != []:
            if trigger1[0]==1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            constraint_expr = self.X[i, j, k] + self.X[j, i, k] <= 1
                            constraint_name = f"cut_0_{i}_{j}_{k}"
                            self.model.addConstr(constraint_expr, name=constraint_name)

            if trigger1[1] == 1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            # 构建约束表达式
                            expr = self.T[i, k] + sum(
                                self.R[i, m] * self.generator.Service_Time[m] for m in range(self.generator.Num_Service)) + \
                                   self.generator.Transfer_Time[i][j] - self.BigM * (1 - self.X[i, j, k])
                            # 构建约束名称，可以根据需要进行修改
                            constr_name = f'cut_1_{i}_{j}_{k}'
                            # 将约束添加到字典中
                            self.model.addConstr(expr <= self.generator.Late_Start_Limit[j], name=constr_name)

            if trigger1[2] == 1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            # 构建约束表达式
                            expr = self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i][j] + min(
                                self.generator.Service_Time) - self.BigM * (1 - self.X[i, j, k])
                            # 构建约束名称，可以根据需要进行修改
                            constr_name = f'cut_2_{i}_{j}_{k}'
                            # 将约束添加到字典中
                            self.model.addConstr(expr <= self.generator.Late_Start_Limit[j], name=constr_name)

            self.model.optimize()
            time1 = self.model.runtime
            print(f"trigger1={trigger1}", end=' ')
            print(time1)
            with open(file_path, "a") as file:
                data_to_append = f"trigger1={trigger1}:{time1}"
                file.write(data_to_append + "\n")
            # self.model.reset()
        if trigger1 == []:
            self.model.optimize()
            time1 = self.model.runtime
            print(f"trigger1={trigger1}", end=' ')
            print(time1)
            with open(file_path, "a") as file:
                data_to_append = f"trigger1={trigger1}:{time1}"
                file.write(data_to_append + "\n")
            # self.model.reset()

        # 根节点加cut，子节点不加cut
        if trigger2 != [0,0,0]:
            if trigger2[0]==1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            constraint_expr = self.X[i, j, k] + self.X[j, i, k] <= 1
                            constraint_name = f"cut_0_{i}_{j}_{k}"
                            self.model.addConstr(constraint_expr, name=constraint_name)

            if trigger2[1]==1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            # 构建约束表达式
                            expr = self.T[i, k] + sum(
                                self.R[i, m] * self.generator.Service_Time[m] for m in range(self.generator.Num_Service)) + \
                                   self.generator.Transfer_Time[i][j] - self.BigM * (1 - self.X[i, j, k])
                            # 构建约束名称，可以根据需要进行修改
                            constr_name = f'cut_1_{i}_{j}_{k}'
                            # 将约束添加到字典中
                            self.model.addConstr(expr <= self.generator.Late_Start_Limit[j], name=constr_name)

            if trigger2[2] == 1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            # 构建约束表达式
                            expr = self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i][j] + min(
                                self.generator.Service_Time) - self.BigM * (1 - self.X[i, j, k])
                            # 构建约束名称，可以根据需要进行修改
                            constr_name = f'cut_2_{i}_{j}_{k}'
                            # 将约束添加到字典中
                            self.model.addConstr(expr <= self.generator.Late_Start_Limit[j], name=constr_name)

            self.model.Params.NodeLimit = 1
            self.model.optimize()
            time1=self.model.runtime


            for i in range(self.generator.Num_Customer):
                for j in range(self.generator.Num_Customer):
                    for k in range(self.generator.Num_Team):
                        if trigger2[0]==1:
                            constraint_name_0 = f"cut_0_{i}_{j}_{k}"
                            constraint_to_remove_0 = self.model.getConstrByName(constraint_name_0)
                            self.model.remove(constraint_to_remove_0)

                        if trigger2[1] == 1:
                            constraint_name_1 = f"cut_1_{i}_{j}_{k}"
                            constraint_to_remove_1 = self.model.getConstrByName(constraint_name_1)
                            self.model.remove(constraint_to_remove_1)

                        if trigger2[2] == 1:
                            constraint_name_2 = f"cut_2_{i}_{j}_{k}"
                            constraint_to_remove_2 = self.model.getConstrByName(constraint_name_2)
                            self.model.remove(constraint_to_remove_2)

            self.model.Params.NodeLimit = GRB.INFINITY
            self.model.optimize()
            time2 = self.model.runtime
            print(f"trigger2={trigger2}", end=' ')
            print(time1+time2)
            with open(file_path, "a") as file:
                data_to_append = f"trigger2={trigger2}:{time1+time2}"
                file.write(data_to_append + "\n")
            # self.model.reset()

        if trigger3!=[0,0,0]:
            self.model.Params.PreCrush = 1
            self.model.optimize(my_callback)
            time1 = self.model.runtime
            print(f"trigger3={trigger3}",end=' ')
            print(time1)
            with open(file_path, "a") as file:
                data_to_append = f"trigger3={trigger3}:{time1}"
                file.write(data_to_append + "\n")
            self.model.reset()
        if trigger4!=[0,0,0,0,0,0]:
            self.model.Params.NodeLimit = 1
            self.model.Params.OutputFlag=0
            self.model.optimize()
            time1 = self.model.runtime
            self.model.reset()

            if trigger4[0] == 1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            constraint_expr = self.X[i, j, k] + self.X[j, i, k] <= 1
                            constraint_name = f"cut_0_{i}_{j}_{k}"
                            self.model.addConstr(constraint_expr, name=constraint_name)

            if trigger4[1] == 1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            # 构建约束表达式
                            expr = self.T[i, k] + sum(
                                self.R[i, m] * self.generator.Service_Time[m] for m in
                                range(self.generator.Num_Service)) + \
                                   self.generator.Transfer_Time[i][j] - self.BigM * (1 - self.X[i, j, k])
                            # 构建约束名称，可以根据需要进行修改
                            constr_name = f'cut_1_{i}_{j}_{k}'
                            # 将约束添加到字典中
                            self.model.addConstr(expr <= self.generator.Late_Start_Limit[j], name=constr_name)

            if trigger4[2] == 1:
                for i in range(self.generator.Num_Customer):
                    for j in range(self.generator.Num_Customer):
                        for k in range(self.generator.Num_Team):
                            # 构建约束表达式
                            expr = self.generator.Early_Start_Limit[i] + self.generator.Transfer_Time[i][j] + min(
                                self.generator.Service_Time) - self.BigM * (1 - self.X[i, j, k])
                            # 构建约束名称，可以根据需要进行修改
                            constr_name = f'cut_2_{i}_{j}_{k}'
                            # 将约束添加到字典中
                            self.model.addConstr(expr <= self.generator.Late_Start_Limit[j], name=constr_name)
            self.model.update()
            self.root_num_constrs = self.model.NumConstrs
            self.model.Params.NodeLimit = 1
            self.model.Params.OutputFlag = 0
            self.model.optimize()
            time2 = self.model.runtime

            for i in range(self.generator.Num_Customer):
                for j in range(self.generator.Num_Customer):
                    for k in range(self.generator.Num_Team):
                        if trigger4[0]==1:
                            constraint_name_0 = f"cut_0_{i}_{j}_{k}"
                            constraint_to_remove_0 = self.model.getConstrByName(constraint_name_0)
                            self.model.remove(constraint_to_remove_0)

                        if trigger4[1] == 1:
                            constraint_name_1 = f"cut_1_{i}_{j}_{k}"
                            constraint_to_remove_1 = self.model.getConstrByName(constraint_name_1)
                            self.model.remove(constraint_to_remove_1)

                        if trigger4[2] == 1:
                            constraint_name_2 = f"cut_2_{i}_{j}_{k}"
                            constraint_to_remove_2 = self.model.getConstrByName(constraint_name_2)
                            self.model.remove(constraint_to_remove_2)

            self.model.Params.NodeLimit = GRB.INFINITY
            self.model.Params.PreCrush = 1
            self.model.Params.OutputFlag = 0
            self.model.optimize(my_callback_1)
            time3 = self.model.runtime
            self.time_final=time2+time3-time1
            print(f"trigger4={trigger4}",end=' ')
            print(time2+time3-time1)
            # with open(file_path, "a") as file:
            #     data_to_append = f"trigger4={trigger4}:{time2+time3-time1}"
            #     file.write(data_to_append + "\n")
            # self.model.reset()

    def print_log(self):
        """
        输出求解结果
        :return:
        """
        print('----------------')
        for k in range(self.generator.Num_Team):
            if self.Z[k].X != 0:
                print(self.Z[k].VarName, self.Z[k].X)

        print('----------------')
        for i in range(self.generator.Num_Customer + 2):
            for m in range(self.generator.Num_Service):
                if self.R[i, m].X != 0:
                    print(self.R[i, m].VarName, self.R[i, m].X)
        print('----------------')
        for i in range(self.generator.Num_Customer + 2):
            for k in range(self.generator.Num_Team):
                if self.Y[i, k].X != 0:
                    print(self.Y[i, k].VarName, self.Y[i, k].X)

        print('----------------')
        for i in range(self.generator.Num_Customer + 2):
            for k in range(self.generator.Num_Team):
                if self.T[i, k].X != 0:
                    print(self.T[i, k].VarName, self.T[i, k].X)

        print('----------------')
        for i in range(self.generator.Num_Customer + 2):
            for j in range(self.generator.Num_Customer + 2):
                for k in range(self.generator.Num_Team):
                    if self.X[i, j, k].X != 0:
                        print(self.X[i, j, k].VarName, self.X[i, j, k].X)

        print('----------------')
        print()
        print('求解时间为：', self.model.runtime)
        print('总性价比为：', self.Model_Obj[0].getValue())
        print('开始时间之和为：', self.Model_Obj[1].getValue())
        print('总利润为：', self.Model_Obj[2].getValue())
        print('原目标函数值为：', self.model.ObjVal)

        print('----------------')
        ServedCustomer = 0
        for i in range(self.generator.Num_Customer):
            if sum(self.Y[i, k].X for k in range(self.generator.Num_Team)) > 0:
                ServedCustomer += 1
        TeamUse = 0
        for k in range(self.generator.Num_Team):
            if self.Z[k].X > 0:
                TeamUse += 1
        print('服务客户数：', ServedCustomer, '未服务客户数：', self.generator.Num_Customer - ServedCustomer)
        print('服务团队占用数：', TeamUse, '服务团队空闲数：', self.generator.Num_Team - TeamUse)
