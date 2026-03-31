import sys
import numpy as np
from numpy import ndarray


class Generator(object):
    """
    参数生成的类
    """
    ser_pro_coe: int | ndarray[int]

    def __init__(self):
        self.Num_Customer = 0  # 顾客数
        self.Num_Service = 0  # 服务数
        self.Num_Product = 0  # 产品数
        self.Num_Team = 0  # 服务团队数

        self.random_seed = 670  # np.random.seed(670)

        # 服务时间
        self.Service_Time = None  # 各服务时间，按服务等级升序排列
        self.service_time_lb = 1
        self.service_time_step = 1

        # 服务价格
        self.Service_Price = None  # 各服务价格,按服务等级升序排列
        self.service_price_step = 1
        self.service_price_lb = 1

        # 服务成本
        self.Service_Cost = None
        self.service_cost_step = 1
        self.service_cost_lb = 1

        # 用户效用值
        self.Utility = None  # 每个客户 每个服务等级对应一个效用值
        self.utility_lb = 4
        self.utility_step = 1

        # 产品库存
        self.Inventory = None  # 设置产品库存，按低级到高级来递减
        self.inventory_step = 1
        self.inventory_lb = 1

        # 产品服务对应关系
        self.Ser_Pro_Corr = None  # np.eye()
        self.ser_pro_coe = 1
        self.ser_pro_coe_lb = 1
        self.ser_pro_coe_ub = 2

        # 客户权重
        self.weight_tri = 0  # 0: weight 相等, 1: weight 不等
        self.weight_gap = 2  # the gap between lb and ub
        self.Weight = None

        # 服务开始时间下界
        self.Early_Start_Limit = None
        self.early_start_lb = 0
        self.early_start_ub = 2
        # 服务开始时间上界
        self.Late_Start_Limit = None
        self.late_start_lb = 0
        self.late_start_ub = 2

        # 任务转移时间
        self.Transfer_Time = None
        self.transfer_lb = 1
        self.transfer_ub = 2

        # 工作时长
        self.Duration = None
        self.duration = 1

        # 任务固定成本
        self.FixedCost = None

        # 任务可变成本
        self.VariableCost = None
        self.variable_cost_lb = 0
        self.variable_cost_ub = 2

    def set_instance(self, num_customer, num_service, num_product, num_team, random_seed):
        """
        导入样本生成的参数
        :param num_customer: 服务的客户数
        :param num_service: 服务等级数量
        :param num_product: 产品的类数，等于服务等级数量
        :param num_team: 提供服务的工人队伍数
        :param random_seed: 随机种子
        :return:
        """
        self.Num_Customer = num_customer
        self.Num_Service = num_service
        self.Num_Product = num_product
        self.Num_Team = num_team
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        if self.Num_Product != self.Num_Service:
            print('Warning!!! 产品服务个数不相等!!!')
            sys.exit()

    def set_params(self, service_time_step, service_price_step, service_cost_step,
                   service_time_lb, service_price_lb, service_cost_lb,
                   early_start_lb, early_start_ub,
                   late_start_lb, late_start_ub,
                   transfer_lb, transfer_ub,
                   utility_lb, utility_step,
                   inventory_step, inventory_lb,
                   weight_tri, weight_gap,
                   variable_cost_lb, variable_cost_ub,
                   ser_pro_coe, ser_pro_coe_lb, ser_pro_coe_ub,
                   ):
        """
        导入参数生成的范围，取值的上下界，或者生成实例的步长
        """
        # 可变成本与客户权重存在一定的线性关系
        self.service_time_step = service_time_step
        self.service_price_step = service_price_step
        self.service_cost_step = service_cost_step

        self.service_time_lb = service_time_lb
        self.service_price_lb = service_price_lb
        self.service_cost_lb = service_cost_lb

        self.early_start_lb = early_start_lb
        self.early_start_ub = early_start_ub
        self.late_start_lb = late_start_lb
        self.late_start_ub = late_start_ub

        self.transfer_lb = transfer_lb
        self.transfer_ub = transfer_ub

        self.weight_tri = weight_tri
        self.weight_gap = weight_gap

        self.utility_lb = utility_lb
        self.utility_step = utility_step

        self.inventory_step = inventory_step
        self.inventory_lb = inventory_lb

        self.variable_cost_lb = variable_cost_lb
        self.variable_cost_ub = variable_cost_ub

        self.ser_pro_coe = ser_pro_coe
        self.ser_pro_coe_lb = ser_pro_coe_lb
        self.ser_pro_coe_ub = ser_pro_coe_ub

    def gen_customer_weight(self, weight_gap, weight_tri):
        """
        生成客户的权重
        :return:
        """
        if weight_tri == 0:
            self.Weight = np.ones(self.Num_Customer)

        if weight_tri == 1:
            self.Weight = np.random.randint(low=1, high=1 + weight_gap, size=self.Num_Customer)

    def gen_service_relations(self):
        """
        生成服务相关的参数，服务时间、价格、成本
        :return:
        """
        self.Service_Time = np.arange(self.service_time_lb,
                                      self.service_time_lb + self.service_time_step * self.Num_Service,
                                      self.service_time_step)

        self.Service_Price = np.arange(self.service_price_lb,
                                       self.service_price_lb + self.service_price_step * self.Num_Service,
                                       self.service_price_step)

        self.Service_Cost = np.arange(self.service_cost_lb,
                                      self.service_cost_lb + self.service_cost_step * self.Num_Service,
                                      self.service_cost_step)

    def gen_customer_relations(self):
        """
        生成客户相关的参数。客户效用和库存量
        :return:
        """
        # 效用值
        self.Utility = np.ones((self.Num_Customer, self.Num_Service)) * self.utility_lb
        for i in range(self.Num_Customer):
            self.Utility[i] += np.arange(0, self.Num_Service * self.utility_step, self.utility_step)

        # 库存
        # 注意库存是递减
        self.Inventory = np.arange(self.Num_Product * self.inventory_step + self.inventory_lb,
                                   self.inventory_lb,
                                   -self.inventory_step)

    def gen_time_relations(self):
        """
        生成时间相关的参数，客户的时间窗
        :return:
        """
        self.Early_Start_Limit = np.random.randint(low=self.early_start_lb, high=self.early_start_ub,
                                                   size=self.Num_Customer)
        self.Late_Start_Limit = self.Early_Start_Limit + np.random.randint(low=self.late_start_lb,
                                                                           high=self.late_start_ub,
                                                                           size=self.Num_Customer)

    def gen_dis_matrix(self):
        """
        生成客户见的距离矩阵
        :return:
        """
        matrix_size = int((self.Num_Customer + 1 + 1) * (self.Num_Customer + 1) / 2)
        m = np.random.randint(low=self.transfer_lb, high=self.transfer_ub,
                              size=matrix_size)

        n = len(m)
        n_matrix = int((1 + int((1 + 8 * n) ** 0.5)) / 2)
        semi_matrix = np.zeros((n_matrix, n_matrix), dtype='int32')
        start_index = 0
        for row in range(n_matrix - 1):
            end_index = start_index + (n_matrix - 1 - row)
            semi_matrix[row, row + 1:] = m[start_index:end_index]
            start_index = end_index
        self.Transfer_Time = semi_matrix + semi_matrix.T

        # depot_0 与 depot_n 的距离应该是相等的
        self.Transfer_Time[-1] = self.Transfer_Time[0]
        for row in range(len(self.Transfer_Time)):
            self.Transfer_Time[row][-1] = self.Transfer_Time[row][0]

        return self.Transfer_Time

    def gen_cost_relations(self):
        """
        生成成本相关的参数，对于服务等级的服务时间、固定成本、可变成本
        :return:
        """
        self.Duration = np.random.randint(low=np.max(self.Service_Time),
                                          high=np.max(self.Service_Time) * 2) * np.random.randint(low=self.Num_Customer,
                                                                                                  high=self.Num_Customer + 5)

        self.FixedCost = np.random.randint(low=np.min(self.Service_Cost), high=np.max(self.Service_Cost) + 1) * 0.5

        self.VariableCost = np.random.uniform(low=self.variable_cost_lb,
                                              high=self.variable_cost_ub,
                                              size=(self.Num_Customer + 2, self.Num_Customer + 2))
        self.VariableCost = self.VariableCost * self.Transfer_Time * 0.75

    def gen_ser_pro_coe(self):
        """
        生成服务与消耗商品的对应关系
        :return:
        """
        self.Ser_Pro_Corr=[]
        for i in range(self.Num_Service):
            self.ser_pro_coe = np.random.randint(low=self.ser_pro_coe_lb,
                                                 high=self.ser_pro_coe_ub,
                                                 size=self.Num_Product)
            self.Ser_Pro_Corr.append(self.ser_pro_coe)
