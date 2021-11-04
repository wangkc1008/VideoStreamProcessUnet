class Config():

    # 初始化类
    def __init__(self):
        # 图片大小
        self.pic_size = [512, 512]
        # 图片通道数量
        self.pic_select = 1
        # 训练网络中的参数
        self.Epoch = 260                # 循环次数
        self.Batch_Size = 2            # 每次多少个batch
        self.LR = 0.001                 # 初始学习速率
        self.Weight_Decay = 5e-4        # 权重衰减
        self.LR_Gamma = 0.3             # 学习率每次衰减多少
        self.LR_Patience = 50           # 能够容忍多少次学习率不变
        self.LR_Min = 1e-8              # 学习率最小值
        self.Epoch_Save_Model = 40      # 多少Epoch保存一下网络
        self.label_class = 3            # 输出是几类网络
        self.x_fold = 5                 # 用于设置X折交叉验证
        self.X_random = True            # 是否进行随机数

        # ======================= 区分训练和测试 =================================
        self.heatmap = False            # 是否进行热力图
        self.stop_grad = False          # 是否冻结预训练网络
        self.save_image = False         # 是否进行图片存储(存储的名字编号)
        self.beta_version = 3           # 第几个训练版本

        self.seg_model_path = "./model/ver1.0_4.pkl"
        self.classify_model_path = "./model/Beta3_fold0_epoch199.pkl"

# 调用生成一个类
HTConfig = Config()
