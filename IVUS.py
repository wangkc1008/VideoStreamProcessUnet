import os
import time
import torch
from torchvision import transforms, models
import unet.unet_model as unet
from config import HTConfig as config
from PIL import Image, ImageDraw, ImageFont


class IVUS:
    def __init__(self, save_origin_path="./origin_images", save_seg_path="./seg_images"):
        self.save_origin_path = save_origin_path
        self.save_seg_path = save_seg_path

        self.seg_model_path = config.seg_model_path
        self.UNet = unet.UNet(n_channels=1, n_classes=2).cuda()
        self.UNet.load_state_dict(torch.load(self.seg_model_path))
        self.UNet.eval()

        self.classify_model_path = config.classify_model_path
        self.ResNet = models.resnet50(pretrained=True).cuda()
        self.ResNet.conv1 = torch.nn.Conv2d(config.pic_select, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False).cuda()
        features = self.ResNet.fc.in_features
        self.ResNet.fc = torch.nn.Linear(features, config.label_class).cuda()
        self.ResNet.load_state_dict(torch.load(self.classify_model_path))
        self.ResNet.eval()

        if not os.path.exists(self.save_origin_path):
            os.makedirs(self.save_origin_path, exist_ok=True)
        if not os.path.exists(self.save_seg_path):
            os.makedirs(self.save_seg_path, exist_ok=True)

        self.font = ImageFont.truetype("./resource/msyhbd.ttc", 20)
        self.preprocess_transforms = transforms.Compose([  # 初等图像预处理变换
            transforms.ToTensor()
        ])

    def seg_and_cls(self, frame, frame_num):
        # 图像预处理

        image_PIL = Image.fromarray(frame)
        image_tensor = self.preprocess_transforms(image_PIL.convert('L'))
        image_tensor_unsq = torch.unsqueeze(image_tensor, 0).cuda()

        # 保存原数据帧
        now_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        image_PIL.save(os.path.join(self.save_origin_path, 'origin_{}_{}.jpg'.format(now_time, frame_num)))

        # 分割
        numpy_img = self.ivus_seg(image_tensor_unsq)
        flag = 0
        if numpy_img[numpy_img == 255].shape[0] != 0:
            frame[:, :, 1][numpy_img == 255] += 100
            flag = 1

        # 分类
        cls_res = self.ivus_classify(image_tensor_unsq)

        # 分类结果写在分割后的图片上
        frame_shape = frame.shape
        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        draw.text((frame_shape[1] - 80, frame_shape[0] - 50), cls_res, fill=(255, 255, 255), font=self.font)

        # 保存处理后的数据帧
        frame.save(os.path.join(self.save_seg_path, 'seg_{}_{}_{}.jpg'.format(now_time, frame_num, flag)))
        return frame

    def ivus_seg(self, image_tensor_unsq):
        test_output = self.UNet(image_tensor_unsq)
        pred_y = torch.max(test_output, 1)[1].data  # 预测结果

        numpy_img = pred_y.cpu().numpy()[0, :, :] * 255
        return numpy_img

    def ivus_classify(self, image_tensor_unsq):
        test_output = self.ResNet(image_tensor_unsq)
        pred_y = torch.max(test_output, 1)[1].cuda().data  # 预测结果
        if pred_y.item() == 1:  # 二分类1为有斑块，0为正常  三分类2为混合斑1为硬斑0为软斑
            res = "硬斑"
        elif pred_y.item() == 0:
            res = "软斑"
        else:
            res = "混合斑"
        return res


