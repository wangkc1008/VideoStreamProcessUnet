"""
Create Date: 2021/7/6
Create Time: 13:25
Author: hxf
"""
import os
import cv2
import argparse
import numpy as np
from IVUS import IVUS

parser = argparse.ArgumentParser(description='处理视频数据，保存分割结果')
parser.add_argument('--video_path', type=str, default="./resource/test1.mp4", help='视频路径')
parser.add_argument('--save_origin_path', type=str, default='./origin_images', help='原始视频帧保存路径')
parser.add_argument('--save_seg_path', type=str, default='./seg_images', help='分割视频帧保存路径')
args = parser.parse_args()


class Producer:
    def __init__(self, video_path, save_origin_path, save_seg_path):
        super(Producer, self).__init__()
        self.video_path = video_path
        video_name = os.path.basename(self.video_path).split('.')[0]
        self.save_origin_path = os.path.join(save_origin_path, video_name)
        self.save_seg_path = os.path.join(save_seg_path, video_name)

        self.cap = cv2.VideoCapture(self.video_path)
        self.ivus = IVUS(self.save_origin_path, self.save_seg_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("video_path: {}".format(self.video_path))
        print("fps: {}".format(self.fps))
        self.frame_num = 0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("size: {}".format((width, height)))
        # 定义编码格式mpge-4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 定义视频文件输入对象
        save_origin_path = os.path.join(self.save_origin_path, 'video_origin.mp4')
        self.out_video_origin = cv2.VideoWriter(save_origin_path, fourcc, self.fps, (width, height))
        save_seg_path = os.path.join(self.save_seg_path, 'video_seg.mp4')
        self.seg_size = (int(width / 1), int(height / 1))
        self.out_video_seg = cv2.VideoWriter(save_seg_path, fourcc, self.fps, self.seg_size)

    def run(self):
        print("视频读取开始")
        while True:
            success, frame = self.cap.read()
            if not success:  # 当前视频读取结束
                print("视频读取结束")
                self.out_video_origin.release()
                self.out_video_seg.release()
                break

            self.out_video_origin.write(frame)
            self.frame_num += 1
            frame = cv2.resize(frame, self.seg_size)
            frame_seg = self.ivus.seg_and_cls(frame, self.frame_num)

            self.out_video_seg.write(cv2.cvtColor(np.array(frame_seg), cv2.COLOR_RGB2BGR))
            print("进度：{}".format(self.frame_num))


if __name__ == "__main__":
    video_path = args.video_path
    save_origin_path = args.save_origin_path
    save_seg_path = args.save_seg_path
    producer = Producer(video_path, save_origin_path, save_seg_path)
    producer.run()


