"""
Create Date: 2021/7/6
Create Time: 13:25
Author: hxf
"""
import argparse

from torch import frac, prod
import cv2
import logging
from logging.handlers import TimedRotatingFileHandler
import multiprocessing as mp
import numpy as np
import os
import subprocess as sp
import time
from IVUS import IVUS

parser = argparse.ArgumentParser(description='处理视频数据，保存分割结果')
# parser.add_argument('--from_video_path', type=str, default="./resource/test1.mp4", help='视频流路径来源')
parser.add_argument('--from_video_path', type=str, default="rtmp://58.200.131.2:1935/livetv/cctv1", help='视频流路径来源')
parser.add_argument('--to_video_path', type=str, default="rtmp://60.208.15.171:1935/live/18", help='视频流路径推送')
parser.add_argument('--save_origin_path', type=str, default='./origin_images', help='原始视频帧保存路径')
parser.add_argument('--save_seg_path', type=str, default='./seg_images', help='分割视频帧保存路径')
parser.add_argument('--save_log_path', type=str, default='./log', help='日志路径')
args = parser.parse_args()

from_video_path = args.from_video_path
to_video_path = args.to_video_path
save_origin_path_p = args.save_origin_path
save_seg_path_p = args.save_seg_path
save_log_path = args.save_log_path
video_name = os.path.basename(from_video_path).split('.')[0]
save_origin_path = os.path.join(save_origin_path_p, video_name)
save_seg_path = os.path.join(save_seg_path_p, video_name)

if not os.path.exists(save_origin_path):
    os.makedirs(save_origin_path, exist_ok=True)
if not os.path.exists(save_seg_path):
    os.makedirs(save_seg_path, exist_ok=True)
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)


def init_log(log_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(os.path.join(save_log_path, 'ProcessVideoOnline_{}.log'.format(log_name)),
                                       when='d',
                                       interval=1,
                                       backupCount=30)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.suffix = "%Y%m%d.log"
    logger.addHandler(handler)

    logger.info("from_video_path: {}, to_video_path: {}, save_origin_path: {}, save_seg_path: {}, "
                "save_log_path: {}".format(from_video_path, to_video_path, save_origin_path_p,
                                           save_seg_path_p, save_log_path))
    logger.info("start pid: {}".format(os.getpid()))

    return logger


def get_video(queue, d):
    logger = init_log("get_video")

    cap = cv2.VideoCapture(from_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    now_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_origin_path_mp4 = os.path.join(save_origin_path, 'video_origin_{}.mp4'.format(now_time))
    out_video_origin = cv2.VideoWriter(save_origin_path_mp4, fourcc, fps, (width, height))

    logger.info("视频流读取开始")
    while True:
        try:
            if queue.qsize() >= d["max_size"]:
                logger.info("处理队列已满: {}".format(queue.qsize()))
                time.sleep(0.1)
                continue

            if queue.qsize() != 0 and not d["flag"]:
                logger.info("等待当前视频流处理: {}".format(queue.qsize()))
                time.sleep(0.1)
                continue

            success, frame = cap.read()
            print(success)
            if not success:
                logger.info("get_video: 暂无视频流")
                d["flag"] = False  # 无视频或者读取结束
                out_video_origin.release()
                time.sleep(1)
                continue

            if success and d['frame_num'] == 0 and queue.qsize() == 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                d["fps"] = fps
                d["size"] = (width, height)

                logger.info("帧率fps: {}".format(fps))
                logger.info("源视频帧大小: {}".format((width, height)))

                now_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                save_origin_path_mp4 = os.path.join(save_origin_path, 'video_origin_{}.mp4'.format(now_time))
                out_video_origin = cv2.VideoWriter(save_origin_path_mp4, fourcc, fps, (width, height))
                d["flag"] = True  # 有视频流

            out_video_origin.write(frame)
            queue.put(frame, block=True, timeout=None)
        except KeyboardInterrupt:
            out_video_origin.release()
            raise


def put_video(queue, d):
    logger = init_log("put_video")
    # 实例化ivus对象
    ivus = IVUS(save_origin_path, save_seg_path)

    while True:
        try:
            if queue.qsize() <= 0:
                logger.info("get_video: 暂无视频流")
                time.sleep(0.1)
                continue

            frame = queue.get(block=True, timeout=None)
            d['frame_num'] += 1

            if d['frame_num'] == 1:
                seg_size = (int(d["size"][0] / 2), int(d["size"][1] / 2))
                logger.info("分割后的视频帧大小: {}".format(seg_size))
                # 将处理后的视频流写到本地做备份
                now_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                save_seg_path_mp4 = os.path.join(save_seg_path, 'video_seg_{}.mp4'.format(now_time))
                print(d["fps"], seg_size, save_seg_path_mp4)
                out_video_seg = cv2.VideoWriter(save_seg_path_mp4, fourcc, d["fps"], seg_size)

                command = ['ffmpeg',
                           '-y',
                           '-f',
                           'rawvideo',
                           '-vcodec',
                           'rawvideo',
                           '-pix_fmt',
                           'bgr24',
                           '-s',
                           "{}x{}".format(seg_size[0], seg_size[1]),
                           '-r', str(d["fps"]),
                           '-i',
                           '-',
                           '-c:v',
                           'libx264',
                           '-pix_fmt',
                           'yuv420p',
                           '-preset',
                           'ultrafast',
                           '-f',
                           'flv',
                           to_video_path]
                pipe = sp.Popen(command, stdin=sp.PIPE)  # shell=False

            frame = cv2.resize(frame, seg_size)
            frame_seg = ivus.seg_and_cls(frame, d['frame_num'])
            out_video_seg.write(cv2.cvtColor(np.array(frame_seg), cv2.COLOR_RGB2BGR))
            pipe.stdin.write(frame.tostring())
            logger.info("处理帧数：{}".format(d['frame_num']))
            if not d["flag"] and queue.qsize() == 0:
                d['frame_num'] = 0
                out_video_seg.release()
                time.sleep(1)
                continue
        except KeyboardInterrupt:
            out_video_seg.release()
            raise


def run():
    max_size = 5
    mp.set_start_method(method="spawn")
    queue = mp.Queue(maxsize=max_size)
    d = mp.Manager().dict()

    d["max_size"] = max_size
    d["flag"] = True
    d["frame_num"] = 0
    d["fps"] = 10
    d["size"] = (512, 512)

    processes = [mp.Process(target=get_video, args=(queue, d)),
                 mp.Process(target=put_video, args=(queue, d))]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == "__main__":
    run()
