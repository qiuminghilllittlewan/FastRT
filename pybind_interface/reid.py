#!/usr/bin/env python3
# encoding: utf-8
import sys
import time
import cv2

sys.path.insert(1, '/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.insert(1, '/home/user/project_ws/devel/lib/python3/dist-packages')

# ros package
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

# reid package
import argparse
# import logging
import json

# import os
import numpy as np
import torch
from torch.backends import cudnn
import torch.nn.functional as F

from parse_data import ParseData
import logging
from tabulate import tabulate
from termcolor import colored
cudnn.benchmark = False
# logger = logging.getLogger('fastreid.visualize_result')
sys.path.append("/home/user/project_ws/src/reid_ros/scripts/fast-reid/projects/FastRT")
from build.pybind_interface.ReID import ReID

class ReidRos():
    def __init__(self):
        self.args, unknown = get_parser().parse_known_args()
        self.dev_no = self.args.dev_no
        clean_obj = ParseData()

        self.query_feats_people = []
        self.query_feats_car = []
        self.cv_image_query = None

        self.update_obj = False
        self.last_array = None

        # self.query_feats_flag = True
        self.query_feats = None

        self.track_flag = False
        self.track_again = []

        self.flag_1 = False
        self.flag_2 = False

        self.track_flag1 = False

        self._cv_bridge = CvBridge()
        self._pub_track_fir_frame = rospy.Publisher("/reid_image_rect" + str(self.dev_no), Image, queue_size=10)

        # tensorrt infer init
        temp_img = np.ones((50, 30, 3), np.uint8)
        test_imgs_list = [temp_img, temp_img]
        self.m = ReID(0)
        self.m1 = ReID(0)
        self.m.build("/home/user/project_ws/src/reid_ros/scripts/fast-reid/projects/FastRT/build/bot_R50_ibn.engine")
        self.m1.build("/home/user/project_ws/src/reid_ros/scripts/fast-reid/projects/FastRT/build/veri_wild_bot_R50_ibn.engine")
        print("build done")

        # warmup
        t_warm = time.time()
        _ = self.m.batch_infer(test_imgs_list)
        _ = self.m1.batch_infer(test_imgs_list)
        print(("warm time spend {} second").format(time.time()-t_warm))

        rospy.Subscriber('/gallery_image_rect' + str(self.dev_no), Image, self.callback_gallery, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber('/query_image_rect' + str(self.dev_no), Image, self.callback_query, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber('/state' + str(self.dev_no), Float32MultiArray, self.callback_track_again, queue_size=1, buff_size=2 ** 24)
        print('Image converter constructor')

    def callback_track_again(self, track_info):
        self.track_again = track_info.data

    def callback_gallery(self, image_msg):
        self.gallery_array = []
        self.gallery_array_people = []
        self.gallery_array_car = []
        self.flag_1 = True
        self.image_msg_gallery = image_msg
        self.cv_image_gallery = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self.gallery_array = json.loads(image_msg.header.frame_id)
        if self.flag_1 and self.flag_2 and self.gallery_array:
            self.proc(self.query_feats_people, self.query_feats_car, self.gallery_array, self.query_array, self.cv_image_gallery, self.cv_image_query, self.update_obj)

    def callback_query(self, image_msg):
        self.query_array = []
        self.query_array_people = []
        self.query_array_car = []
        self.flag_2 = True
        self.cv_image_query = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self.query_array = json.loads(image_msg.header.frame_id)
        for array in self.query_array:
            if array[4] == 0:
                self.query_array_people.append(array[0:4])
            else:
                self.query_array_car.append(array[0:4])
        # [self.query_array.append(rect[0:4]) for rect in json.loads(image_msg.header.frame_id)]
        if self.last_array == self.query_array:
            self.update_obj = False
        else:
            self.update_obj = True
            temp_query_feats_people = get_query_feats(self.cv_image_query, self.query_array_people)
            temp_query_feats_car = get_query_feats(self.cv_image_query, self.query_array_car)
            if len(temp_query_feats_people) > 0:
                self.query_feats_people = self.m.batch_infer(temp_query_feats_people)
            else:
                self.query_feats_people = []
            if len(temp_query_feats_car) > 0:
                self.query_feats_car = self.m1.batch_infer(temp_query_feats_car)
            else:
                self.query_feats_car = []
            self.last_array = self.query_array.copy()

    def proc(self, q_feat_people, q_feat_car, fir_array, sec_array, cv_image1, cv_image2, update_obj):
        t_init = time.time()
        ros_bbox_list = []
        parse_obj = ParseData()
        ann1, ann2, query_people_count, query_car_count = parse_obj.display(fir_array, sec_array, cv_image1, cv_image2)
        # if self.track_flag and len(self.track_again) == 0 and (update_obj | self.update_obj) is not True:
        if self.track_flag and self.track_flag1:
            [ros_bbox_list.append(rect[0:4]) for rect in ann1]
            # ros_bbox_list = ann1
        else:
            show_test(query_people_count, parse_obj.gallery_array_people,
                      query_car_count, parse_obj.gallery_array_car)
            ros_bbox_list_people, dict_people = self.match_obj(parse_obj, q_feat_people, parse_obj.direct_feats_people)
            ros_bbox_list_car, dict_car = self.match_obj(parse_obj, q_feat_car, parse_obj.direct_feats_car, dis_thresh=0.045, flag=False)
            if len(dict_people) == len(dict_car) == 0:
                ros_bbox_list.append(-1)
            else:
                # ros_bbox_list_people.extend(ros_bbox_list_car)
                ros_bbox_list = ros_bbox_list_people + ros_bbox_list_car
                ros_bbox_list.append(-2)
                self.track_flag = True
                self.track_again = []
        self.flag_1 = False
        # self.flag_2 = False
        msg_img = self._cv_bridge.cv2_to_imgmsg(self.cv_image_gallery, "bgr8")
        msg_img.header.frame_id = json.dumps(ros_bbox_list, ensure_ascii=False)
        self._pub_track_fir_frame.publish(msg_img)
        print("Total time: ", time.time()-t_init)
        print('Reid frame:', 1/(time.time()-t_init))

    def match_obj(self, parse_obj, q_feat, g_batch, dis_thresh=0.09, flag=True):
        ros_bbox_list_temp = []
        match_info = dict()
        if len(q_feat) and len(g_batch):
            if flag:
                t_infer = time.time()
                g_feat = self.m.batch_infer(g_batch)
                print('Inference time', time.time() - t_infer)
            else:
                g_feat = self.m1.batch_infer(g_batch)
            q_feat = torch.Tensor(q_feat)
            g_feat = torch.Tensor(g_feat)
            q_feat = F.normalize(q_feat)
            g_feat = F.normalize(g_feat)
            distmat = 1 - torch.mm(q_feat, g_feat.t())
            distmat = distmat.numpy()
            match_info, _, _ = parse_obj.lapjv(distmat, dis_thresh)  # car 0.045 person 0.09
            if len(match_info):
                for k in match_info:
                    conf = float((dis_thresh - distmat[k[0]][k[1]]) / dis_thresh + 0.5)
                    conf = min(1, conf)
                    if flag:
                        parse_obj.gallery_array_people[k[1]].append(conf)
                        ros_bbox_list_temp.append(parse_obj.gallery_array_people[k[1]])
                    else:
                        parse_obj.gallery_array_car[k[1]].append(conf)
                        ros_bbox_list_temp.append(parse_obj.gallery_array_car[k[1]])
        return ros_bbox_list_temp, match_info

    def main(self):
        rospy.spin()

def show_test(query_people_count, gallery_people, query_car_count, gallery_car):
    headers = ['item', 'query_people', 'gallery_people', 'query_car', 'gallery_car']
    csv_results = [
        ['# images', query_people_count, len(gallery_people), query_car_count, len(gallery_car)]
    ]
    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="pipe",
        headers=headers,
        numalign="left",
    )
    print(colored(table, "cyan"))

def get_query_feats(img, arr):
    temp_feats = []
    if len(arr) > 0:
        for i, area in enumerate(arr):
            part_image = np.array(img[np.int32(area)[1]:np.int32(area)[3], np.int32(area)[0]:np.int32(area)[2]])
            temp_feats.append(part_image)
    return temp_feats


def get_parser():

    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument("--dev_no", type=int, default=2, help="dev number")
    return parser


if __name__ == '__main__':
    rospy.init_node('reid_proc_ros2')
    tensor = ReidRos()
    tensor.main()
