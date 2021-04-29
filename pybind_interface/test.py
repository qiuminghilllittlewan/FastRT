import sys

sys.path.append("../")
from build.pybind_interface.ReID import ReID
import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np
import random

if __name__ == '__main__':
    iter_ = 20000
    m = ReID(0)
    m.build("/home/user/project_ws/src/reid_ros/scripts/fast-reid/projects/FastRT/build/bot_R50_ibn.engine")
    print("build done")
    frame_list = []
    frame = cv2.imread("/home/user/project_ws/src/reid_ros/scripts/fast-reid/datasets/TEST/query/00000001_0002_1.jpg")
    frame1 = cv2.imread("/home/user/project_ws/src/reid_ros/scripts/fast-reid/datasets/TEST/query/00000001_0002_2.jpg")

    frame_list = [frame, frame1]
    m.infer(frame)
    t0 = time.time()
    m.infer(frame)
    print(time.time()-t0)

    img = cv2.imread("/home/user/gallery.bmp")
    img1 = np.array(img[374:579, 708:795])
    img2 = np.array(img[355:595, 891:979])

    img_list = [img1, img2]


    # for i in range(img1.shape[0]):
    #     for j in range(img1.shape[1]):
    #         for k in range(img1.shape[2]):
    #             result = random.sample(range(-4, 4), 8)
    #             pix = img1[i, j, k]
    #             img1[i, j, k] = pix + result[0]




    for i in range(iter_):
        t1 =time.time()
        # m.infer(frame)
        dd = m.batch_infer(img_list)
        cc = m.batch_infer(frame_list)

        t1 = time.time()
        cc = torch.Tensor(cc)
        dd = torch.Tensor(dd)
        cc = F.normalize(cc)
        dd = F.normalize(dd)
        print(time.time()-t1)
        print('f')

    total = time.time() - t0
    print("CPP API fps is {:.1f}, avg infer time is {:.2f}ms".format(iter_ / total, total / iter_ * 1000))
    print("successful")