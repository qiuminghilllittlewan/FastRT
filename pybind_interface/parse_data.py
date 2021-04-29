import os
import json
import cv2
import numpy as np
import shutil
import lap

class ParseData:
    def __init__(self):
        # self.last_path = os.path.dirname(os.getcwd())
        self.last_path = '/home/user/project_ws/src/reid_ros/scripts/fast-reid'
        self.match_dict = {}
        root_path = os.path.join(self.last_path, 'datasets/TEST')

        self.train_path = os.path.join(self.last_path, 'datasets/TEST/bounding_box_train')
        self.test_path = os.path.join(self.last_path, 'datasets/TEST/bounding_box_test')
        self.query_path = os.path.join(self.last_path, 'datasets/TEST/query')
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        else:
            shutil.rmtree(root_path)
            os.mkdir(root_path)
        os.mkdir(self.train_path)
        os.mkdir(self.test_path)
        os.mkdir(self.query_path)
        self.direct_feats = []
        self.direct_feats_people = []
        self.direct_feats_car = []
        self.gallery_array_people = []
        self.gallery_array_car = []
        

    # 将json解析成xywh，class
    def proc_json(self, data_path):
        with open(data_path, 'r') as f:
            name = data_path.split('.')[0].split('/')[-1]
            data = json.load(f)
            labels = data['class']
            rec = data['box']
        return labels, rec, name

    # 将目标检测的结果通过ros发送
    def proc_ros_data(self,fir_array):
        labels = []
        ann = []
        fir_array = fir_array.data
        # fir_array = [55,66,99,100,class]

        if len(fir_array) > 0:
            for i in range(int(len(fir_array)/5)):
                # rec.append([[fir_array[5*i+1], fir_array[5*i+2]], [fir_array[5*i+3], fir_array[5*i+4]]])
                ann.append([fir_array[5*i], fir_array[5*i+1], fir_array[5*i+2], fir_array[5*i+3]])
                labels.append(fir_array[5 * i])
        return labels, ann

    # 制作数据集
    def ready_datasets(self, image1, rec1, image2, rec2):
        if len(rec1) & len(rec2):
            for i, area in enumerate(rec1):
                part_image = image1[np.int32(area)[1]:np.int32(area)[3], np.int32(area)[0]:np.int32(area)[2]]
                save_path = os.path.join(self.test_path, '00000001_0001_' + str(i + 1) + '.jpg')
                cv2.imwrite(save_path, part_image)
            for j, area in enumerate(rec2):
                part_image = image2[np.int32(area)[1]:np.int32(area)[3], np.int32(area)[0]:np.int32(area)[2]]
                save_path = os.path.join(self.query_path, '00000001_0002_' + str(j + 1) + '.jpg')
                cv2.imwrite(save_path, part_image)

    # 制作数据集
    def ready_datasets1(self, image1, labels1, rec1, image2, labels2, rec2):
        for i, area in enumerate(rec1):
            lt, rd = area[0:2]
            part_image = image1[np.int32(lt)[1]:np.int32(rd)[1], np.int32(lt)[0]:np.int32(rd)[0]]
            if len(labels1) >= len(labels2):
                save_path = os.path.join(self.test_path, '00000001_0001_' + str(i + 1) + '.jpg')
            else:
                save_path = os.path.join(self.query_path, '00000001_0002_' + str(i + 1) + '.jpg')
            cv2.imwrite(save_path, part_image)
        for j, area in enumerate(rec2):
            lt, rd = area[0:2]
            part_image = image2[np.int32(lt)[1]:np.int32(rd)[1], np.int32(lt)[0]:np.int32(rd)[0]]
            if len(labels1) >= len(labels2):
                save_path = os.path.join(self.query_path, '00000001_0002_' + str(j + 1) + '.jpg')
            else:
                save_path = os.path.join(self.test_path, '00000001_0001_' + str(j + 1) + '.jpg')
            cv2.imwrite(save_path, part_image)
        return len(labels1), len(labels2)

    # 处理匹配之后的标签
    def proc_labels(self, distmat, data_loader, num, match, sort_flag=False):
        self.match_dict = {}
        if sort_flag:
            for n, index in enumerate(match):
                self.match_dict[str(index[1])] = index[0]
            return None, self.match_dict


        # 优化匹配算法
        indices = []
        for n, index in enumerate(match):
            index_list = []
            index_list.append(index[1])
            indices.append(index_list)
        # 传统匹配算法
        # indices = np.argsort(distmat, axis=1)
        # indices = indices.tolist()
        query_info = data_loader.dataset.img_items
        all_size = len(query_info)

        # query_img = query_info['images']
        # cam_id = query_info['camids']
        # query_name = query_info['img_paths'].split('/')[-1]
        for i, indice in enumerate(indices):
            k = int(query_info[i][0].split('.')[0].split('_')[-1]) - 1
            temp = query_info[num+indice[0]][0].split('.')[0].split('_')[-1]
            temp = int(temp) - 1
            # xxxx = distmat[i]
            # if str(indice[0]) in self.match_dict.keys():
            if str(temp) in self.match_dict.keys():
                self.match_dict[str(temp)] = k
                # continue
            else:
                self.match_dict[str(temp)] = k
        print(self.match_dict)

        result = []
        return result, self.match_dict

    # 用于ROS订阅目标检测信息
    def display(self, fir_ann, sec_ann, cv_image1, cv_image2):
        count_people = 0
        count_car = 0
        if len(fir_ann):
            for array in fir_ann:
                if array[4] == 0:
                    self.gallery_array_people.append(array[0:4])
                else:
                    self.gallery_array_car.append(array[0:4])
            for array in sec_ann:
                if array[4] == 0:
                    count_people += 1
                else:
                    count_car += 1
            self.direct_feats_people = crop_img(cv_image1, self.gallery_array_people)
            self.direct_feats_car = crop_img(cv_image1, self.gallery_array_car)

        # if len(fir_ann) and len(sec_ann):
        #     for j, area in enumerate(sec_ann):
        #         part_image = cv_image2[np.int32(area)[1]:np.int32(area)[3], np.int32(area)[0]:np.int32(area)[2]]
        #         save_path = os.path.join(self.query_path, '00000001_0002_' + str(j + 1) + '.jpg')
        #         cv2.imwrite(save_path, part_image)
        #         self.direct_feats.append(part_image)
            # for i, area in enumerate(fir_ann):
            #     part_image = cv_image1[np.int32(area)[1]:np.int32(area)[3], np.int32(area)[0]:np.int32(area)[2]]
            #     save_path = os.path.join(self.test_path, '00000001_0001_' + str(i + 1) + '.jpg')
            #     cv2.imwrite(save_path, part_image)
                # self.direct_feats.append(part_image)
        return fir_ann, sec_ann, count_people, count_car

    # 用于系统演示
    def display1(self, images_path):
        ann_path = images_path.replace('images', 'ann')
        path_list = sorted(os.listdir(ann_path))
        json_dict = {}
        for i, ann1_name in enumerate(path_list):
            if ann1_name not in json_dict.keys():
                ann1_name.replace('json', 'jpg')
                img1_name = ann1_name.replace('json', 'jpg')
                ann1_path = os.path.join(ann_path, ann1_name)
                img1_path = os.path.join(images_path, img1_name)
                ann2_name = path_list[i+1]
                img2_name = ann2_name.replace('json', 'jpg')
                ann2_path = os.path.join(ann_path, ann2_name)
                img2_path = os.path.join(images_path, img2_name)
                json_dict[ann1_name] = i
                json_dict[ann2_name] = i+1
                self.image1 = cv2.imread(img1_path)
                self.image2 = cv2.imread(img2_path)
                self.labels1, self.rec1, self.name1 = self.proc_json(ann1_path)
                self.labels2, self.rec2, self.name2 = self.proc_json(ann2_path)
                self.id_num1, self.id_num2 = self.ready_datasets(self.image1, self.labels1, self.rec1, self.image2,
                                                                 self.labels2, self.rec2)
                yield i

    def lapjv(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0,2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

def crop_img(cv_image, arr):
    temp_feat = []
    if len(arr):
        for j, area in enumerate(arr):
            if area[3]-area[1] > 0 and area[2]-area[0] > 0:
                part_image = np.array(cv_image[np.int32(area)[1]:np.int32(area)[3], np.int32(area)[0]:np.int32(area)[2]])
                temp_feat.append(part_image)
    return temp_feat
