import os
from models.object_detection_model import ObjectDetectionModel
import cv2
from lxml import etree
import xml.etree.cElementTree as ET
import numpy as np
import yaml
import json

import logging
logging.basicConfig(format="🕐  %(asctime)s - %(message)s", level=logging.INFO)

import argparse

from tqdm import tqdm
from PIL import Image
import time

import signal

class DataSet(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        if '.mp4' in self.dataset_path in self.dataset_path:
            self.dataset = Video(self.dataset_path)
        else:
            self.dataset = ImageDir(self.dataset_path)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, key):
        return self.dataset[key]

    def get_progress(self):
        return self.dataset.get_progress()
    
    def set_progress(self, ind):
        self.dataset.set_progress(ind)


class Video(object):
    def __init__(self, video_path='../data/video/MVI_7739.mp4'):
        self.__video = cv2.VideoCapture(video_path)
        self.__total_frames = int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __read_frame(self, frame_num):
        assert frame_num < self.__total_frames, f"Fail!\nFrame Num {frame_num} exceed threshold {self.__total_frames}\n"
        assert frame_num >= 0, f"Fail!\nFrame Num {frame_num} less than 0\n"

        self.__video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, image_bgr = self.__video.read()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        return image_bgr, image_rgb

    def __len__(self):
        return self.__total_frames
    
    def __getitem__(self, key):
        return self.__read_frame(frame_num=key)

    def get_progress(self):
        return 0

    def set_progress(self, ind):
        pass

class ImageDir(object):
    def __init__(self, img_dir_path="../data/img_data/integr"):
        self.__img_dir_path = img_dir_path
        try:
            with open(f"{self.__img_dir_path}/img_list.txt", 'r') as f:
                self.__img_list = f.readlines()
        except FileNotFoundError:
            with open(f"{self.__img_dir_path}/img_list.txt", 'w') as f:
                self.__img_list = [i for i in os.listdir(self.__img_dir_path) if '.jpg' in i]
                f.write('\n'.join(self.__img_list))

        self.__total_img_num = len(self.__img_list)
        
        try:
            with open(f"{self.__img_dir_path}/label_progress.yaml", 'r') as f:
                self.__progress = yaml.load(f.read())
        except FileNotFoundError:
            with open(f"{self.__img_dir_path}/label_progress.yaml", 'w') as f:
                self.__progress = {
                    'ind': 0,
                }
                f.write(yaml.dump(self.__progress))

    def get_progress(self):
        return self.__progress['ind']
    
    def set_progress(self, ind):
        self.__progress['ind'] = ind
        with open(f"{self.__img_dir_path}/label_progress.yaml", 'w') as f:
            f.write(yaml.dump(self.__progress))

    def __read_img(self, img_ind):
        assert img_ind < self.__total_img_num, f"Fail!\nImage Num {img_ind} exceed threshold {self.__total_img_num}\n"
        assert img_ind >= 0, f"Fail!\nImage Num {img_ind} less than 0\n"

        img = Image.open(f"{self.__img_dir_path}/{self.__img_list[img_ind]}".strip())
        image_rgb = np.array(img.convert("RGB"))
        image_bgr = image_rgb[:, :, ::-1]

        # image_bgr = cv2.imread(f"{self.__img_dir_path}/{self.__img_list[img_ind]}".strip())

        # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return image_bgr, image_rgb

    def __len__(self):
        return self.__total_img_num
    
    def __getitem__(self, key):
        return self.__read_img(img_ind=key)


class BirdObserver(object):
    def __init__(self, dataset_path='../data/img_data/integr', label_dir='../results/integr', model='nas', port=2333):
        # use yaml to load model info
        with open('config.yaml', 'r') as f:
            config = yaml.load(f.read())
        self.detect_model = ObjectDetectionModel(model_name=config[model]['model_name'],
                                                 category_index=config[model]['category_index'],
                                                 port=port)
        
        self.dataset_path = dataset_path
        self.label_dir = label_dir
        self.dataset = DataSet(self.dataset_path)

        self.vott_dict = None

        self.to_stop = False

    def stop(self, signalnum, frame):
        self.to_stop = True

    def observe(self, num_records_if_not_full=100, full_records=True):
        for result in self.records_generator(num_records_if_not_full, full_records):
            self.write_result(result)
            self.cache_result_vott(result)

            if self.to_stop == True:
                with open(f"{self.label_dir}/images.json", 'w') as f:
                    json.dump(self.vott_dict, f)
                break
        else:
            with open(f"{self.label_dir}/images.json", 'w') as f:
                json.dump(self.vott_dict, f)
        
    def records_generator(self, num_records_if_not_full=100, full_records=True, grid=False):
        if full_records:
            step = 1
            ar_sample_ind_to_observe = range(self.dataset.get_progress(), len(self.dataset))
        else:
            step = len(self.dataset) // num_records_if_not_full
            ar_sample_ind_to_observe = np.arange(num_records_if_not_full).astype(int) * step
            ar_sample_ind_to_observe = ar_sample_ind_to_observe[ar_sample_ind_to_observe >= self.dataset.get_progress()]

        for sample_ind in ar_sample_ind_to_observe:
            try:
                image_bgr, image_rgb = self.dataset[sample_ind]
                
                if not grid:
                    label_out = self.detect_model.run_inference_for_single_image(image_rgb)
                else:
                    pass

                self.dataset.set_progress(self.dataset.get_progress()+step)

                yield {
                    'image_bgr': image_bgr,
                    'image_rgb': image_rgb,
                    'label_out': label_out,
                    'sample_ind': sample_ind,
                    'image_width': image_bgr.shape[1],
                    'image_height': image_bgr.shape[0],
                }

            except AssertionError as e:
                print(e.args[0])

                return 1

    def write_result(self, result):
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(os.path.join(self.label_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.label_dir, 'labels'), exist_ok=True)

        cv2.imwrite(os.path.join(
            self.label_dir, 'images',
            f"{os.path.basename(self.dataset_path).split('.')[0]}_{result['sample_ind']:0>6}.jpg"
        ), result['image_bgr'])

        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = os.path.basename(self.dataset_path)
        ET.SubElement(annotation, 'filename').text = f"{os.path.basename(self.dataset_path).split('.')[0]}_{result['sample_ind']}.jpg"
        ET.SubElement(annotation, 'segmented').text = '0'
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(result['image_width'])
        ET.SubElement(size, 'height').text = str(result['image_height'])
        ET.SubElement(size, 'depth').text = '3'
        for label, box in zip(result['label_out']['detection_labels'],
                              result['label_out']['detection_boxes']):
            object_tag = ET.SubElement(annotation, 'object')
            ET.SubElement(object_tag, 'name').text = label
            # ET.SubElement(object_tag, 'name').text = 'intermidiate_egret'
            ET.SubElement(object_tag, 'pose').text = 'Unspecified'
            ET.SubElement(object_tag, 'truncated').text = '0'
            ET.SubElement(object_tag, 'difficult').text = '0'
            
            bndbox = ET.SubElement(object_tag, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(box[1]))
            ET.SubElement(bndbox, 'ymin').text = str(int(box[0]))
            ET.SubElement(bndbox, 'xmax').text = str(int(box[3]))
            ET.SubElement(bndbox, 'ymax').text = str(int(box[2]))

        xml_string = ET.tostring(annotation)
        root = etree.fromstring(xml_string)
        xml_string = etree.tostring(root, pretty_print=True)
        
        save_path = os.path.join(self.label_dir, 'labels',
                    f"{os.path.basename(self.dataset_path).split('.')[0]}_{result['sample_ind']:0>6}.xml")

        with open(save_path, 'wb') as f_temp:
            f_temp.write(xml_string)

        logging.info("📦  label saved")

    def cache_result_vott(self, result):
        if self.vott_dict is None:
            try: 
                with open(f"{self.label_dir}/images.json", 'r') as f:
                    self.vott_dict = json.load(f)
            except FileNotFoundError:
                self.vott_dict = {"frames": {},
                            "framerate":"1",
                            "inputTags":"little_egret,cattle_egret,great_egret,intermidiate_egret",
                            "suggestiontype":"track",
                            "scd":False,
                            "visitedFrames":[],
                            "tag_colors":["#117B76", "#ED62A7", "#FC6554", "#FAE047"]}
                # with open(f"{self.label_dir}/images.json", 'w') as f:
                    # json.dump(self.vott_dict, f)

        try:
            visited_frame_now = self.vott_dict['visitedFrames'][-1] + 1
        except IndexError:
            visited_frame_now = 0
        try:
            id_now = self.vott_dict['frames'][str(self.vott_dict['visitedFrames'][-1])][-1]['id'] + 1
        except IndexError:
            id_now = 0

        self.vott_dict['frames'][str(visited_frame_now)] = []
        self.vott_dict['visitedFrames'].append(visited_frame_now)

        for i, (label, box) in enumerate(zip(result['label_out']['detection_labels'], result['label_out']['detection_boxes'])):
            y1, x1, y2, x2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            width = result['image_width']
            height = result['image_height']

            tags = [label]
            # tags = ['intermidiate_egret']

            self.vott_dict['frames'][str(visited_frame_now)].append(
                    {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "id": id_now + i, "width": width, "height": height, 
                        "type": "Rectangle", "tags": tags, "name": i+1,
                    }
            )
        
        # with open(f"{self.label_dir}/images.json", 'w') as f:
            # json.dump(vott_dict, f)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='../data/img_data/greegr',
					help='The dataset to make labels from')
    parser.add_argument('-l', '--label_dir', type=str, default='../results/greegr')
    parser.add_argument('-m', '--model', type=str, default='nas',
                    help='The object detection model')
    parser.add_argument('-p', '--port', type=int, default=2333,
                    help='server port')
    args = parser.parse_args()

    bird_observer = BirdObserver(dataset_path=args.dataset, model=args.model, label_dir=args.label_dir, port=args.port)
    signal.signal(signal.SIGINT, bird_observer.stop)
    signal.signal(signal.SIGTERM, bird_observer.stop)
    signal.signal(signal.SIGABRT, bird_observer.stop)
    bird_observer.observe()
