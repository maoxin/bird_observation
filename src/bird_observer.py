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


class Video(object):
    def __init__(self, video_path='../data/video/MVI_7739.mp4'):
        self.video = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self, frame_num):
        assert frame_num < self.total_frames, f"Fail!\nFrame Num {frame_num} exceed threshold {self.total_frames}\n"
        assert frame_num >= 0, f"Fail!\nFrame Num {frame_num} less than 0\n"

        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, image_bgr = self.video.read()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        return image_bgr, image_rgb

    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, key):
        return self.read_frame(frame_num=key)

class ImageDir(object):
    pass

class BirdObserver(object):
    def __init__(self, dataset_path='../data/video/MVI_7739.mp4', label_dir='../results', model='nas'):
        # use yaml to load model info
        with open('config.yaml', 'r') as f:
            config = yaml.load(f.read())
        self.detect_model = ObjectDetectionModel(model_name=config[model]['model_name'],
                                                 category_index=config[model]['category_index'])
        
        self.dataset_path = dataset_path
        self.label_dir = label_dir
        self.dataset = DataSet(self.dataset_path)

    def observe(self):
        results = []
        for result in self.records_generator():
            self.write_result(result)
            results.append(result)
        
        vott_dict = {"frames": {},
                        "framerate":"1",
                        "inputTags":"little_egret,cattle_egret,great_egret,Ondatra_zibethicus,Columba_palumbus,Carex_lupulina,Lottia_gigantea",
                        "suggestiontype":"track",
                        "scd":False,
                        "visitedFrames":list(range(len(results))),
                        "tag_colors":["#ce0d82"]}
        
        for i, result in enumerate(results):
            vott_dict['frames'][str(i)] = []

            for label, box in zip(result['label_out']['detection_labels'],
                result['label_out']['detection_boxes']):

                y1, x1, y2, x2 = box
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                width = result['image_width']
                height = result['image_height']

                tags = [label]
                
                vott_dict['frames'][str(i)].append(
                    {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "id": i, "width": width, "height": height, 
                        "type": "Rectangle", "tags": tags, "name": i + 1,
                    }
                )

        with open("../results/images.json", 'w') as f:
            json.dump(vott_dict, f)

    def records_generator(self, num_records=100, grid=False):
        step = len(self.dataset) // 100
        ar_sample_ind_to_observe = np.arange(100).astype(int) * step

        for sample_ind in ar_sample_ind_to_observe:
            try:
                image_bgr, image_rgb = self.dataset[sample_ind]
                
                if not grid:
                    label_out = self.detect_model.run_inference_for_single_image(image_rgb)
                else:
                    pass

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

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='../data/video/MVI_7739.mp4',
					help='The dataset to make labels from')
    parser.add_argument('-m', '--model', type=str, default='nas',
                    help='The object detection model')
    args = parser.parse_args()

    bird_observer = BirdObserver(dataset_path=args.dataset, model=args.model)
    bird_observer.observe()
