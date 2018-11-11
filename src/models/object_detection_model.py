import os
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.expanduser('~/.virtualenvs/env_tensorflow/lib/python3.6/site-packages/tensorflow/models/research'))
sys.path.append(os.path.expanduser('~/.virtualenvs/env_tensorflow/lib/python3.6/site-packages/tensorflow/models/research/object_detection'))
from object_detection.utils import ops as utils_ops

import logging


class ObjectDetectionModel(object):
    def __init__(self, model_name, category_index, port=2333):
        self.cluster = tf.train.ClusterSpec({"frcnn": ["192.168.1.200:{}".format(port)]})

        # self.model_name = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
        # self.model_name = 'faster_rcnn_nas_coco_2018_01_28'
        self.model_name = model_name
        self.model_file = f"{self.model_name}.tar.gz"

        self.download_base = 'http://download.tensorflow.org/models/object_detection/'

        self.path_to_frozon_graph = f"{self.model_name}/frozen_inference_graph.pb"
        
        self.num_classes = 90

        if not os.path.isfile(self.model_file):
            logging.info(f"📦  Download Model '{self.model_name}'...")
            opener = urllib.request.URLopener()
            opener.retrieve(f"{self.download_base}{self.model_file}", self.model_file)
            tar_file = tarfile.open(self.model_file)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, os.getcwd())
            logging.info("📦  Download completed.")
        
        logging.info(f"📤  Load Model '{self.model_name}'...")

        with tf.device("/job:frcnn/task:0"):
            self.detection_graph = tf.Graph()
            
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.path_to_frozon_graph, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    self.tensor_dict = {}
                    for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes',
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                    self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        
        logging.info(f"📤  Load completed.")

        self.sess = tf.Session("grpc://192.168.1.200:{}".format(port), graph=self.detection_graph)

        # self.category_index = {
            # 3: 'Private Car',
            # 4: 'Motorcycles',
            # 6: 'Double Decker Bus',
            # 7: 'Tram',
            # 8: 'Truck',
        # }
        self.category_index = category_index

    def run_inference_for_single_image(self, image):
        logging.info(f"🔍  Inference begin...")
        # with self.detection_graph.as_default():
        # with tf.Session() as sess:
            # Run inference
        output_dict = self.sess.run(self.tensor_dict,
                                feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)[:output_dict['num_detections']]
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0][:output_dict['num_detections']]
        output_dict['detection_scores'] = output_dict['detection_scores'][0][:output_dict['num_detections']]

        # rescale to image
        output_dict['detection_boxes'][:, 0] = output_dict['detection_boxes'][:, 0] * image.shape[0]
        output_dict['detection_boxes'][:, 1] = output_dict['detection_boxes'][:, 1] * image.shape[1]
        output_dict['detection_boxes'][:, 2] = output_dict['detection_boxes'][:, 2] * image.shape[0]
        output_dict['detection_boxes'][:, 3] = output_dict['detection_boxes'][:, 3] * image.shape[1]

        output_dict['detection_boxes'] = output_dict['detection_boxes'].astype('int')

        class_filter = np.isin(output_dict['detection_classes'], list(self.category_index.keys()))
        score_filter = output_dict['detection_scores'] >= 0.5
        mix_filter = (class_filter & score_filter)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][mix_filter]
        output_dict['detection_classes'] = output_dict['detection_classes'][mix_filter]
        output_dict['detection_scores'] = output_dict['detection_scores'][mix_filter]
        output_dict['num_detections'] = len(output_dict['detection_scores'])
        output_dict['detection_labels'] = [self.category_index[i] for i in output_dict['detection_classes']]
        
        logging.info(f"🔍   Inference completed.")
        
        return output_dict

    def close(self):
        self.sess.close()
