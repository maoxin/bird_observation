import sys
from os.path import realpath, dirname, join
sys.path.append(join(dirname(dirname(realpath(__file__))), 'src'))
import numpy as np
from utils import grid_img

class TestGridImg(object):
    def test_overlap(self):
        for k in range(10):
            img_height, img_width = np.random.randint(3, 31, size=(2,))
            ar_img = np.random.randn(img_height, img_width)
            
            for l in range(10):
                kernal_height = np.random.randint(2, img_height)
                kernal_width = np.random.randint(2, img_width)
                stride_height = np.random.randint(1, kernal_height)
                stride_width = np.random.randint(1, kernal_width)

                sub_imgs = grid_img(ar_img, kernal_width, kernal_height, stride_width, stride_height)
                for i in range(len(sub_imgs) - 1):
                    for j in range(len(sub_imgs[0]) - 1):
                        assert (sub_imgs[i][j].ar_img[:, stride_width:] == sub_imgs[i][j+1].ar_img[:, :kernal_width - stride_width]).all()
                        assert (sub_imgs[i][j].ar_img[stride_height:, :] == sub_imgs[i+1][j].ar_img[:kernal_height - stride_height, :]).all()
                        assert (sub_imgs[i][j].ar_img[stride_height:, stride_width:] == sub_imgs[i+1][j+1].ar_img[:kernal_height - stride_height, :kernal_width - stride_width]).all()
