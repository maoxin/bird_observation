import numpy as np

class SubImage(object):
    def __init__(self, ar_img, x, y):
        self.ar_img = ar_img
        self.x = x
        self.y = y
        self.height, self.width = self.ar_img.shape

    def __str__(self):
        return str(self.ar_img)


def grid_img(ar_img, kernal_width, kernal_height, stride_width, stride_height):
    """
    crop an image (ar_img) to a grid of subimage with a kernal with a stride
    """
    img_height, img_width = ar_img.shape

    n_h = int(np.ceil((img_width - kernal_width) / stride_width)) + 1
    # number of windows along y axis
    n_v = int(np.ceil((img_height - kernal_height) / stride_height)) + 1
    # number of windows along x axis

    sub_imgs = []
    for i in range(n_h):
        sub_imgs_h = []
        for j in range(n_v):
            x = j * stride_width
            y = i * stride_height
            sub_img = ar_img[y: y + kernal_height, x: x + kernal_width]
            # for an array foo with shape (2, ), foo[0:5] will return foo[0:1]

            sub_imgs_h.append(SubImage(sub_img, x, y))
        sub_imgs.append(sub_imgs_h)
        
    return sub_imgs


