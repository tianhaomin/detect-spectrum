# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:56:41 2017

@author: Administrator
"""

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image
im = Image.open('F:/project/operatordata/pics/try/3d_fig.png')
im1 = im.convert('L')
im1.show()
im.show()
im_array = np.array(im)
