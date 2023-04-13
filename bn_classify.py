# -*- coding: utf-8 -*-
"""BN_classify.ipynb

Original file is located at
    https://colab.research.google.com/drive/1c_bgABtJ5_4X6DTXQ5o_4LcOddQ3rzcK
"""

import fastbook
from fastbook import *
from fastai.vision.widgets import *
path = Path('.')
path.ls(file_exts='.pkl')
learn = load_learner(path/'black_normal_classifier.pkl')

def prediction(image):
  img = PILImage.create(image)
  #with out_pl : display(img.to_thumb(128))
  #out_pl.clear_output()
  pred, _, prob = learn.predict(img)
  return int(_)

img_path = './WIN_20230331_12_25_29_Pro.jpg'
out = prediction(img_path)
print(out)
