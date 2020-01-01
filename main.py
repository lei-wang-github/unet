from __future__ import print_function

from model import *
from data import *

import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from skimage import color
import cv2
from keras.models import load_model

import tensorflow as tf

#
# def prepare(file_path):
#     img_size = 256  # 50 in txt-based
#     img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
#     img = img / 255
#     img = cv2.resize(img, (img_size, img_size))  # resize image to match model's expected sizing
#     img = np.reshape(img, img.shape + (1,)) if (not False) else img
#     img = np.reshape(img, (1,) + img.shape)
#     return img
#
#
# model = load_model('unet_membrane.hdf5')
# prediction = model.predict([prepare('0.png')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
# print("Prediction shape = ", prediction.shape)
# prediction = prediction[0,:,:,:]
# prediction = labelVisualize1(2, prediction)
# print("Prediction shape = ", prediction.shape)
# prediction = prediction[:,:,0]
# print("Prediction shape = ", prediction.shape)
# cv2.imwrite('0_prediction.png', prediction)

# load the image
background = cv2.imread('0.png')
# print("background after read", background.shape)
overlay = cv2.imread('0_prediction.png')
# print("overlay after read", overlay.shape)
# overlay = cv2.resize(overlay, (int(512),int(512)))
# print("overlay_new after resize", overlay_new.shape)
# overlay_new_color = color.gray2rgb(overlay_new)
# print("overlay_new_color after convert to rgb", overlay_new_color.shape)
# red_multiplier = [0.0, 0.0, 1.0]
# np.multiply(overlay, [0.0, 0.0, 1.0], out=overlay, casting='unsafe')
## overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_HOT)
# print("overlay_new_color after red_multiplier", overlay_new_color.shape)
# added_image = cv2.addWeighted(background,0.8,overlay,0.8,0)

# cv2.imwrite('combined.png', added_image)
# cv2.imshow("Output", added_image)
# cv2.waitKey(0)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/TrainSolar','image','label',data_gen_args,save_to_dir = 'data/membrane/TrainSolar/save')

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=8,callbacks=[model_checkpoint])

# model = load_model('unet_membrane.hdf5')

testGene = testGenerator("data/membrane/TestSolar")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/TestSolar",results)
