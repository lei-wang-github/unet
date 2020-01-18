from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # works on windows 10 to force it to use CPU

from model import *
from data import *

RunWithGPU = False
PerformTraining = False
import numpy as np
if RunWithGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from skimage import color
import cv2
from keras.models import load_model

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
        
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
                    zca_whitening=True,
                    brightness_range=[0.2,1.8],
                    width_shift_range=0.10,
                    height_shift_range=0.10,
                    shear_range=0.05,
                    zoom_range=0.20,
                    horizontal_flip=True,
                    fill_mode='nearest')

if PerformTraining:
    myGene = trainGenerator(2,'data/Solar/train','image','label',data_gen_args,save_to_dir = None)
    
    model = unet()
    model_checkpoint = ModelCheckpoint('unet_Solar.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=200,epochs=15,callbacks=[model_checkpoint])

    testGene = testGenerator("data/Solar/test")
    results = model.predict_generator(testGene, 30, verbose=1)
    saveResult("data/Solar/test", results)

else:
    model = load_model('unet_Solar.hdf5')


test_single = test_image_prep('./0.png')
t1 = time.time()
result = model.predict(test_single)
print("elapsed-time =", time.time() - t1)
saveResult("./", result)


# tflite_convert --output_file=unet_Solar.tflite --keras_model_file=unet_Solar.hdf5 --input_shapes=1,512,512,1
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# tflite_quant_model = converter.convert()

test_single_image = test_image_prep('data/Solar/test/1.png')
test_lite_img = load_model_lite_single_predict('unet_Solar.tflite', test_single_image)

