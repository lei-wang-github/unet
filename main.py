from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # works on windows 10 to force it to use CPU

from model import *
from data import *

RunWithGPU = True
PerformTraining = True
import numpy as np
if RunWithGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model

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
        

data_gen_args = dict(rotation_range=0.2,
#                    zca_whitening=False,
#                   brightness_range=[0.8,1.2],
                    width_shift_range=0.05,
                    height_shift_range=0.050,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

if PerformTraining:
    myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)
    
    model = unet()
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_delta=0.00001)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=20)
    tboard = tf.keras.callbacks.TensorBoard(log_dir='logs')
    
    model.fit_generator(myGene, steps_per_epoch=300, epochs=400, callbacks=[model_checkpoint])

else:
    model = load_model('./unet_membrane.hdf5')
   
    if False:
        testGene = testGenerator("data/membrane/test")
        results = model.predict_generator(testGene, 30, verbose=1)
        saveResult("data/membrane/test", results)

# predict with normal tensorflow model.predict
test_single = test_image_prep('data/membrane/test/0.png')
t1 = time.time()
result = model.predict(test_single)
print("elapsed-time =", time.time() - t1)
saveResult("./", result)

if True:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
    
    test_single_image = test_image_prep('data/membrane/test/0.png')
    test_lite_img = load_model_lite_single_predict(interpreter, test_single_image)
    

