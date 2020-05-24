from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # works on windows 10 to force it to use CPU

import configparser
from model import *
from data import *

config = configparser.ConfigParser()
config.read('configuration.txt')

RunWithGPU = eval(config['execution mode']['RunWithGPU'])
PerformTraining = eval(config['execution mode']['PerformTraining'])

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
                    width_shift_range=0.01,
                    height_shift_range=0.01,
                    shear_range=0.03,
                    zoom_range=0.02,
                    horizontal_flip=True,
                    fill_mode='nearest')

imageFileExtension = config['data attributes']['imageExt']
train_path = "data/" + config['data path']['dataSourceName'] + "/train"
val_path = "data/" + config['data path']['dataSourceName'] + "/valid"
test_path = "data/" + config['data path']['dataSourceName'] + "/test"
numberOfTestImages = int(config['data path']['numberOfTestImages'])
modelSaveName = config['model type']['modelType'] + \
                "_" + config['data path']['dataSourceName'] + \
                config['data attributes']['image_height'] + \
                "x" + \
                config['data attributes']['image_width'] + \
                "-rd" + \
                config['model type']['modelReductionRatio']
usePretrainedWeights = eval(config['execution mode']['UsePretrainedWeights'])
modelSaveNameExt = ".hdf5"

if PerformTraining:
    trainGen = trainGenerator(2, train_path, 'image', 'label', data_gen_args, save_to_dir=None)
    valGen = trainGenerator(2, val_path, 'image', 'label', data_gen_args, save_to_dir=None)
    
    modelFunctionName = config['model type']['modelType'] + "()"
    model = eval(modelFunctionName)
    if tf.io.gfile.exists(modelSaveName + modelSaveNameExt) and usePretrainedWeights:
        model.load_weights(modelSaveName + modelSaveNameExt)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(modelSaveName + modelSaveNameExt, monitor='loss', verbose=1, save_best_only=True)
    ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_delta=0.00001)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=int(config["training settings"]["EarlyStopPatience"]))
    # tboard = tf.keras.callbacks.TensorBoard(log_dir='logs')
    
    model.fit_generator(trainGen,
                        steps_per_epoch=int(config['training settings']['steps_per_epoch']),
                        epochs=int(config['training settings']['N_epochs']),
                        callbacks=[model_checkpoint, EarlyStopping])

model = load_model(modelSaveName + modelSaveNameExt)

# test all the images in the test folder
testGene = testImageSingleClassGenerator(test_path)
#testGene = testGenerator(test_path)
results = model.predict_generator(testGene, numberOfTestImages, verbose=1)
#saveResult(test_path, results)
saveTestResult(test_path, results)

# predict a single image with normal tensorflow model.predict
testImageFile = test_path + "/" + "test" + "/" + "0" + imageFileExtension
test_single = test_image_prep(testImageFile)
t1 = time.time()
result = model.predict(test_single)
print("Tensorflow model predict elapsed-time =", time.time() - t1)
saveResult("./", result)

# predict a single image with tensorflow lite. The elapsed time report on PC CPU mode typically will be 7-8 times faster than on Wiper.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open(modelSaveName + ".tflite", "wb").write(tflite_quant_model)
print("Tensor lite model saved at " + modelSaveName + ".tflite")
interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)

test_single_image = test_image_prep(testImageFile)
test_lite_img = load_model_lite_single_predict(interpreter, test_single_image)
    

