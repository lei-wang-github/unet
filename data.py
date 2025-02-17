from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
from skimage import color
import time
import tensorflow as tf
from PIL import Image
import configparser

config = configparser.ConfigParser()
config.read('configuration.txt')
img_height = int(config['data attributes']['image_height'])
img_width = int(config['data attributes']['image_width'])
imageFileExtension = config['data attributes']['imageExt']

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.05] = 1
        mask[mask <= 0.05] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (img_height,img_width),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def testGenerator(test_path,num_image = 30,target_size = (img_height,img_width),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d"%i + imageFileExtension),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def testImageSingleClassGenerator(test_path, batch_size=1, target_size=(img_height, img_width), flag_multi_class=False,
                  color_mode='grayscale'):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        directory=test_path,
        classes=['test'],
        shuffle=False, #alphanumeric order
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode=None)

    for im in test_generator:
        im = im
        yield im


def test_image_prep(image_file_path, target_size=(img_height, img_width), flag_multi_class=False, as_gray=True):
    img = io.imread(image_file_path, as_gray=as_gray)
    img = img / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
    img = np.reshape(img, (1,) + img.shape)
    return img


def load_model_lite_single_predict(interpreter, tf_image):
    # Load TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(tf_image, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    t1 = time.time()
    interpreter.invoke()
    print("TFLITE ouput: out" + imageFileExtension + " elapsed-time =", time.time() - t1)
  
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    outputs = interpreter.get_tensor(output_details[0]['index'])
    
    output = outputs[0]
    img = output[:,:,0]
    io.imsave('out' + imageFileExtension, img)
    return img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*"%image_prefix + imageFileExtension))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def labelVisualize1(num_class, img, color_dict = COLOR_DICT):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    imgB = io.imread(os.path.join(save_path, "0" + imageFileExtension), as_gray=True)
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        imgB = cv2.imread(os.path.join(save_path, "%d"%i + imageFileExtension))
        imgB = cv2.resize(imgB, (int(img_width), int(img_height)))
        io.imsave(os.path.join(save_path,"%d_predict"%i + imageFileExtension),img)
        imgM = cv2.imread(os.path.join(save_path, "%d_predict"%i + imageFileExtension))
        overlay = cv2.resize(imgM, (int(img_width), int(img_height)))
        overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_HOT)
        #np.multiply(overlay, [1.0, 0.0, 0.0], out=overlay, casting='unsafe')
        added_image = cv2.addWeighted(imgB, 0.5, overlay, 0.5, 0)
        io.imsave(os.path.join(save_path, "%d_predict_combined"%i + imageFileExtension), added_image)
        
def saveTestResult(save_path, npyfile):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        directory=save_path,
        classes=['test'],
        shuffle=False, #alphanumeric order
        target_size=(int(img_width), int(img_height)),
        batch_size=1,
        color_mode='grayscale',
        class_mode=None)
    for resultImgArray,testImgTensor, filename in zip(npyfile, test_generator, test_generator.filenames):
        x = testImgTensor
        y = resultImgArray
        z = filename
        z = os.path.splitext(os.path.basename(z))[0]
        x = tf.squeeze(x, [0]) #4D tensor to 3D array
        x_PIL = tf.keras.preprocessing.image.array_to_img(x, scale=True) #and scale to [0-255]
        y_PIL = tf.keras.preprocessing.image.array_to_img(y, scale=True)
        x_CV2 = cv2.cvtColor(np.array(x_PIL), cv2.COLOR_RGB2BGR)
        y_CV2 = cv2.cvtColor(np.array(y_PIL), cv2.COLOR_RGB2BGR)
        x_CV2 = cv2.resize(x_CV2, (int(img_width), int(img_height)))
        y_CV2 = cv2.resize(y_CV2, (int(img_width), int(img_height)))
        y_CV2 = cv2.applyColorMap(y_CV2, cv2.COLORMAP_HOT)
        added_image = cv2.addWeighted(x_CV2, 0.5, y_CV2, 0.5, 0)
        io.imsave(os.path.join(save_path, z + "_predict_combined_test" + imageFileExtension), added_image)

