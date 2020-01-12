import math
import numpy as np 
import tensorflow as tf

def msra_stddev(x, k_h, k_w): 
    return 1/math.sqrt(0.5*k_w*k_h*x.get_shape().as_list()[-1])

def mse_ignore_nans(preds, targets, **kwargs):
    #Computes mse, ignores targets which are NANs
    
    # replace nans in the target with corresponding preds, so that there is no gradient for those
    targets_nonan = tf.where(tf.is_nan(targets), preds, targets)
    return tf.reduce_mean(tf.square(targets_nonan - preds), **kwargs)

def conv2d(input_, output_dim, 
        k_h=3, k_w=3, d_h=2, d_w=2, msra_coeff=1,
        name="conv2d"):

    # USAGE : conv2d(curr_inp, param['out_channels'], k_h=param['kernel'], k_w=param['kernel'], d_h=param['stride'], d_w=param['stride'], name=name + str(nl), msra_coeff=msra_coeff)))
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=msra_coeff * msra_stddev(input_, k_h, k_w))) # Create new variable under scope name
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0)) # Same here for bias

        return tf.nn.bias_add(tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME'), b)

def conv2d_transpose(input_, output_shape, 
        k_h=3, k_w=3, d_h=2, d_w=2, msra_coeff=1,
        name="deconv2d"):
        output_dim = output_shape[-1]
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=msra_coeff * msra_stddev(input_, k_h, k_w)))
    

        return tf.nn.conv2d_transpose(input_, w,output_shape=output_shape, strides=[1, d_h, d_w, 1], padding='SAME')

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, name='linear', msra_coeff=1):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=msra_coeff * msra_stddev(input_, 1, 1)))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b
    
def conv_encoder(data, params, name, msra_coeff=1):
    layers = []
    for nl, param in enumerate(params):
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
            
        layers.append(lrelu(conv2d(curr_inp, param['out_channels'], k_h=param['kernel'], k_w=param['kernel'], d_h=param['stride'], d_w=param['stride'], name=name + str(nl), msra_coeff=msra_coeff)))
        
    return layers[-1]


        
def fc_net(data, params, name, last_linear = False, return_layers = [-1], msra_coeff=1):
    layers = []
    for nl, param in enumerate(params):
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
        
        if nl == len(params) - 1 and last_linear:
            layers.append(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff))
        else:
            layers.append(lrelu(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff)))
            
    if len(return_layers) == 1:
        return layers[return_layers[0]]
    else:
        return [layers[nl] for nl in return_layers]

def flatten(data):
    return tf.reshape(data, [-1, np.prod(data.get_shape().as_list()[1:])])


##=======================================================================================
##                      TEST SEGMENTATION UNET
##=======================================================================================
def conv_encoder_aux(data, params, name, msra_coeff=1):
    layers = []
    for nl, param in enumerate(params):
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
        layers.append(lrelu(conv2d(curr_inp, param['out_channels'], k_h=param['kernel'], k_w=param['kernel'], d_h=param['stride'], d_w=param['stride'], name=name + str(nl), msra_coeff=msra_coeff)))
        
    return layers

def conv_decoder(data,layers_encoder, params, name, msra_coeff=1):
    layers = []
    for nl, param in reversed(enumerate(params)):
        if nl == len(params)-1:
            continue
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
        outputShape = layers_encoder[nl].get_shape()
        layers.append(lrelu(conv2d_transpose(curr_inp, param['out_channels'], k_h=param['kernel'], k_w=param['kernel'], d_h=param['stride'], d_w=param['stride'], name=name + str(nl), msra_coeff=msra_coeff)))
        curr_inp = tf.concat([layers_encoder[nl],layers[-1]])  
        layers.append(tf.nn.max_pool(curr_inp,ksize=(2,2),strides=(2,2),padding="same"))
    return layers[-1]

def UNET(data,params,name,msra_coeff=1):
    outputEncoded = my_ops.conv_encoder(data, params, 'encoder', msra_coeff=msra_coeff)
    dataAux = conv2d(outputEncoded[-1], 32, 
        k_h=2, k_w=2, d_h=2, d_w=2, msra_coeff=msra_coeff,
        name="conv2dAUX"):
    outputDecoded = conv_decoder(p_img_conv,p_img_convs[-1])
    outputDecoded = conv2d(outputEncoded, 1, 
        k_h=2, k_w=2, d_h=1, d_w=1, msra_coeff=msra_coeff,
        name="UNETFINALCONV")
    return outputDecoded



##=======================================================================================
##                      TEST SEGMENTATION RCNN
##=======================================================================================
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn.config import Config


# Root directory of the project
ROOT_DIR = os.path.abspath("rcnn")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib



# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # COCO has 80 classes

config = CocoConfig()
config.cla
# config = Config()
config.display()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=False)


