from django.shortcuts import render
from .forms import ImageUploadForm
###########################################
# for garbage collection
import gc

# for warnings
import warnings
warnings.filterwarnings("ignore")

# utility libraries
import os
import copy
import tqdm
import numpy as np 
import pandas as pd 
import cv2, random, time, shutil, csv
import tensorflow as tf
import math

# keras libraries
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing.image import load_img

# importing InceptionV3 and its Preprocessor
from keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input

# importing Xception and its Preprocessor
from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input

# importing InceptionResNetV2 and its Preprocessor
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input

# Creating List of Models and its Preprocessors
models = [InceptionV3,  InceptionResNetV2, Xception]
preprocs = [inception_preprocessor,  inc_resnet_preprocessor, xception_preprocessor]

import pandas as pd
data_df = pd.read_csv('./labels.csv')
class_names=sorted(data_df['breed'].unique())

# FEATURE EXTRACTION OF VALIDAION AND TESTING ARRAYS
#    Same as above except image augmentations function.
def get_val_features(model_name, data_preprocessor, data):

    dataset = tf.data.Dataset.from_tensor_slices(data)
    ds = dataset.batch(30)
    input_size = data.shape[1:]
    #Prepare pipeline.
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False, input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    #Extract feature.
    feature_maps = feature_extractor.predict(ds, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps

def get_concat_features(feat_func, models, preprocs, array):
    feats_list = []
    for i in range(len(models)):
        print(f"\nStarting feature extraction with {models[i].__name__} using {preprocs[i].__name__}\n")
        # applying the above function and storing in list
        feats_list.append(feat_func(models[i], preprocs[i], array))
    
    # features concatenating
    final_feats = np.concatenate(feats_list, axis=-1)
    # memory saving
    del(feats_list, array)
    gc.collect()
    return final_feats

from keras.models import load_model
trained_models=[]
for i in range(3):
    model = load_model(f'./My_Model_{i}')
    trained_models.append(model)

#################################################
## File Handeling Functions
def handle_uploaded_file(f):
	with open('img.jpg', 'wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)


def home(request):
	return render(request,'home.html')

def imageprocess(request):
	form = ImageUploadForm(request.POST, request.FILES)
	if form.is_valid():
		handle_uploaded_file(request.FILES['image'])
		img_g = load_img('./img.jpg', target_size = (331,331,3))
		img_g = np.expand_dims(img_g, axis=0)
		## Getting Features
		test_f = get_concat_features(get_val_features, models, preprocs, img_g)
		## Predicting Output
		y_pred=trained_models[0].predict(test_f, batch_size=128)/3
		for dnn in trained_models[1:]:
		 y_pred+=dnn.predict(test_f, batch_size=128)/3
		#print(f'Predicted: {}')
		res=[]
		res.append(class_names[np.argmax(y_pred[0])])
		print(res)
		return render(request,'result.html', { 'res' : res[0] })

	return render(request,'home.html')
	

