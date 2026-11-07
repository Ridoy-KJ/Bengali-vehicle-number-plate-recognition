import os

import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import tensorflow as tf
import pytesseract as pt
from autoocr import AutoOCR
from tensorflow.keras.preprocessing.image import load_img, img_to_array


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(APP_DIR, '..', 'models', 'object_detectionv3.h5'))
PREDICT_DIR = os.path.join(APP_DIR, 'static', 'predict')
ROI_DIR = os.path.join(APP_DIR, 'static', 'roi')
TESSDATA = os.environ.get('TESSDATA_PREFIX', r'C:\Program Files\Tesseract-OCR\tessdata')

os.makedirs(PREDICT_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        'Trained weights were not found at {}. '
        'Train InceptionResNetV2 with the Kaggle notebook and place '
        'object_detectionv3.h5 in the repository models/ directory.'.format(MODEL_PATH)
    )

model = tf.keras.models.load_model(MODEL_PATH)

def object_detection(path,filename):
    
    # Read image
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    
    # Data preprocessing
    image_arr_224 = img_to_array(image1)/255.0 # Convert to array & normalized
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    
    # Make predictions
    coords = model.predict(test_arr)
    
    # Denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    
    # Draw bounding on top the image
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    
    #convert into bgr
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(PREDICT_DIR, filename), image_bgr)
    return coords

def OCR(path,filename):
    img = np.array(load_img(path))
    cods = object_detection(path,filename)
    xmin ,xmax,ymin,ymax = cods[0]
    roi = img[ymin:ymax,xmin:xmax] 
    roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
    roi_path = os.path.join(ROI_DIR, filename)
    cv2.imwrite(roi_path, roi_bgr)
    # extract text from image
    # text = pt.image_to_string(roi,lang='ben')
    # print(text)
    oa = AutoOCR(lang='bangla')
    oa.set_datapath(TESSDATA)
    text = oa.get_text(roi_path)
    return text