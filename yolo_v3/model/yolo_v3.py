"""
write by: Li Shuai
Time: 2018/12/04
Localization: NIO DLLab, Shanghai
yolov3 model body for self-driving experiments
"""
import numpy as np
import  keras.backend as K
from keras.models import load_model

class YOLO_V3(object):
    def __init__(self,obj_threshold,nms_threshold):
        self._objthres=obj_threshold
        self._nms_thres=nms_threshold
        self._yolo=load_model('data/yolo_v3.h5')

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def _process_feats(self,out,anchors,mask):
        """
        process the output features
        :param out: tensor(N,N,3,4+1+80), output feature map of yolo
        :param anchors: list,anchors for box
        :param mask: list,mask for anchors
        :return:
        the instance argument should be match with local custom data
        boxes:ndarray(N,N,3,4),x,y,w,h for each box
        box_confidence:ndarray(N,N,3,1) confidence for each box
        box_class_probs:adarray(N,N,3,80),class probs for per box
        """