#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
#import matplotlib.pyplot as plt
#import numpy as np


#%%
yolo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(yolo_dir)
cwd = os.getcwd()
from darkflow.net.build import TFNet
import cv2
import sys 

class yolo(object):
    
    modeldir = yolo_dir
    parentdir = cwd

        
    def __init__(self,):
        print("yolo model created")
        
    def load(self,model='tiny-yolo',threshold=0.22):
        os.chdir(yolo_dir)
        
        model_list =['tiny-yolo','tiny-yolo-voc','yolo']
        if model in model_list: 
            self.options = {"model": "cfg/"+model+".cfg", "load": "bin/"+model+".weights", "threshold": threshold,"gpu":0.9}
        else:
            raise ValueError('model passed in is not in supported model list. Please pass in correct model.(tiny-yolo;tiny-yolo-voc,yolo)')
            
        self.tfnet = TFNet(self.options)
        os.chdir(self.parentdir)
        #print(self.parentdir)
        return None
    
    def predict(self,img_path):
        imgcv = cv2.imread(img_path,1)
        result = self.tfnet.return_predict(imgcv)
        return result
    
    def predict_imgcv(self,imgcv):
        result = self.tfnet.return_predict(imgcv)
        return result
    
    def predict_imgcv_list(self,imgcvs,threshold=0.3):
        buffer_inp = list()
        buffer_pre = list()
        boxex_list = list()
        
        for fr in imgcvs:
            preprocessed = self.tfnet.framework.preprocess(fr)
            buffer_pre.append(preprocessed)
            buffer_inp.append(fr)
        
        feed_dict = {self.tfnet.inp: buffer_pre}
        net_out = self.tfnet.sess.run(self.tfnet.out, feed_dict)
        for im,single_out in zip(buffer_inp, net_out):
            h, w, _ = im.shape
            boxes = self.tfnet.framework.findboxes(single_out)
            boxesInfo = list()
            for box in boxes:
                tmpBox = self.tfnet.framework.process_box(box, h, w, threshold)
                if tmpBox is None:
                    continue
                boxesInfo.append({
                    "label": tmpBox[4],
                    "confidence": tmpBox[6],
                    "topleft": {
                        "x": tmpBox[0],
                        "y": tmpBox[2]},
                    "bottomright": {
                        "x": tmpBox[1],
                        "y": tmpBox[3]}
                })
            
            ## add processed boxex into a list   
            boxex_list.append(boxesInfo)
	
        return boxex_list
    
    def draw_save_pic(self,img_path,out_path):
        imgcv = cv2.imread(img_path,1)
        result = self.tfnet.return_predict(imgcv)
        draw_box(result,imgcv)
        cv2.imwrite( os.path.join(out_path), imgcv);
        return imgcv

    def draw_return_pic(self,frame):
        buffer_inp = list()
        buffer_pre = list()
        results = list()
        for fr in frame:
            preprocessed = self.tfnet.framework.preprocess(fr)
            buffer_pre.append(preprocessed)
            buffer_inp.append(fr)
        
        feed_dict = {self.tfnet.inp: buffer_pre}
        net_out = self.tfnet.sess.run(self.tfnet.out, feed_dict)
        for img, single_out in zip(buffer_inp, net_out):
            postprocessed = self.tfnet.framework.postprocess(single_out, img, False)
            results.append(postprocessed)
	
        return results

    def demo(self,option='camera'):
        self.tfnet.FLAGS.demo = option
        self.tfnet.camera()
        
        return None
    
    def video_return_json(self,option):
        camera = cv2.VideoCapture(0) ## 0 is the default camera
        while camera.isOpened():
            _,frame = camera.read()
            if frame is None:
                print('\n End of Video')
            
            result = self.tfnet.return_predict(frame)
            sys.stdout.write('%s\r' % result[0])
            sys.stdout.flush()
            #cv2.imshow('',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        camera.release()
        cv2.destroyAllWindows()          
          

## other helper functions       
def draw_box(result,imgcv):
    for i,r in enumerate(result):
        x1 = r['topleft']['x']
        y1 = r['topleft']['y']
        x2 = r['bottomright']['x']
        y2 = r['bottomright']['y']
        cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,255,0),4)
    return imgcv
    







