from __future__ import division

import gzip
import numpy as np
import argparse
import cv2
from math import cos, sin, pi
import sys

# Image size

WIDTH = 160
HIGH = 120
CHANN = 3


DATA_DIRS = "../2016-10-10"

class dataset:
    'read images and make them available for training, testing and validation'
    
    'edit threse constants and put your own generated files'

    def __init__(self):
        print "opening training, test and validation files"

        TRAIN_IMAGES = DATA_DIRS + '/imgs_trn.bin.gz'
        TRAIN_ANGLES =  DATA_DIRS + '/ang_trn.bin.gz'
        TEST_IMAGES = DATA_DIRS + '/imgs_tst.bin.gz'
        TEST_ANGLES = DATA_DIRS + '/ang_tst.bin.gz'
        VAL_IMAGES = DATA_DIRS + '/imgs_val.bin.gz'
        VAL_ANGLES = DATA_DIRS + '/ang_val.bin.gz'

        self.image_size = (HIGH * WIDTH * CHANN)
        self.images = []
        self.batch_types = ['train', 'test', 'validation']
        self.batch_size = 0

        # train data state info
        self.trn_angs = []
        self.trn_angs_ready = False
        self.trn_angs_idx = 0

        # test data state info
        self.tst_angs= []
        self.tst_angs_ready = False
        self.tst_angs_idx = 0

        # validation data state info
        self.val_angs = []
        self.val_angs_ready = False
        self.val_angs_idx = 0
        
        try:
            self.trn_imgs_file = gzip.open(TRAIN_IMAGES, 'rb')
            self.trn_ang_file = gzip.open(TRAIN_ANGLES, 'rb')
            self.tst_imgs_file = gzip.open(TEST_IMAGES, 'rb')
            self.tst_ang_file = gzip.open(TEST_ANGLES, 'rb')
            self.val_imgs_file = gzip.open(VAL_IMAGES, 'rb')
            self.val_ang_file = gzip.open(VAL_ANGLES, 'rb')
        except IOError as exc:
            print "%s" % (exc)
            sys.exit(2)
        print "done!"

    def close(self):
        self.trn_ang_file.close()
        self.trn_ang_file.close()
        self.tst_imgs_file.close()
        self.tst_ang_file.close()

    def next_batch(self, batch_size, batch_type):
        self.batch_size = batch_size 
        if batch_type not in self.batch_types:
            assert False
       
        if batch_type == 'train':
            # read images in batch
            '''DEBUG'''
            #print "entering train, angles flag: %d" % (self.trn_angs_ready)
            '''END DEBUG'''
            self.images,num_imgs_read = self._readimgs(self.trn_imgs_file)
            # read training angles once
            if self.trn_angs_ready == False:
                self.trn_angs = self._readangs(self.trn_ang_file)
                self.trn_angs_ready = True
            ang_low_idx = self.trn_angs_idx
            ang_up_idx = ang_low_idx + num_imgs_read
            if ang_up_idx > len(self.trn_angs):
                ang_low_idx = 0 
                ang_up_idx = num_imgs_read
            ret_angles = self.trn_angs[ang_low_idx:ang_up_idx]
            if len(ret_angles) == 0:
                ang_low_idx = 0
                self.trn_angs_idx = num_imgs_read
                ang_up_idx = num_imgs_read
                ret_angles = self.trn_angs[0:num_imgs_read]
            else: self.trn_angs_idx += num_imgs_read
            # return batch of images and angles
            return [self.images, ret_angles]
        
        elif batch_type == 'test':
            # read images in batch
            '''DEBUG'''
            #print "entering test, angles flag: %d" % (self.tst_angs_ready)
            '''END DEBUG'''
            self.images,num_imgs_read = self._readimgs(self.tst_imgs_file)
            # read training angles once
            if self.tst_angs_ready == False:
                '''debug'''
                #print "\tinside test angle false body"
                ''' end debug'''
                self.tst_angs = self._readangs(self.tst_ang_file)
                self.tst_angs_ready = True
            '''debug'''
            #print "leaving test angle false body, angles readed: %d" %(len(self.tst_angs))
            '''end debug'''
            ang_low_idx = self.tst_angs_idx
            ang_up_idx = ang_low_idx + num_imgs_read
            ret_angles = self.tst_angs[ang_low_idx:ang_up_idx]
            if len(ret_angles) == 0:
                ang_low_idx = 0
                self.tst_angs_idx = num_imgs_read
                ang_up_idx = num_imgs_read
                ret_angles = self.tst_angs[0:num_imgs_read]
            else: self.tst_angs_idx += num_imgs_read
            # return batch of images and angles
            return [self.images, ret_angles]
        
        elif batch_type == 'validation':
            # read images in batch
            '''DEBUG'''
            #print "entering validation, angles flag: %d" % (self.val_angs_ready)
            '''END DEBUG'''
            self.images,num_imgs_read = self._readimgs(self.val_imgs_file)
            # read training angles once
            if self.val_angs_ready == False:
                self.val_angs = self._readangs(self.val_ang_file)
                self.val_angs_ready = True
            ang_low_idx = self.val_angs_idx
            ang_up_idx = ang_low_idx + num_imgs_read
            ret_angles = self.val_angs[ang_low_idx:ang_up_idx]
            if len(ret_angles) == 0:
                ang_low_idx = 0
                self.val_angs_idx = num_imgs_read
                ang_up_idx = num_imgs_read
                ret_angles = self.val_angs[0:num_imgs_read]
            else: self.val_angs_idx += num_imgs_read
            # return batch of images and angles
            return [self.images, ret_angles]
    
    def _readimgs(self,f):
        try:
            imgs = f.read(self.batch_size * self.image_size)
        except IOError as exc:
            print "%s" % (exc)
            sys.exit(-1)
        num_imgs_read = int(len(imgs) / self.image_size)
        if num_imgs_read < self.batch_size or num_imgs_read == 0:
            f.seek(0)
            imgs = f.read(self.batch_size * self.image_size)
            num_imgs_read = int(len(imgs) / self.image_size)
        imgs = np.frombuffer(imgs, dtype=np.uint8)
        imgs = imgs.reshape(num_imgs_read, self.image_size)
        return [imgs, num_imgs_read] 

    def _readangs(self,f):
        try:
            angles = f.readlines()
        except IOError as exc:
            print "%s" % (exc)
            sys.exit(-1)
        angs = np.array([float(angles[0])], dtype=np.float32)
        for i in range(1, len(angles)):
            angs = np.append(angs, [float(angles[i])],0)
        # Mean normalization (-0.5 : 0.5)
        angs = (angs - angs.mean())/(angs.max() - angs.min())

        return angs

def show_video(f_imgs, angles_file, predict=None, t=60, frame_size=[HIGH, WIDTH, CHANN]):
    cv2.namedWindow('Center Camera', cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    rows, cols, chan = frame_size
    sz = rows*cols*chan
    #speed = 0 # DEBUG
    while True:
        try:
            img = f_imgs.read(sz)
            ang = angles_file.next()
            if predict != None: ang_pred = predict_file.next()
        except Exception as exc:
            print "%s"%(exc)
            break
        if len(img) == 0 or len(img) < sz: break
        #ang = ang.replace('\n','').split(',') #DEBUG
        #speed = float(ang[1]) #DEBUG
        #ang = float(ang[0]) #DEBUG
        if predict != None: ang_pred = float(ang_pred)
        ang = float(ang)
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape(rows, cols, chan)
        draw_steering_angle(img, ang)
        if predict != None: draw_steering_angle(img, ang_pred, color=[255, 0,0])
        #print speed # DEBUG
        cv2.imshow('Center Camera', img)
        cv2.waitKey(t)
    cv2.destroyAllWindows()
    
def draw_steering_angle(img, angle, frame_size=[HIGH, WIDTH, CHANN], color=[0, 255, 0]):
    rows, cols, chan = frame_size
    line_len = rows / 2.0
    P1 = (int(cols/2), rows)
    #P2 = (cols/2, rows/2)
    alfa = (pi/2.0) - angle
    P2c = cos(alfa)*line_len
    P2r = sin(alfa)*line_len
    P2 = (int(round((cols/2)-P2c)), int(round(rows - P2r)))
    #print  angle
    cv2.line(img, P1, P2, color, thickness=2)
    

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", help="images gzip file", required=True )
    parser.add_argument("--angles", help="angles gzip file", required=True)
    parser.add_argument("--predict", help="predicted angles gzip file", default=None)
    args = parser.parse_args()

    #print args # DEBUG

    try:
        imgs_file = gzip.open(args.imgs, 'rb') 
        angles_file = gzip.open(args.angles, 'rb')
        if args.predict != None: predict_file = gzip.open(args.predict, 'rb')
    except Exception as exc:
        print "%s" % (exc)
        imgs_file.close()
        angles_file.close()
        if args.predict != None: predict_file.close()
        sys.exit(2)
    if args.predict != None: show_video(imgs_file, angles_file, predict=predict_file)    
    else: show_video(imgs_file, angles_file)    
    imgs_file.close()
    angles_file.close()
    if args.predict != None: predict_file.close()


   
