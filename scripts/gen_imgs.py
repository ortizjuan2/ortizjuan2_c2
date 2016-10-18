#!/usr/bin/python
from __future__ import division
__version__ = '0.1'
__author__ = 'Juan C. Ortiz'

import rosbag
import sys
import numpy as np
import cv2
import gzip
import argparse
import time
import os

num_imgs_aprx = 0
MIN_SPEED_FILTER =  1.0

# Image size

INPUT_WIDTH = 640
INPUT_HIGH = 480
INPUT_CHANN = 3

OUTPUT_WIDTH = 160
OUTPUT_HIGH = 120
OUTPUT_CHANN = 3

def make_ste_dict(ste_data, secs='secs', nsecs='nsecs', angle='steering_wheel_angle', speed='speed'):
    ''' Make a dictionary of steering messages, frequency interpolated to 100Hz ''' 
    ste_dict = {}
    for msg in ste_data:
        if msg[1].speed > MIN_SPEED_FILTER:
            if msg[1].header.stamp.secs not in ste_dict:
                ste_dict[msg[1].header.stamp.secs] = [[msg[1].header.stamp.nsecs, \
                        msg[1].steering_wheel_angle, msg[1].speed]]
            else:
                ste_dict[msg[1].header.stamp.secs].append([msg[1].header.stamp.nsecs, \
                        msg[1].steering_wheel_angle, msg[1].speed])
    ''' Interpolate to 100Hz '''
    for key in ste_dict:
        new_data = []
        for i in range(len(ste_dict[key])-1):
            new_data.append(ste_dict[key][i])
            t = (ste_dict[key][i+1][0]-ste_dict[key][i][0])/2.0 + ste_dict[key][i][0]
            ang = (ste_dict[key][i+1][1]-ste_dict[key][i][1])/2.0 + ste_dict[key][i][1]
            spe = (ste_dict[key][i+1][2]-ste_dict[key][i][2])/2.0 + ste_dict[key][i][2]
            if spe > MIN_SPEED_FILTER:
                new_data.append([t, ang, spe])
        #new_data.append(ste_dict[key][i])
        ste_dict[key] = new_data
    return ste_dict


def gen_images(camera_data, ste_dict, compress, maxoutput, skip):
    ''' Generate images file and steering angle information, 
        two separated imcompressed files.
        Images are downsampled to OUTPUT_HIGHxOUTPUT_WIDTHxOUTPUT_CHANN.
    '''
    tname = time.strftime('%m%d%H%M%S')
    img_f  = './imgs_'+str(OUTPUT_HIGH)+'x'+str(OUTPUT_WIDTH)+'.' + tname + '.bin.gz'
    ang_f = './ang_data.' + tname + '.bin.gz'
    imgs_written = 0
    num_img = 0
    width = OUTPUT_WIDTH 
    height = OUTPUT_HIGH 
    if maxoutput < 0: maxoutput = 0
    if skip < 0: skip = 0
    global num_imgs_aprx
    try:
        img_file = gzip.open(img_f, 'wb')
        ang_file = gzip.open(ang_f, 'wb')
    except Exception as exc:
        print "%s" % (exc)
        return -1 
    for msg in camera_data:
        num_img += 1
        sys.stdout.write("%2d%%" % ((num_img/num_imgs_aprx)*100))
        sys.stdout.flush()
        sys.stdout.write("\b"* (3))  
        if msg[1].header.stamp.secs in ste_dict:
            steer = ste_dict[msg[1].header.stamp.secs]
            best = [999999999]
            msg_t = msg[1].header.stamp.nsecs
            for i in range(len(steer)):
                t = msg_t - steer[i][0]
                if (t > 0.0) and (t < best[0]):
                    best = [t, steer[i][0], steer[i][1], steer[i][2]] # nsecs, angle, speed
                elif t <= 0.0: break
            if best[0] != 999999999:
                ''' wirte image and angle to file'''
                if best[3] <= MIN_SPEED_FILTER:
                    continue
                imgs_written += 1
                if imgs_written <= skip and skip != 0:
                    continue
                #img = msg[1].data
                #img = np.array([[ord(x) for x in img]], dtype=np.uint8)
                img = np.frombuffer(msg[1].data, dtype=np.uint8)
                if compress == 'yes':
                    img = cv2.imdecode(img, cv2.IMREAD_REDUCED_COLOR_8 )
                else:
                    img = img.reshape(INPUT_HIGH, INPUT_WIDTH,INPUT_CHANN)  
                dst = cv2.resize(img,(width, height), 0, 0, cv2.INTER_LINEAR)
                try:
                    img_file.write(dst.data[0:width*height*3])
                    #ang_file.write(str(best[2])+','+str(best[3])+'\n') #DEBUG
                    ang_file.write(str(best[2])+'\n') #DEBUG
                except Exception as exc:
                    print "%s" % (exc)
                    print best
                    img_file.close()
                    ang_file.close()
                    return -1
                if imgs_written >= (maxoutput+skip) and maxoutput != 0:
                    break
    sys.stdout.write("100%\n")
    img_file.close()
    ang_file.close()
    return imgs_written-skip


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", help="bag with camera messages", required=True )
    parser.add_argument("--steering", help="bag with steering angle messages", required=True)
    parser.add_argument("--imcompressed", help="confirm if images are copressed or not", default="no")
    parser.add_argument("--maxoutput", help="maximum number of output images desired", type=int, default=0)
    parser.add_argument("--skip", help="number of initial images to skip, useful to generate different sets of images",
            type=int, default=0)
    args = parser.parse_args()

    cam_bagname = args.camera
    ste_bagname = args.steering
    try:
        sys.stdout.write("Reading camera bag file, please wait...")
        sys.stdout.flush()
        cambag = rosbag.Bag(cam_bagname)
        sys.stdout.write("Done.\n")
        sys.stdout.write("Reading steering angle bag file, please wait...")
        sys.stdout.flush()
        stebag = rosbag.Bag(ste_bagname)
        sys.stdout.write("Done.\n")
    except Exception as exc:
        print "%s" % (exc)
        sys.exit(2)
    # get steering report messages
    ste_raw = stebag.read_messages(topics=['/vehicle/steering_report'])
    sys.stdout.write("Reading angle data, please wait... ")
    ste = make_ste_dict(ste_raw)
    print "Completed."
    print "%d seconds readed"%(len(ste))
    num_imgs_aprx = len(ste) * 20 
    #camera_data = bag.read_messages(topics=['/center_camera/image_color',\
    #        '/left_camera/image_color', '/right_camera/image_color'])
    if args.imcompressed == 'yes':
        camera_data = cambag.read_messages(topics=['/center_camera/image_color/compressed'])
    else:
        camera_data = cambag.read_messages(topics=['/center_camera/image_color'])
    print 'Generating images and angles files, please wait...' 
    num_images = gen_images(camera_data, ste, args.imcompressed, args.maxoutput, args.skip)
    if num_images > 0:
        print "Completed, %d images written." %(num_images)
    else: print "Process aborted."
    cambag.close()
    stebag.close()

