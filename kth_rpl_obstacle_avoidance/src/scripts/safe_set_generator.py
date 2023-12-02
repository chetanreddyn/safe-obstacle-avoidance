#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt

class SafeSet:
    def __init__(self):
        self.bridge = CvBridge() #it helps in converting ROS Image type to cv2 image type i.e np array
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback_rgb)
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw",Image, self.callback_depth)

        self.rgb_img = None
        self.depth_img = None

        self.image_1d = None
        self.image_cost_1d = None
        self.image_weighed_cost_1d = None
        # Specifying the Minimum and Maximum Range 
        # of the Depth Camera
        self.min_cam_range = 0 
        self.max_cam_range = 5

        self.slice_1d = 0.6 #Calculated from the top of the image
        # Useful Data
        self.nan_percent_1d = None
        self.save_depth_image = False
        self.save_rgb_image = False

    def callback_rgb(self, data):
        self.rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if self.save_rgb_image:
            save_path = 'FirstFrame_RGB.npy'
            print('Saved RGB to ',save_path)
            np.save(save_path, self.rgb_img)
            self.save_rgb_image = False

    def callback_depth(self,data):
        depth_img_raw = self.bridge.imgmsg_to_cv2(data)
        if self.save_depth_image:
            save_path = 'FirstFrame_depth.npy'
            print('Saved depth to ',save_path)
            np.save(save_path, self.rgb_img)
            self.save_depth_image = False
        h,w = depth_img_raw.shape

        mid_line = depth_img_raw[int(self.slice_1d*h),:]
        nan_mask = np.isnan(mid_line)

        mid_line_processed = np.nan_to_num(mid_line,-1)

        self.image_1d = self.generate_img_1D(mid_line,min_val = self.min_cam_range,max_val=self.max_cam_range)

        costs_1d = self.cost_func(mid_line)       
        self.image_cost_1d = self.generate_img_1D(costs_1d,min_val=self.min_cam_range,max_val=self.max_cam_range)

        #weighed_depth_1d = mid_line*costs_1d
        #self.image_weighed_cost_1d = self.generate_img_1D(weighed_depth_1d,min_val=0,max_val=self.max_range)

        self.nan_percent_1d = 100*nan_mask.sum()/len(nan_mask)

        print("1D NAN values = {:.2f}%".format(self.nan_percent_1d))

    def cost_func(self,data_1d,w=1920):
        '''Ensure that the Maximum value is 1 to preserve scaling'''
        costs = [0 for i in range(w)]
        max_val = self.max_cam_range 
        #This should be the same as the argument passed to 
        # generate the costs image
        for x in range(w):
            costs[x] =  max_val - data_1d[x]*(4*x/w - 4*x*x/(w*w))
        #print(np.nanmax(costs[x]))
        #costs = np.vectorize(lambda x:4*x/w - 4*x*x/(w*w))(x_axis)
        return costs

    

    def generate_img_1D(self,data_1d,h=1080,w=1920,min_val=None,max_val=None):
        '''Takes 1D data and returns an image array for visualisation
            h: Height of Image in Pixels
            w: Width of Image in Pixels (expects the data_1d to have w number of entries)
            min_val: Minimum Value of the 1D array to be used in scaling
            max_val: Maximum Value of the 1D array to be used in scaling
            These value must be specified for consistent scaling
        '''

        if min_val == None:
            min_val = np.nanmin(data_1d)

        if max_val == None:
            max_val = np.nanmax(data_1d)
        
        img_arr = np.ones((h,w,3),dtype=np.uint8)*255

        for j in range(w):
            val = data_1d[j] #normalise it

            if np.isnan(val):
                # if it is a NAN value, Plotting it in Red
                # 10 here simply represents the number of pixels from the top to draw the point
                cv2.circle(img_arr,(j,10),1,color=(0,0,255),thickness=20)
            
            else:              
                # if it is a non-NAN value, normalising 
                # and depicting the depth on the image      
                val_normalised = h*(val-min_val)/(max_val-min_val)
                cv2.circle(img_arr,(j,h-int(val_normalised)),1,color=(255,0,0),thickness=20)

        return img_arr
    
    def show_images(self):
        if self.image_1d is None or self.image_cost_1d is None:
            pass
        else:
            
            #print('In Show Image')
            img_resized = cv2.resize(self.image_1d,None,fx=0.15,fy=0.15)
    
            cost_resized = cv2.resize(self.image_cost_1d,None,fx=0.15,fy=0.15)
            #weighed_depth_resized = cv2.resize(self.image_weighed_cost_1d,None,fx=0.15,fy=0.15)


            cv2.imshow("1D depth",img_resized)
            cv2.imshow("1D Cost Map",cost_resized)
            # plt.figure(1)
            
            # plt.imshow(img_resized)
            
            # plt.figure('cost_map')
            # plt.imshow(cost_resized)

            # plt.draw()
            # plt.pause(0.001)
            # cv2.imshow("1D Weighed Cost",weighed_depth_resized)
            cv2.waitKey(1)


def main():
    rospy.init_node('safe_set_generator', anonymous=True)
    ss_obj = SafeSet() #safe_set object
    
    rate = rospy.Rate(1000)
    # figsize = (3,3)
    # plt.figure(1,figsize=figsize)
    while not rospy.is_shutdown():
        ss_obj.show_images()
        #rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()