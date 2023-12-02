#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge() #it helps in converting ROS Image type to cv2 image type i.e np array
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback_rgb)
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw",Image, self.callback_depth)
        self.save_images = True

        self.slice_1d = 0.6
        self.rgb_img = None
        self.depth_img = None

    def callback_rgb(self, data):
        self.rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def get_sliced_img(self,img,row_frac):
        slice_thickness = 0.01

        h,_,_ = img.shape
        row = int(row_frac*h)

        img_sliced = img.copy()
        img_sliced[row-int(slice_thickness*h):row+int(slice_thickness*h),:] = [255,0,0]
        return img_sliced

    def callback_depth(self,data):

        depth_img_raw = self.bridge.imgmsg_to_cv2(data)
        nan_mask = (np.isnan(depth_img_raw)) #nan_mask has true for all nan values


        depth_img_gray = depth_img_raw*(~nan_mask) #nan values are reduced to zero instead of nan
        depth_img_gray_norm = cv2.normalize(depth_img_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        depth_img_rgb = cv2.cvtColor(depth_img_gray_norm,cv2.COLOR_GRAY2RGB)

        nan_mask_rgb = np.zeros_like(depth_img_rgb,dtype=np.uint8)

        nan_mask_rgb[:,:,2] = nan_mask.astype(np.uint8)*255
        depth_img_final = cv2.bitwise_or(nan_mask_rgb,depth_img_rgb)
        self.depth_img = depth_img_final
        
        self.depth_img_raw = depth_img_raw
        print("2D Image NAN values: {:.1f}%".format(nan_mask.astype(np.uint8).sum()*100/depth_img_raw.size))


    def show_images(self):
        if self.depth_img is None or self.rgb_img is None:
            pass
        else:
            
            #print('In Show Image')
            rgb_resized = cv2.resize(self.rgb_img,None,fx=0.15,fy=0.15)

            

            depth_resized = cv2.resize(self.depth_img,None,fx=0.15,fy=0.15)
            depth_resized_slice = self.get_sliced_img(depth_resized,self.slice_1d)

            if self.save_images:

                np.save('FirstFrame_RGB.npy', self.rgb_img)
                np.save('FirstFrame_Depth.npy',self.depth_img_raw)
                
                self.save_images = False

            both_images = np.vstack([rgb_resized,depth_resized_slice])
            cv2.imshow('RGB and Depth Images',both_images)
            # cv2.imshow("RGB Image",rgb_rbesized)
            # cv2.imshow("Depth Image",depth_resized)
            cv2.waitKey(1)
        
def main():
    rospy.init_node('visualiser', anonymous=True)
    ic = ImageSubscriber()

    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        i = ic.show_images()
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




