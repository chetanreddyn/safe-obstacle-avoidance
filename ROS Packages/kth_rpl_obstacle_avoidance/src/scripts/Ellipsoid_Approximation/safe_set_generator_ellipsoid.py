#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from kth_rpl_obstacle_avoidance.msg import SafeSetEllipsoidsConfig 
import cv2
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt

class SafeSetEllipsoid:
    def __init__(self, robot_config,algo_config,render_config):
        
        ### ROS Stuff ###
        self.bridge = CvBridge() #it helps in converting ROS Image type to cv2 image type i.e np array
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw",Image, self.callback_depth)

        self.slice_1d = 0.6 #Calculated from the top of the image
        self.image_1d = None
        self.depth_1d = None
        self.rendered_data = None
        # self.save_depth_data_resized = False

        self.theta_min  = robot_config['theta_min']
        self.theta_max  = robot_config['theta_max']
        self.max_depth  = robot_config['max_depth']
        self.min_depth  = robot_config['min_depth']
        self.r_limit    = robot_config['r_limit']
        self.r0         = robot_config['r0']
        self.fov = self.theta_max - self.theta_min


        # Attributes used to store the depth data 
        self.depth_1d = None
        self.angle_1d = None
        self.X_1d     = None
        self.radial_depth_1d = None
        self.ellipses_data = None

        self.reference_direction = 0 # Default
    
        # Safe Set Generation Algorithm Parameters
        self.render_plt            = algo_config['render_plt']
        self.depth_val_outside_fov = algo_config['depth_val_outside_fov']
        self.r_max_ellipse         = algo_config['r_max_ellipse']
        self.d_s                   = algo_config['d_s']
        self.b_i                   = algo_config['b_i']
        self.num_ellipses          = algo_config['num_ellipses']

        # Rendering Properties
        self.y_limit               = render_config['y_limit'] # Indicates the maximum local region to be shown in the rendered image (= r_limit or r_max_ellipse)
        self.x_limit               = self.y_limit

        self.img_height            = render_config['img_height'] # For rendering

        self.scale                 = self.img_height/(2*self.y_limit)
        self.img_width             = int(2*self.scale*self.x_limit)
        self.ellipse_color         = render_config['ellipse_color']
        self.thickness             = render_config['thickness']
        self.theta_ellipse_resolution = render_config['theta_ellipse_resolution']


    def callback_depth(self,data):
        '''Receives the Depth Image from the Topic and Calls the Update Safe Set Function'''

        depth_img = self.bridge.imgmsg_to_cv2(data)
        # depth_img = data # COMMENT THIS LINE AND UNCOMMENT THE ABOVE LINE IN ROS
        
        fov = self.fov
        num_ellipses = self.num_ellipses
        num_ellipses_fov = int(num_ellipses*fov/(2*np.pi)) # number of ellipses inside the fov
        h,w = depth_img.shape

        aspect_ratio = w/h

        depth_img = cv2.resize(depth_img,(int(num_ellipses_fov),int(num_ellipses_fov/aspect_ratio)))

        h,w = depth_img.shape # height and width after downsizing

        mid_line = depth_img[int(self.slice_1d*h),:]
        
        self.depth_1d = np.nan_to_num(mid_line, nan=self.min_depth)
        self.angle_1d = np.linspace(self.theta_max,self.theta_min,self.depth_1d.shape[0]) #Discretised Angle
        self.X_1d = self.depth_1d*(1/np.tan(self.angle_1d))
        self.radial_depth_1d = (self.X_1d**2 + self.depth_1d**2)**0.5

        # self.image_1d = self.generate_img_1D(mid_line,min_val = self.min_depth,max_val=self.max_depth)


        self.update_safe_set(self.depth_1d)
        # print(len(self.ellipses_data))
        if self.render_plt and self.ellipses_data:
            # print('Rendered')
            self.render()

    ############# Safe Set Generation Functions ###########
    def update_reference_direction(self,reference_direction):
        '''Updates the Reference Direction whenever required'''

        self.reference_direction = reference_direction

    def get_ellipse_points(self,center,semi_major,semi_minor,theta_i,theta_ellipse_resolution = 30):
        '''
        Generates Ellipse Points (used in visualising) given the center, semi_major axis length, semi_minor axis length and orientation theta_i
        '''
        # Create an array of angles from 0 to 2*pi
        
        theta_ellipse = np.linspace(0, 2*np.pi, theta_ellipse_resolution)

        # Parametric equations for an ellipse
        x = center[0] + semi_major * np.cos(theta_ellipse)
        y = center[1] + semi_minor * np.sin(theta_ellipse)

        # Rotation matrix to rotate the ellipse
        rotation_matrix = np.array([[np.cos(theta_i), -np.sin(theta_i)],
                                    [np.sin(theta_i), np.cos(theta_i)]])

        # Apply rotation to the ellipse points
        ellipse_points_raw = np.dot(rotation_matrix, np.array([x - center[0], y - center[1]]))
        x_rotated, y_rotated = ellipse_points_raw[0] + center[0], ellipse_points_raw[1] + center[1]

        ellipse_points = np.array([x_rotated,y_rotated]) # Shape is (2,theta_ellipse_resolution)
        return ellipse_points


    def update_safe_set(self, radial_depth_1d):
        
        '''Updates the Best Local Safe Set as new depth data comes in'''

        theta_min             = self.theta_min
        theta_max             = self.theta_max
        fov                   = self.fov
        angle_1d              = self.angle_1d
        depth_val_outside_fov = self.depth_val_outside_fov
        r_max_ellipse         = self.r_max_ellipse
        d_s                   = self.d_s   # Buffer to the radial distance
        b_i                   = self.b_i # Semi Minor Axis Length


        resolution_outside_fov_1 = int((theta_min/fov)*angle_1d.shape[0]) # Number of Ellipses from theta_max to 360 degrees
        resolution_outside_fov_2 = int(((2*np.pi-theta_max)/fov)*angle_1d.shape[0]) # Number of Ellipses from 0 degrees to theta_min

        angle_1d_outside_fov_1 = np.linspace(0,theta_min,resolution_outside_fov_1) # 0 degrees to theta_min
        angle_1d_outside_fov_2 = np.linspace(theta_max,2*np.pi,resolution_outside_fov_2) # theta_max to 360 degrees

        # Constant Minimum radial depth for outside fov
        radial_depth_1d_outside_fov_1 = np.linspace(depth_val_outside_fov,depth_val_outside_fov,resolution_outside_fov_1) 
        radial_depth_1d_outside_fov_2 = np.linspace(depth_val_outside_fov,depth_val_outside_fov,resolution_outside_fov_2)

        # The complete 360 axis
        angle_1d_360 = np.hstack([angle_1d_outside_fov_1,angle_1d[::-1],angle_1d_outside_fov_2])
        radial_depth_1d_360 = np.hstack([radial_depth_1d_outside_fov_1,radial_depth_1d[::-1],radial_depth_1d_outside_fov_2])
 
        depth_data_processed = np.hstack([angle_1d_360.reshape(-1,1),radial_depth_1d_360.reshape(-1,1)]) #shape = (~num_ellipses,2)

        # depth_data_processed = dict(zip(angle_1d_360,radial_depth_1d_360))
        ellipses_data_plt = {}

         # Maximum depth range of the stereo/depth camera
        P = depth_data_processed.shape[0]
        ellipses_data_msg = np.zeros(shape=(P,5)) 
        # Assuming that the shape of data is (P,5) where row i is (cx_i,cy_i,a_i,b_i,theta_i) as defined in Formulation 3 (2)


        for i in range(depth_data_processed.shape[0]): # i represents the i th point and the i th ellipse

            theta_i,r_i = depth_data_processed[i][0],depth_data_processed[i][1]
            C = ((r_max_ellipse+r_i)/2)*np.array([[np.cos(theta_i)],[np.sin(theta_i)]]) # Center of the Ellipse shape = (2,1)
            a_i = (r_max_ellipse-r_i)/2 + d_s # Semi Major Axis Length Shape = scalar

            ellipse_points = self.get_ellipse_points(center=C,semi_major=a_i,semi_minor=b_i,theta_i=theta_i) # Shape = (2,resolution)
            ellipses_data_plt[theta_i] = ellipse_points
            
            ellipses_data_msg[i][0] = C[0][0] # x coordinate of center
            ellipses_data_msg[i][1] = C[1][0] # y coordinate of center
            ellipses_data_msg[i][2] = a_i # semi major axis length
            ellipses_data_msg[i][3] = b_i # semi minor axis length
            ellipses_data_msg[i][4] = theta_i # Orientation of the Ellipse

        self.ellipses_data_msg = ellipses_data_msg
        self.ellipses_data = ellipses_data_plt

    def show_images(self):
        # print('here2')
        if self.rendered_data is None:
            # print('here3')
            pass
        else:
            # img_resized = cv2.resize(self.image_1d,None,fx=0.15,fy=0.15)
            # cv2.imshow("1D depth",img_resized)
            
            if self.render_plt:
                depth_data_resized = cv2.resize(self.rendered_data,None,fx=1,fy=1)

                # if self.save_depth_data_resized:
                #     save_path = 'FirstFrame_Depth.npy'
                #     print('Saved RGB to ',save_path)
                #     np.save('PolarPlot.npy', depth_data_resized)
                #     np.save('1D_data.npy',self.depth_1d)
                    
                #     self.save_depth_data_resized = False


                
                cv2.imshow('Depth Safe Sets!',depth_data_resized)

            cv2.waitKey(1)


    ####### Plotting Helper Functions ########
    def render(self):

        # fig,ax = plt.subplots(figsize=(10,10))
        # for ellipse_points in self.ellipses_data.values():
        #     ax.plot(ellipse_points[0],ellipse_points[1],color='r')

        # fig.canvas.draw()

        # plt_data0 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # # print(data0.shape)
        # # print(fig.canvas.get_width_height())

        img_height = self.img_height
        img_width = self.img_width
        x_limit = self.x_limit
        y_limit = self.y_limit
        scale = self.scale


        ellipse_color = self.ellipse_color
        thickness = self.thickness
        isClosed = True

        canvas = np.ones(shape=(img_height,img_width,3))
        for ellipse_points in self.ellipses_data.values():
            X_pixel = ((np.array([[1,0],[0,-1]])@ellipse_points + np.array([[x_limit],[y_limit]]))*scale).astype(np.int32)
            cv2.polylines(canvas,[X_pixel.T.reshape(-1,1,2)],isClosed,ellipse_color,thickness)   
    
        self.rendered_data = canvas


def main():

    algo_config = {'render_plt':False,
                'depth_val_outside_fov':0.8,
                'r_max_ellipse':5,
                'd_s':0.2,
                'b_i':0.15,
                'num_ellipses':200
                }

    robot_config = {'theta_min':30*np.pi/180, # Minimum Angle of the Camera
                    'theta_max':150*np.pi/180, # Maximum Angle of the Camera
                    'max_depth':5, # Maximum Depth Perceived by the Camera (Used in the Loss Function)
                    'min_depth':0.4, # Minimum Depth Perceived by the Camera
                    'r_limit'  :2.5, # Maximum Radial Distance to Depict in the Plots
                    'r0'       :0.15, # Robot Radius 
                    'l'        :-0.08, #Distance of axle from Centre
                    # 'resolution': 1920
                    }
    
    # Rendering Properties
    render_config = {'y_limit': 6,
                    'img_height': 300,
                    'ellipse_color':(255,0,0),
                    'thickness': 1,
                    'theta_ellipse_resolution':50
                    }

    
    rospy.init_node('safe_set_generator', anonymous=True)
    pub = rospy.Publisher('/safe_set_config', SafeSetEllipsoidsConfig, queue_size=10)

    ss_obj = SafeSetEllipsoid(robot_config,algo_config,render_config) #safe_set object
    
    rate = rospy.Rate(1)
    # figsize = (3,3)
    # plt.figure(1,figsize=figsize)
    while not rospy.is_shutdown():
        ss_obj.show_images()

        ss_config = SafeSetEllipsoidsConfig()
        # print("Feasible Set Percent",ss_obj.feasible_safe_sets_percent)
        # print('here')
        if ss_obj.angle_1d is not None:

            ss_config.data = ss_obj.ellipses_data_msg.reshape(-1)
            ss_config.columns = 5
            # ss_config.t2 = ss_obj.best_safe_set_config['Shrunk_theta2']*180/np.pi
            # ss_config.depth = ss_obj.best_safe_set_config['Shrunk_depth']
            pub.publish(ss_config)
            current_time = rospy.Time.now()
            print("Published Safe Set | Time:", current_time.to_sec())
            # print(ss_config)

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()