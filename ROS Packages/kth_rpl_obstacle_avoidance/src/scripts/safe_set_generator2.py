#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from kth_rpl_obstacle_avoidance.msg import SafeSetConfig 
import cv2
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString,Point,LinearRing
from shapely.ops import unary_union


class SafeSet:
    def __init__(self, robot_config,algo_config):
        
        ### ROS Stuff ###
        self.bridge = CvBridge() #it helps in converting ROS Image type to cv2 image type i.e np array
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw",Image, self.callback_depth)
        self.slice_1d = 0.6 #Calculated from the top of the image
        self.image_1d = None
        self.depth_1d = None
        self.rendered_data = None
        self.save_depth_data_resized = True

        self.theta_min  = robot_config['theta_min']
        self.theta_max  = robot_config['theta_max']
        self.max_depth  = robot_config['max_depth']
        self.min_depth  = robot_config['min_depth']
        self.r_limit    = robot_config['r_limit']
        self.r0         = robot_config['r0']


        self.step_index = 100
        # self.resolution= robot_config['resolution']

        self.fov = self.theta_max - self.theta_min

        # self.radial_distances = [self.min_depth for i in range(self.resolution)]
        # self.theta_axis = np.linspace(self.theta_max,self.theta_min,len(self.radial_distances),endpoint=False)

        self.radial_distances = None
        self.theta_axis = None
        self.reference_direction = 0 # Default
    

        self.best_safe_set_config = {}

        self.k_depth = algo_config['k_depth']
        self.k_angular = algo_config['k_angular']
        self.k_deviation = algo_config['k_deviation']
        self.metric_type = algo_config['metric_type']
        self.render_plt = algo_config['render_plt']
        self.min_safe_depth = algo_config['min_safe_depth']
        self.min_safe_angle = algo_config['min_safe_angle']
        self.feasible_safe_sets_percent = 0
        

    def callback_depth(self,data):
        '''Receives the Depth Image from the Topic and Calls the Update Safe Set Function'''

        depth_img_raw = self.bridge.imgmsg_to_cv2(data)
        h,_ = depth_img_raw.shape

        mid_line = depth_img_raw[int(self.slice_1d*h),:]
        self.depth_1d = mid_line
        mid_line_processed = np.nan_to_num(mid_line,nan=self.min_depth)

        self.image_1d = self.generate_img_1D(mid_line,min_val = self.min_depth,max_val=self.max_depth)


        self.update_safe_set(mid_line_processed)

        if self.feasible_safe_sets_percent>0 and self.render_plt:
            self.render()

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

    ############# Safe Set Generation Functions ###########
    def update_reference_direction(self,reference_direction):
        '''Updates the Reference Direction whenever required'''
        self.reference_direction = reference_direction

    def update_safe_set(self, radial_distances):
        '''Updates the Best Safe Set as new depth data comes in'''
        self.radial_distances = radial_distances
        self.theta_axis = np.linspace(self.theta_max,self.theta_min,len(self.radial_distances),endpoint=False)

        depth_points = np.hstack([self.theta_axis.reshape(-1,1),self.radial_distances.reshape(-1,1)])
        
        left_extreme_robot, right_extreme_robot = self.get_left_right_extremes()

        depth_points_cartesian = self.get_depth_points_cartesian(left_extreme_robot,right_extreme_robot)
        depth_line_s = LineString(depth_points_cartesian)

        c = 0 # Counts the Number of Feasible Safe Sets
        t = 0 # Counts the Total Number of Safe Sets
        best_safe_set_metric=0
        self.feasible_safe_sets_percent = 0
        for i in range(0,len(self.radial_distances),self.step_index):
            for j in range(i+1, len(self.radial_distances),self.step_index):
                
                t+=1
                r_min = depth_points[i:j,1].min()  # 1 represents index for radial distance!
                theta1 = depth_points[j,0]
                theta2 = depth_points[i,0]
                
                left_robot_point, right_robot_point = self.get_robot_points(theta1=theta1,theta2=theta2,rs=r_min)

                if left_robot_point[0]-right_robot_point[0]<np.pi: #Checking if the safe set is too narrow
                    continue

                left_depth_point = (theta2,r_min)
                right_depth_point = (theta1,r_min)
                safe_set_points = np.array([left_depth_point,left_robot_point,right_robot_point,right_depth_point,left_depth_point])

                safe_set_points_cartesian = self.get_safe_set_points_cartesian(safe_set_points)

                # Checking if the safe set intersects with the depth line
                safe_set_points_s = LinearRing(safe_set_points_cartesian)
                intersection_obj = depth_line_s.intersection(safe_set_points_s)

                if intersection_obj.is_empty:
                    
                    deviation_component = abs(np.pi/2+self.reference_direction-(theta1+theta2)/2)/self.fov
                    angular_component = (theta2-theta1)/self.fov

                    if self.metric_type == 'sum':
                        safe_set_metric = self.k_depth*r_min/self.max_depth + self.k_angular*angular_component + self.k_deviation*deviation_component
                    elif self.metric_type == 'product':
                        safe_set_metric = r_min*angular_component/deviation_component
                    else:
                        raise
                    phi = np.arccos(self.r0/r_min)

                    t1 = theta1+np.pi/2-phi
                    t2 = theta2-np.pi/2+phi
                    
                    if safe_set_metric>best_safe_set_metric and r_min>self.min_safe_depth and (t2-t1)>self.min_safe_angle:
                        c+=1
                        best_safe_set_metric=safe_set_metric
                        self.best_safe_set_config['i']=i
                        self.best_safe_set_config['j']=j
                        self.best_safe_set_config['theta1'] = theta1
                        self.best_safe_set_config['theta2'] = theta2
                        self.best_safe_set_config['r_min'] = r_min
                        self.best_safe_set_config['Metric'] = safe_set_metric
                        self.best_safe_set_config['left_robot_point'] = left_robot_point
                        self.best_safe_set_config['right_robot_point'] = right_robot_point
                        self.best_safe_set_config['left_depth_point'] = left_depth_point
                        self.best_safe_set_config['right_depth_point'] = right_depth_point



                        self.best_safe_set_config['Shrunk_theta1'] = t1
                        self.best_safe_set_config['Shrunk_theta2'] = t2
                        self.best_safe_set_config['Shrunk_depth'] = r_min-self.r0
                
        self.feasible_safe_sets_percent = (c/t)*100

    ########## HELPER FUNCTIONS FOR update_safe_set ##########
    def get_left_right_extremes(self):
        '''
        Returns the left and right tangent points in Polar Coordinates on the robot for theta_min and theta_max
        '''

        right_extreme_depth = (self.theta_min,self.radial_distances[-1])
        _,right_extreme_robot = self.get_tangent_coordinates(right_extreme_depth[0],right_extreme_depth[1],self.r0)

        left_extreme_depth = (self.theta_max,self.radial_distances[0]) #Left Extreme point of the Depth Line (Blue) in Polar Coordinates
        left_extreme_robot,_ = self.get_tangent_coordinates(left_extreme_depth[0],left_extreme_depth[1],self.r0)

        return left_extreme_robot,right_extreme_robot  
    

    def get_cartesian(self,polar_coords):
        '''
        polar_coords: A tuple
        '''
        theta,r = polar_coords
        return r*np.cos(theta),r*np.sin(theta)
    
    def get_safe_set_points_cartesian(self,safe_set_points,shrinking_factor=0.99):
        '''
        Used to form the linear ring to check intersection with depth line
        '''
        safe_set_points_cartesian = np.hstack([shrinking_factor*safe_set_points[:,1:2]*np.cos(safe_set_points[:,0:1]),shrinking_factor*safe_set_points[:,1:2]*np.sin(safe_set_points[:,0:1])])
        return safe_set_points_cartesian
    
    def get_depth_points_cartesian(self,left_extreme_robot,right_extreme_robot):
        '''
        Used to build the depth_line
        '''
        x_axis_points = (self.radial_distances*np.cos(self.theta_axis)).reshape(-1,1)
        y_axis_points = (self.radial_distances*np.sin(self.theta_axis)).reshape(-1,1)

        left_extreme_robot_cartesian = self.get_cartesian(left_extreme_robot)
        right_extreme_robot_cartesian = self.get_cartesian(right_extreme_robot)

        x_axis_points[0,0] = left_extreme_robot_cartesian[0]
        y_axis_points[0,0] = left_extreme_robot_cartesian[1]

        x_axis_points[-1,0] = right_extreme_robot_cartesian[0]
        y_axis_points[-1,0] = right_extreme_robot_cartesian[1]

        return np.hstack([x_axis_points,y_axis_points]) 

    def get_tangent_coordinates(self,theta,d,r0):
        '''
        d,theta: Polar Coordinates of the point
        r0: Radius of the Robot

        Returns the 2 Points of Intersections of the tangents from a point to a circle
        '''
        if d<=r0:
            print("Depth 1d",self.radial_distances[-10:])
            raise
        phi = np.arccos(r0/d)
        

        assert 0<phi<np.pi/2,"Phi: ,r0: ,d: ".format(phi,r0,d)

        left_point = (theta+phi,r0)
        right_point = (theta-phi,r0)
        points = [left_point,right_point]
        return points

    def get_robot_points(self,theta1,theta2,rs):
        '''
        theta1: Angular Distance of Right Point, lower angle wrt x-axis
        theta2: Angular Distance of Left Point, higher angle wrt x-axis
        rs: Min Radial Distance in (theta1,theta2)
        Returns the two points on the Robot
        '''
            
        _,right_robot_point = self.get_tangent_coordinates(theta1,rs,self.r0)
        left_robot_point,_ = self.get_tangent_coordinates(theta2,rs,self.r0)

        return left_robot_point,right_robot_point 

    ############# FUNCTIONS USED FOR VISUALISATION ##########
    # def render(self):
    #     '''
    #     Returns the ax object populated with depth_line and end tangents
    #     '''
    #     desired_dpi = 80  # Adjust the DPI to your desired resolution
    #     fig = plt.figure(dpi=desired_dpi)
    #     # fig = plt.figure(figsize=(10, 10))
    #     ax = plt.subplot(polar=True)

    #     # Plots the Depth Line
    #     ax.plot(self.theta_axis,self.radial_distances)

    #     # Plot radial lines
    #     # ax.plot([theta_min, theta_min], [0, 2.5], color='r', linestyle='-', linewidth=2, label='Theta 1')
    #     # ax.plot([theta_max, theta_max], [0, 2.5], color='r', linestyle='-', linewidth=2, label='Theta 2')

    #     # Plots the Robot as a Filled Circle
    #     theta_robot_fill = np.linspace(0, 2 * np.pi, 100)  # Create 100 points around the circumference
    #     r = np.full_like(theta_robot_fill, self.r0)  # Set the radius for all points
    #     # print(r)
    #     # Plot the circle
    #     ax.fill(theta_robot_fill, r, color='green')

    #     # Plotting Extreme Left and Right Tangents
    #     #Left Extreme Tangent
    #     left_extreme_depth = (self.theta_max,self.radial_distances[0]) #Left Extreme point of the Depth Line (Blue) in Polar Coordinates
    #     left_extreme_robot,_ = self.get_tangent_coordinates(left_extreme_depth[0],left_extreme_depth[1],self.r0)
    #     ax.plot([left_extreme_depth[0],left_extreme_robot[0]],[left_extreme_depth[1],left_extreme_robot[1]],color='r')

    #     #Right Extreme Tangent
    #     right_extreme_depth = (self.theta_min,self.radial_distances[-1])
    #     _,right_extreme_robot = self.get_tangent_coordinates(right_extreme_depth[0],right_extreme_depth[1],self.r0)
    #     ax.plot([right_extreme_depth[0],right_extreme_robot[0]],[right_extreme_depth[1],right_extreme_robot[1]],color='r')
    #     # ax.set_rticks([0.1,1])  # Adjust the tick values as needed


    #     #Setting Limits
    #     # ax.set_thetamin((theta_min-np.pi/2)*180/np.pi)
    #     # ax.set_thetamax((theta_max+np.pi/2)*180/np.pi)

        
    #     left_robot_point = self.best_safe_set_config['left_robot_point']
    #     right_robot_point = self.best_safe_set_config['right_robot_point']
    #     left_depth_point = self.best_safe_set_config['left_depth_point']
    #     right_depth_point = self.best_safe_set_config['right_depth_point']

    #     safe_set_points = np.array([left_depth_point,left_robot_point,right_robot_point,right_depth_point,left_depth_point])
    #     ax.scatter(safe_set_points[:,0],safe_set_points[:,1],c='b',linewidths=1)
    #     ax.plot(safe_set_points[:,0],safe_set_points[:,1],c='b')

    #     ax.set_thetamin(0)
    #     ax.set_thetamax(180)
    #     ax.set_aspect('equal')
    #     ax.set_rmax(self.r_limit)

    #     fig.canvas.draw()

    #     data0 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     # print(data0.shape)
    #     # print(fig.canvas.get_width_height())
    #     data = data0.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     self.rendered_data = data
    def render(self,only_depth_line=False):
        '''
        Returns the ax object populated with depth_line and end tangents
        '''
        desired_dpi = 80  # Adjust the DPI to your desired resolution
        fig = plt.figure(dpi=desired_dpi)
        # fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(polar=True)
        # Plots the Depth Line
        ax.plot(self.theta_axis,self.radial_distances)

        # Plot radial lines
        # ax.plot([theta_min, theta_min], [0, 2.5], color='r', linestyle='-', linewidth=2, label='Theta 1')
        # ax.plot([theta_max, theta_max], [0, 2.5], color='r', linestyle='-', linewidth=2, label='Theta 2')

        # Plots the Robot as a Filled Circle
        theta_robot_fill = np.linspace(0, 2 * np.pi, 100)  # Create 100 points around the circumference
        r = np.full_like(theta_robot_fill, self.r0)  # Set the radius for all points
        # print(r)

        # Plot the circle
        ax.fill(theta_robot_fill, r, color='green')

        # Plotting Extreme Left and Right Tangents
        #Left Extreme Tangent
        left_extreme_depth = (self.theta_max,self.radial_distances[0]) #Left Extreme point of the Depth Line (Blue) in Polar Coordinates
        left_extreme_robot,_ = self.get_tangent_coordinates(left_extreme_depth[0],left_extreme_depth[1],self.r0)
        ax.plot([left_extreme_depth[0],left_extreme_robot[0]],[left_extreme_depth[1],left_extreme_robot[1]],color='r')

        #Right Extreme Tangent
        right_extreme_depth = (self.theta_min,self.radial_distances[-1])
        _,right_extreme_robot = self.get_tangent_coordinates(right_extreme_depth[0],right_extreme_depth[1],self.r0)
        ax.plot([right_extreme_depth[0],right_extreme_robot[0]],[right_extreme_depth[1],right_extreme_robot[1]],color='r')

        #Setting Limits
        # ax.set_thetamin((theta_min-np.pi/2)*180/np.pi)
        # ax.set_thetamax((theta_max+np.pi/2)*180/np.pi)
        ax.set_thetamin(0)
        ax.set_thetamax(360)
        ax.set_rmax(self.r_limit)

        # if only_depth_line:
        #     pass 
        # else:
        left_robot_point = self.best_safe_set_config['left_robot_point']
        right_robot_point = self.best_safe_set_config['right_robot_point']
        left_depth_point = self.best_safe_set_config['left_depth_point']
        right_depth_point = self.best_safe_set_config['right_depth_point']

        safe_set_points = np.array([left_depth_point,left_robot_point,right_robot_point,right_depth_point,left_depth_point])
        ax.scatter(safe_set_points[:,0],safe_set_points[:,1],c='g',linewidths=1)
        ax.plot(safe_set_points[:,0],safe_set_points[:,1],c='g')


        # Shrinking the Safe Set by Robot Radius
        d = self.best_safe_set_config['r_min']
        theta1 = self.best_safe_set_config['theta1']
        theta2 = self.best_safe_set_config['theta2']
        r0 = self.r0
        phi = np.arccos(r0/d)

        t1 = theta1+np.pi/2-phi
        t2 = theta2-np.pi/2+phi
        ax.plot([0,t1,t2,0],[0,d-r0,d-r0,0])
        

        fig.canvas.draw()

        data0 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # print(data0.shape)
        # print(fig.canvas.get_width_height())
        data = data0.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.rendered_data = data
    def show_images(self):
        if self.image_1d is None or self.rendered_data is None:
            pass
        else:
            img_resized = cv2.resize(self.image_1d,None,fx=0.15,fy=0.15)
            cv2.imshow("1D depth",img_resized)
            
            if self.render_plt:
                depth_data_resized = cv2.resize(self.rendered_data,None,fx=1,fy=1)

                if self.save_depth_data_resized:
                    save_path = 'FirstFrame_Depth.npy'
                    print('Saved RGB to ',save_path)
                    np.save('PolarPlot.npy', depth_data_resized)
                    np.save('1D_data.npy',self.depth_1d)
                    
                    self.save_depth_data_resized = False


                
                cv2.imshow('Depth Safe Sets!',depth_data_resized)
            
            print("Theta1: {:.2f}, Theta2: {:.2f}, Depth of Safe Set: {:.2f}, Metric: {:.2f}".format(self.best_safe_set_config['theta1']*180/np.pi,
                                                                                        self.best_safe_set_config['theta2']*180/np.pi,
                                                                                        self.best_safe_set_config['r_min'],
                                                                                        self.best_safe_set_config['Metric']))
            cv2.waitKey(1)


def main():

    robot_config = {'theta_min':2*np.pi/6, # Minimum Angle of the Camera
                    'theta_max':4*np.pi/6, # Maximum Angle of the Camera
                    'max_depth':2.5, # Maximum Depth Perceived by the Camera (Used in the Loss Function)
                    'min_depth':0.3, # Minimum Depth Perceived by the Camera
                    'r_limit'  :2.5, # Maximum Radial Distance to Depict in the Plots
                    'r0'       :0.15, # Robot Radius 
                    # 'resolution': 1920
                    }
    algo_config = {'k_depth':2,
                   'k_angular':1,
                   'k_deviation':0,
                   'metric_type':'sum',
                   'render_plt':True,
                   'min_safe_depth':0.1,
                   'min_safe_angle':5*np.pi/180}
    
    rospy.init_node('safe_set_generator', anonymous=True)
    pub = rospy.Publisher('/safe_set_config', SafeSetConfig, queue_size=10)

    ss_obj = SafeSet(robot_config,algo_config) #safe_set object
    
    rate = rospy.Rate(2)
    # figsize = (3,3)
    # plt.figure(1,figsize=figsize)
    while not rospy.is_shutdown():
        ss_obj.show_images()
        ss_config = SafeSetConfig()
        print("Feasible Set Percent",ss_obj.feasible_safe_sets_percent)
        if ss_obj.feasible_safe_sets_percent > 0:

            ss_config.t1 = ss_obj.best_safe_set_config['Shrunk_theta1']*180/np.pi
            ss_config.t2 = ss_obj.best_safe_set_config['Shrunk_theta2']*180/np.pi
            ss_config.depth = ss_obj.best_safe_set_config['Shrunk_depth']
            pub.publish(ss_config)
            current_time = rospy.Time.now()
            print(current_time)
            print(ss_config)

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()