#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from kth_rpl_obstacle_avoidance.msg import SafeSetConfig 
import tf.transformations
import numpy as np
from cvxopt import matrix, solvers
import cv2


 

def publish_velocity_commands():
    rospy.init_node('velocity_publisher', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    rate = rospy.Rate(100)  # 10 Hz (adjust the frequency as needed)

    while not rospy.is_shutdown():
        cmd = Twist()
        cmd.linear.x = 0.1  # Linear velocity (m/s)
        cmd.angular.z = 0  # Angular velocity (radians/s)
        pub.publish(cmd)
    

        rate.sleep()

class Controller:
    def __init__(self,robot_config, algo_config,reference_angle):
        rospy.Subscriber('/odom',Odometry,self.odom_callback)
        rospy.Subscriber('/safe_set_config',SafeSetConfig,self.update_safeset_callback)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.theta_min  = robot_config['theta_min']
        self.theta_max  = robot_config['theta_max']
        self.max_depth  = robot_config['max_depth']
        self.min_depth  = robot_config['min_depth']
        self.r_limit    = robot_config['r_limit']
        self.r0         = robot_config['r0']
        self.l          = robot_config['l']

        # Algo Related
        self.gamma      = algo_config['gamma']
        self.Q          = np.array([[algo_config['Q_u1'],0],[0,1]]).astype(np.float64)
        self.u1_limit   = algo_config['u1_limit']  # Upper Bound of Linear Velocity
        self.u2_limit   = algo_config['u2_limit']  # Upper Bound for the Angular Velocity
        self.k          = algo_config['k']  # Reference Angular Velocity = -k*theta
        self.u1_ref     = algo_config['u1_ref']  # Forward Velocity Reference
        self.img_width  = algo_config['img_width']
        self.img_height = algo_config['img_height']
        self.y_limit    = algo_config['y_limit']
        self.buffer_radius = algo_config['buffer_radius']

        self.reference_angle_global = reference_angle # in global coordinate system



        self.scale = self.img_height/self.y_limit
        self.x_limit = (self.img_width/self.img_height)*self.y_limit

        self.robot_state   = None
        self.safeset    = {'t1':None,'t2':None,'depth':None,'origin':None}

    def get_X_pixel(self,x_local,y_local):
        # Converts from continuous coordinate system to pixel indices in the np image

        y_limit = self.y_limit

        scale = self.scale
        x_limit = self.x_limit

        y_shift = y_limit
        x_shift = -x_limit/2
        X_pixel = np.array([[0,-1],[1,0]])@np.array([[x_local-x_shift],[y_local-y_shift]])*scale
        x_pixel,y_pixel = int(X_pixel[0,0]),int(X_pixel[1,0])
        return y_pixel,x_pixel

    def odom_callback(self,data):
        pose = data.pose.pose
        linear_velocity = data.twist.twist.linear
        angular_velocity = data.twist.twist.angular

        quaternion = [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
        # Convert the quaternion to a rotation matrix
        rotation_matrix = tf.transformations.quaternion_matrix(quaternion)

        # Extract the yaw angle (rotation about Z-axis)
        assert self.l<0 # Remove the 180 degree
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])+np.pi

        if yaw>np.pi:
            yaw = yaw-np.pi*2

        self.robot_state = [pose.position.x,pose.position.y,yaw,linear_velocity.x,linear_velocity.z]
        current_time = rospy.Time.now()
        # print("Time: {}, (x, y, theta): {:.2f}, {:.2f}, {:.2f}, (u,w): {:.2f},{:.2f}".format(str(current_time)[:4],pose.position.x, pose.position.y, yaw,linear_velocity.x,angular_velocity.z))

    def update_safeset_callback(self,data):
        # print('inside update safeset')
        self.safeset['t1']      = data.t1*np.pi/180
        self.safeset['t2']      = data.t2*np.pi/180
        self.safeset['depth']   = data.depth

        self.safeset['origin'] = self.robot_state # The safeset origin in global coordinate system


    ######## HELPER FUNCTIONS TO SYNTHESIS THE CONTROL ##########
    def get_Xp(self,Xc,l):
        xp = Xc[0]+l*np.cos(Xc[2])
        yp = Xc[1]+l*np.sin(Xc[2])
        thetap = Xc[2]

        Xp = np.array([xp,yp,thetap])
        return Xp

    def h_val(self,theta_set_i,X,l):
        '''
        Helper Function to write the functions as given in FORMULATION 2 (2) in NOTES
        '''
        x = X[0][0]
        y = X[1][0]
        theta = X[2][0]
        return -np.tan(theta_set_i)*x + y - l*np.sin(theta) + l*np.tan(theta_set_i)*np.cos(theta)


    def get_G_h_opt(self,X,U_ref):
        # G and h are the inequalities for the CBF in the optimisation problem
    
        gamma = self.gamma
        l = self.l
        t1 = self.safeset['t1']
        t2 = self.safeset['t2']
        depth = self.safeset['depth']

        u1_limit = self.u1_limit
        u2_limit = self.u2_limit
        theta = X[2][0] #extracting theta for easier usage in later formulae

        #if either t1 or t2 is a 90 degree angle
        if t1==np.pi/2:
            t1 = t1 - 0.1*np.pi/180 #Nudging t1 by 0.1 degree if it is 90 degrees
        if t2==np.pi/2:
            t2 = t2 + 0.1*np.pi/180

        # Finding a1,a2,a3 depending on theta and safe set config (NOTE: these a_i's are different from that implemented below, look at written notes)
        if 0<=t1<np.pi/2:
            a1 = np.array([-np.tan(t1),1,-(l*np.cos(theta)+l*np.sin(theta)*np.tan(t1))])
            h1 = self.h_val(t1,X,l)

        elif np.pi/2<t1<=np.pi:
            a1 = -np.array([-np.tan(t1),1,-(l*np.cos(theta)+l*np.sin(theta)*np.tan(t1))])
            h1 = -self.h_val(t1,X,l)

        if 0<=t2<np.pi/2:
            a2 = np.array([np.tan(t2),-1,l*np.cos(theta)+l*np.sin(theta)*np.tan(t2)])
            h2 = -self.h_val(t2,X,l)

        elif np.pi/2<t2<=np.pi:
            a2 = -np.array([np.tan(t2),-1,l*np.cos(theta)+l*np.sin(theta)*np.tan(t2)])
            h2 = self.h_val(t2,X,l)


        a3 = np.array([0,-1,l*np.cos(theta)])
        h3 = -X[1][0]+l*np.sin(theta)+depth

        # From system dynamics: x_dot = Ax + Bu
        B = np.array([[np.cos(theta),0],[np.sin(theta),0],[0,1]])

        G_opt = -np.vstack([a1@B,a2@B,a3@B,np.array([-1,0]),np.array([0,-1])])
        h_opt = np.vstack([gamma*h1+a1@B@U_ref,gamma*h2+a2@B@U_ref,gamma*h3+a3@B@U_ref,np.array([u1_limit-U_ref[0][0]]),np.array([u2_limit-U_ref[1][0]])])

        assert h1>=0, 'h1 = {}'.format(h1)
        assert h2>=0, 'h2 = {}'.format(h2)
        assert h3>=0, 'h3 = {}, h3 = {:.2f} + {:.2f} + {:.2f}'.format(h3,-X[1][0],l*np.sin(theta),depth)
        
        info = {'a1':a1,'a2':a2,'a3':a3,'G_opt':G_opt,'h_opt':h_opt,'B_model':B}

        return G_opt,h_opt,info

    def get_U(self,X,U_ref):
        
        # U is synthesised by solving the optimisation problem
        
        # l = self.l
        # t1 = self.safeset['t1']
        # t2 = self.safeset['t2']
        # depth = self.safeset['depth']
        # gamma = self.gamma
        # Q = self.Q

        G_opt, h_opt,info = self.get_G_h_opt(X,U_ref)

        Q = matrix(self.Q)
        p = matrix(np.zeros(shape=(2,1)))
        G = matrix(G_opt)
        h = matrix(h_opt)

        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G, h)

        U_dash = np.array(sol['x']).reshape(2,1)
        U = U_dash+U_ref


        return U,info

    def scale_angle(self,angle,info):
        if -2*np.pi<=angle<=-np.pi:
            angle = angle+2*np.pi

        if np.pi<angle<=2*np.pi:
            angle = angle-2*np.pi

        if angle>2*np.pi or angle<-2*np.pi:
            raise ValueError("angle is outside [-2*pi,2*pi]: {}".format(angle),info)

        return angle

    def publish_vel(self):

        x0 = self.safeset['origin'][0]
        y0 = self.safeset['origin'][1]
        yaw0 = self.safeset['origin'][2]

        coordinate_axis_rotation = yaw0-np.pi/2
        transformation_matrix = np.array([[np.cos(coordinate_axis_rotation),np.sin(coordinate_axis_rotation)],
                                          [-np.sin(coordinate_axis_rotation),np.cos(coordinate_axis_rotation)]])

        x_global = self.robot_state[0]
        y_global = self.robot_state[1]
        yaw_global = self.robot_state[2]

        X_local = transformation_matrix@np.array([[x_global-x0],[y_global-y0]])

        #Adding Buffer distance
        t_avg = (self.safeset['t1']+self.safeset['t2'])/2
        # print(t_avg*180/np.pi)
        buffer_radius = self.buffer_radius
        # print(t_avg*180/np.pi,X_local)
        X_local = X_local + buffer_radius*np.array([[np.cos(t_avg)],[np.sin(t_avg)]])
        # print('after',X_local)

        yaw_local = yaw_global - self.scale_angle(coordinate_axis_rotation,info={'coordinate_axis_rotation':coordinate_axis_rotation})
        yaw_local = self.scale_angle(yaw_local,info={'yaw_local':yaw_local,'yaw_global':yaw_global})

        # Used to draw arrows on the safe set
        self.x_local = X_local[0,0]
        self.y_local = X_local[1,0]
        self.yaw_local = yaw_local
        self.reference_angle_local = self.reference_angle_global-coordinate_axis_rotation

        X = np.vstack([X_local,yaw_local])
        
        
        theta_deviation = self.scale_angle(yaw_global - self.reference_angle_global,info=['theta_deviation',yaw_global]) #Calculate theta_deviation in safeset space

        # print("Local=({:.2f},{:.2f},{:.2f}), Global=({:.2f},{:.2f},{:.2f}), Del Theta={:.2f}".format(X[0,0],X[1,0],X[2,0]*180/np.pi,
        #                                                                                              x_global,y_global,yaw_global*180/np.pi,
        #                                                                                              theta_deviation*180/np.pi))

        # u1_ref = self.u1_ref
        # u2_ref = -self.k*(theta_deviation)
        # U_ref = np.array([u1_ref,u2_ref]).reshape(2,1)
        # U,info = self.get_U(X,U_ref)
        # print("Local=({:.2f},{:.2f},{:.2f}), Global=({:.2f},{:.2f},{:.2f}),U=({:.2f},{:.2f})".format(X[0,0],X[1,0],X[2,0]*180/np.pi,
        #                                                                                              x_global,y_global,yaw_global*180/np.pi,
        #                                                                                              U[0,0],U[1,0]))
        
        # cmd = Twist()
        # cmd.linear.x = -U[0,0]  # Linear velocity (m/s)
        # cmd.angular.z = U[1,0]  # Angular velocity (radians/s)
        # self.pub.publish(cmd)

    def get_points(self,theta,depth):
        other_point = (depth/np.tan(theta),depth)
        return [(0,0),other_point]
    def render_safe_set(self):

        img_height = self.img_height
        img_width = self.img_width
        t1 = self.safeset['t1']
        t2 = self.safeset['t2']
        depth = self.safeset['depth']

        canvas = np.ones(shape=(img_height,img_width,3))
        # Define the color (BGR format, (0, 0, 255) is red)
        arrow_len = 0.25
        ref_arrow_len = 0.3
        color = (0, 10, 0)
        robot_arrow_color = (0,0,255)
        reference_arrow_color = (255,0,0)
        thickness = int(img_width/150)

        # Define the radius of the circle (in pixels)
        radius = 2

        right_line = self.get_points(t1,depth)
        left_line = self.get_points(t2,depth)

        cv2.line(canvas, self.get_X_pixel(*right_line[0]), self.get_X_pixel(*right_line[1]), color, thickness)
        cv2.line(canvas, self.get_X_pixel(*left_line[0]), self.get_X_pixel(*left_line[1]), color, thickness)
        cv2.line(canvas, self.get_X_pixel(*left_line[1]), self.get_X_pixel(*right_line[1]), color, thickness)

        cv2.circle(canvas, self.get_X_pixel(self.x_local,self.y_local),radius, color=(0,0,255), thickness=2)

        x2 = self.x_local+arrow_len*np.cos(self.yaw_local)
        y2 = self.y_local+arrow_len*np.sin(self.yaw_local)

        cv2.arrowedLine(canvas,self.get_X_pixel(self.x_local,self.y_local),self.get_X_pixel(x2,y2),color = robot_arrow_color,thickness=int(thickness))

        x2_ref = self.x_local+ref_arrow_len*np.cos(self.reference_angle_local)
        y2_ref = self.y_local+ref_arrow_len*np.sin(self.reference_angle_local)
        cv2.arrowedLine(canvas,self.get_X_pixel(self.x_local,self.y_local),self.get_X_pixel(x2_ref,y2_ref),color = reference_arrow_color,thickness=int(thickness))


        self.rendered_safe_set = canvas

        cv2.imshow('Safe Set',canvas)
        # cv2.imshow("RGB Image",rgb_rbesized)
        # cv2.imshow("Depth Image",depth_resized)
        cv2.waitKey(1)

def main():
    rospy.init_node("controller")

    robot_config = {'theta_min':2*np.pi/6, # Minimum Angle of the Camera
                'theta_max':4*np.pi/6, # Maximum Angle of the Camera
                'max_depth':2.5, # Maximum Depth Perceived by the Camera (Used in the Loss Function)
                'min_depth':0.3, # Minimum Depth Perceived by the Camera
                'r_limit'  :2.5, # Maximum Radial Distance to Depict in the Plots
                'r0'       :0.15, # Robot Radius 
                'l'        :-0.08
                # 'resolution': 1920
                }

    algo_config = {'gamma':1, # High Gamma means that the safety filter acts only near the boundary
                   'Q_u1':100,
                   'u1_limit':0.26,
                   'u2_limit':1.82,
                   'k':0.1,
                   'u1_ref':0.1,
                   'img_width':320,
                   'img_height':240,
                   'y_limit':2.5,
                   'buffer_radius':0.3}


    render_safe_set = True
    reference_angle = 0*np.pi/180

    controller_obj = Controller(robot_config=robot_config,algo_config=algo_config,reference_angle=reference_angle)

    rate = rospy.Rate(1000)

    while not rospy.is_shutdown():
        # print('here')
        current_time = rospy.get_rostime()

        # Print the current ROS time
        # print("Current ROS Time: %f seconds" % current_time.to_sec())

        # # controller_obj.vel_publish()
        if controller_obj.safeset['t1']!=None:
            print("Safe Set Configuration Received!")
            # print(round(controller_obj.safeset['t1']))
            controller_obj.publish_vel()
            controller_obj.render_safe_set()
        else:
            pass
            # print('t1 is None')
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
