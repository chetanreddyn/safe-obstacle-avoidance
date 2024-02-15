#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from kth_rpl_obstacle_avoidance.msg import SafeSetEllipsoidsConfig 
import tf.transformations
import numpy as np
from cvxopt import matrix, solvers
import cv2


 

# def publish_velocity_commands():
#     rospy.init_node('velocity_publisher', anonymous=True)
#     pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

#     rate = rospy.Rate(100)  # 10 Hz (adjust the frequency as needed)

#     while not rospy.is_shutdown():
#         cmd = Twist()
#         cmd.linear.x = 0.1  # Linear velocity (m/s)
#         cmd.angular.z = 0  # Angular velocity (radians/s)
#         pub.publish(cmd)
    

#         rate.sleep()

class Controller:
    def __init__(self,robot_config, control_algo_config,render_config,reference_angle):

        rospy.Subscriber('/odom',Odometry,self.odom_callback)
        rospy.Subscriber('/safe_set_config',SafeSetEllipsoidsConfig,self.update_safeset_callback)
        rospy.Subscriber('/ref_vel',Twist,self.update_ref_vel)

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        self.theta_min  = robot_config['theta_min']
        self.theta_max  = robot_config['theta_max']
        self.max_depth  = robot_config['max_depth']
        self.min_depth  = robot_config['min_depth']
        self.r_limit    = robot_config['r_limit']
        self.r0         = robot_config['r0']
        self.l          = robot_config['l']

        # Algo Related
        self.gamma      = control_algo_config['gamma']
        self.Q          = np.array([[control_algo_config['Q_u1'],0],[0,1]]).astype(np.float64)
        self.u1_limit   = control_algo_config['u1_limit']  # Upper Bound of Linear Velocity
        self.u2_limit   = control_algo_config['u2_limit']  # Upper Bound for the Angular Velocity
        self.k          = control_algo_config['k']  # Reference Angular Velocity = -k*theta
        self.u1_ref     = control_algo_config['u1_ref']  # Forward Velocity Reference

        self.reference_angle_global = reference_angle # in global coordinate system


        # Rendering Properties
        self.render_plt            = render_config['render_plt']
        self.y_limit               = render_config['y_limit'] # Indicates the maximum local region to be shown in the rendered image (= r_limit or r_max_ellipse)
        self.x_limit               = self.y_limit

        self.img_height            = render_config['img_height'] # For rendering

        self.scale                 = self.img_height/(2*self.y_limit)
        self.img_width             = int(2*self.scale*self.x_limit)
        self.ellipse_color         = render_config['ellipse_color']
        self.thickness             = render_config['thickness']
        self.theta_ellipse_resolution = render_config['theta_ellipse_resolution']
        
        self.ellipses_data = None # Used to store the ellipse points for rendering



        self.robot_state   = None # Used to store the global coordinates of the robot state
        self.safeset    = {'C_ellipses':None,'A_ellipses':None,'origin':None,'received':False} 
        self.h_i_ellipses   = None
        self.U_ref_msg = None
        # Shape of C_ellipses must be (P,2,1) and shape of A_ellipses must be (P,2,2) where P is the number of ellipses
        
    # def get_X_pixel(self,x_local,y_local):
    #     # Converts from continuous coordinate system to pixel indices in the np image

    #     y_limit = self.y_limit

    #     scale = self.scale
    #     x_limit = self.x_limit

    #     y_shift = y_limit
    #     x_shift = -x_limit/2
    #     X_pixel = np.array([[0,-1],[1,0]])@np.array([[x_local-x_shift],[y_local-y_shift]])*scale
    #     x_pixel,y_pixel = int(X_pixel[0,0]),int(X_pixel[1,0])
    #     return y_pixel,x_pixel

    def update_ref_vel(self,data):
        self.U_ref_msg = data

        # self.pub.publish(self.U_ref_msg)

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

    def get_ellipse_points(self,center,semi_major,semi_minor,theta_i):
        '''
        Generates Ellipse Points (used in visualising) given the center, semi_major axis length, semi_minor axis length and orientation theta_i
        Returns: np array with (2,theta_ellipse_resolution)
        '''
        # Create an array of angles from 0 to 2*pi
        theta_ellipse_resolution = self.theta_ellipse_resolution
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

    def update_safeset_callback(self,data_msg):
        '''
        Assuming that the shape of data is (P,5) where row i is (cx_i,cy_i,a_i,b_i,theta_i) as defined in Formulation 3 (2)
        '''
        # print('SafeSet Updated, Data Shape = ',len(data_msg.data))

        columns = data_msg.columns
        data = np.array(data_msg.data).reshape(-1,columns)
        # print("Data Received and Interpreted!: data.shape=",data.shape)
        P = data.shape[0]
        C_ellipses = np.zeros(shape=(P,2,1))
        A_ellipses = np.zeros(shape=(P,2,2))

        ellipses_data_plt = {}

        for i in range(data.shape[0]):
            row = data[i]

            theta_i = row[4]
            a_i = row[2]
            b_i = row[3]
            
            assert a_i >0 and b_i>0
            
            P_i = np.array([[1/a_i**2,0],[0,1/b_i**2]])
            R_i = np.array([[np.cos(theta_i), -np.sin(theta_i)],
                                        [np.sin(theta_i), np.cos(theta_i)]])
            A_i = R_i@P_i@R_i.T

            assert A_i.shape == (2,2)

            A_ellipses[i] = A_i
            C_i = row[:2].reshape(2,1)
            C_ellipses[i] = C_i


            # For Rendering Purposes
            ellipse_points = self.get_ellipse_points(center=C_i,semi_major=a_i,semi_minor=b_i,theta_i=theta_i) # Shape = (2,resolution)
            ellipses_data_plt[theta_i] = ellipse_points

        self.safeset['C_ellipses'] = C_ellipses
        self.safeset['A_ellipses'] = A_ellipses
        self.safeset['origin'] = self.robot_state # The safeset origin in global coordinate system
        self.safeset['received'] = True
        self.ellipses_data = ellipses_data_plt



        # for i in range(depth_data_processed.shape[0]): # i represents the i th point and the i th ellipse

        #     theta_i,r_i = depth_data_processed[i][0],depth_data_processed[i][1]
        #     C = ((r_max_ellipse+r_i)/2)*np.array([[np.cos(theta_i)],[np.sin(theta_i)]]) # Center of the Ellipse shape = (2,1)
        #     a_i = (r_max_ellipse-r_i)/2 + d_s # Semi Major Axis Length Shape = scalar

        #     ellipse_points = self.get_ellipse_points(center=C,semi_major=a_i,semi_minor=b_i,theta_i=theta_i) # Shape = (2,resolution)
        #     ellipses_data_plt[theta_i] = ellipse_points
            
        #     ellipses_data_msg[i][0] = C[0][0] # x coordinate of center
        #     ellipses_data_msg[i][1] = C[1][0] # y coordinate of center
        #     ellipses_data_msg[i][2] = a_i # semi major axis length
        #     ellipses_data_msg[i][3] = b_i # semi minor axis length
        #     ellipses_data_msg[i][4] = theta_i # Orientation of the Ellipse

        # self.ellipses_data_msg = ellipses_data_msg
        # self.ellipses_data = ellipses_data_plt


    ######## HELPER FUNCTIONS TO SYNTHESIS THE CONTROL ##########
    # def get_Xp(self,Xc,l):
    #     xp = Xc[0]+l*np.cos(Xc[2])
    #     yp = Xc[1]+l*np.sin(Xc[2])
    #     thetap = Xc[2]

    #     Xp = np.array([xp,yp,thetap])
    #     return Xp

    # def h_val(self,theta_set_i,X,l):
    #     '''
    #     Helper Function to write the functions as given in FORMULATION 2 (2) in NOTES
    #     '''
    #     x = X[0][0]
    #     y = X[1][0]
    #     theta = X[2][0]
    #     return -np.tan(theta_set_i)*x + y - l*np.sin(theta) + l*np.tan(theta_set_i)*np.cos(theta)


    def get_G_h_opt(self,X,U_ref):
        # G and h are the inequalities for the CBF in the optimisation problem
    
        # gamma = self.gamma
        # l = self.l
        # t1 = self.safeset['t1']
        # t2 = self.safeset['t2']
        # depth = self.safeset['depth']


        # theta = X[2][0] #extracting theta for easier usage in later formulae

        # #if either t1 or t2 is a 90 degree angle
        # if t1==np.pi/2:
        #     t1 = t1 - 0.1*np.pi/180 #Nudging t1 by 0.1 degree if it is 90 degrees
        # if t2==np.pi/2:
        #     t2 = t2 + 0.1*np.pi/180

        # # Finding a1,a2,a3 depending on theta and safe set config (NOTE: these a_i's are different from that implemented below, look at written notes)
        # if 0<=t1<np.pi/2:
        #     a1 = np.array([-np.tan(t1),1,-(l*np.cos(theta)+l*np.sin(theta)*np.tan(t1))])
        #     h1 = self.h_val(t1,X,l)

        # elif np.pi/2<t1<=np.pi:
        #     a1 = -np.array([-np.tan(t1),1,-(l*np.cos(theta)+l*np.sin(theta)*np.tan(t1))])
        #     h1 = -self.h_val(t1,X,l)

        # if 0<=t2<np.pi/2:
        #     a2 = np.array([np.tan(t2),-1,l*np.cos(theta)+l*np.sin(theta)*np.tan(t2)])
        #     h2 = -self.h_val(t2,X,l)

        # elif np.pi/2<t2<=np.pi:
        #     a2 = -np.array([np.tan(t2),-1,l*np.cos(theta)+l*np.sin(theta)*np.tan(t2)])
        #     h2 = self.h_val(t2,X,l)


        # a3 = np.array([0,-1,l*np.cos(theta)])
        # h3 = -X[1][0]+l*np.sin(theta)+depth

        # # From system dynamics: x_dot = Ax + Bu
        # B = np.array([[np.cos(theta),0],[np.sin(theta),0],[0,1]])

        # G_opt = -np.vstack([a1@B,a2@B,a3@B,np.array([-1,0]),np.array([0,-1])])
        # h_opt = np.vstack([gamma*h1+a1@B@U_ref,gamma*h2+a2@B@U_ref,gamma*h3+a3@B@U_ref,np.array([u1_limit-U_ref[0][0]]),np.array([u2_limit-U_ref[1][0]])])

        # assert h1>=0, 'h1 = {}'.format(h1)
        # assert h2>=0, 'h2 = {}'.format(h2)
        # assert h3>=0, 'h3 = {}, h3 = {:.2f} + {:.2f} + {:.2f}'.format(h3,-X[1][0],l*np.sin(theta),depth)
        
        # info = {'a1':a1,'a2':a2,'a3':a3,'G_opt':G_opt,'h_opt':h_opt,'B_model':B}


        A_ellipses = self.safeset['A_ellipses'] # Shape = (P,2,2)
        C_ellipses = self.safeset['C_ellipses'] # Shape = (P,2,1)
        gamma = self.gamma
        l = self.l
        nu = 2 # Number of Controls
        P = A_ellipses.shape[0]

        u1_limit = self.u1_limit
        u2_limit = self.u2_limit


        G_opt = np.zeros((P+2,nu))
        h_opt = np.zeros((P+2,1))

        xp    = X[0][0]
        yp    = X[1][0]
        theta_p = X[2][0] 

        # # From system dynamics: x_dot = Ax + Bu
        B1 = np.array([[1,0,l*np.sin(theta_p)],
                       [0,1,-l*np.cos(theta_p)]])
        
        B = np.array([[np.cos(theta_p),0],[np.sin(theta_p),0],[0,1]])


        h_i_ellipses   = np.zeros(shape=(P,1))
        info = {}

        for i in range(P):
            C_i = C_ellipses[i]
            A_i = A_ellipses[i]

            x_minus_c = np.array([[xp - l*np.cos(theta_p) - C_i[0][0]],
                                  [yp - l*np.sin(theta_p) - C_i[1][0]]])
            

            # Computing CBF or h_i values 
            h_i = (x_minus_c.T)@A_i@x_minus_c-1
            h_i_ellipses[i] = h_i[0][0]

            if h_i[0][0]<0:
                orientation_ellipse = np.arctan2(C_i[1][0],C_i[0][0])
                print("Outside Safe Region, Angle of Ellipse: {:.2f}".format(orientation_ellipse*180/np.pi))
                raise

            a_i = (x_minus_c.T)@(A_i + A_i.T)@B1
            G_opt_i = -a_i@B

            h_opt_i = gamma*h_i + a_i@B@U_ref

            G_opt[i] = G_opt_i
            h_opt[i] = h_opt_i

            # For debugging 
            # orientation_ellipse = round(np.arctan2(C_i[1][0],C_i[0][0])*180/np.pi)
            # info[orientation_ellipse] = {'A_i':A_i,
            #                              'C_i':C_i,
            #                              'x_minus_c':x_minus_c,
            #                              'h_i':h_i,
            #                              'a_i':a_i,
            #                              'G_opt_i':G_opt_i,
            #                              'h_opt_i':h_opt_i
            #                              }


        # self.h_i_ellipses = h_i_ellipses

        G_opt[-2,:] = np.array([[1,0]])
        G_opt[-1,:] = np.array([[0,1]])
        h_opt[-2][0]= u1_limit - U_ref[0][0]
        h_opt[-1][0]= u2_limit - U_ref[1][0]
        info['h_i_min'] = h_i_ellipses.min()
        # Need to append u limits to G
        return G_opt,h_opt,info

    def get_U(self,X,U_ref):
        
        # U is synthesised by solving the optimisation problem
        
        # l = self.l
        # t1 = self.safeset['t1']
        # t2 = self.safeset['t2']
        # depth = self.safeset['depth']

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
        # t_avg = (self.safeset['t1']+self.safeset['t2'])/2
        # # print(t_avg*180/np.pi)
        # buffer_radius = self.buffer_radius
        # print(t_avg*180/np.pi,X_local)
        # X_local = X_local + buffer_radius*np.array([[np.cos(t_avg)],[np.sin(t_avg)]])
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

        # print("Safe Set Configuration Received, ")
        # print("Local=({:.2f},{:.2f},{:.2f}), Global=({:.2f},{:.2f},{:.2f}), Del Theta={:.2f}".format(X[0,0],X[1,0],X[2,0]*180/np.pi,
        #                                                                                              x_global,y_global,yaw_global*180/np.pi,
        #                                                                                              theta_deviation*180/np.pi))

        # u1_ref = self.u1_ref
        # u2_ref = -self.k*(theta_deviation)
        if self.U_ref_msg:
            u1_ref = -self.U_ref_msg.linear.x
            u2_ref = self.U_ref_msg.angular.z
            U_ref = np.array([u1_ref,u2_ref]).reshape(2,1)
            U,info = self.get_U(X,U_ref)
            print("h_min: {:.2f}, Local=({:.2f},{:.2f},{:.2f}), Global=({:.2f},{:.2f},{:.2f}),U=({:.2f},{:.2f}), U_ref=({:.2f},{:.2f})".format(
                info['h_i_min'],X[0,0],X[1,0],X[2,0]*180/np.pi,
                                                                                                        x_global,y_global,yaw_global*180/np.pi,
                                                                                                        U[0,0],U[1,0],
                                                                                                        U_ref[0,0],U_ref[1,0]))
            
            cmd = Twist()
            cmd.linear.x = -U[0,0]  # Linear velocity (m/s)
            cmd.angular.z = U[1,0]  # Angular velocity (radians/s)
            self.pub.publish(cmd)

    def get_points(self,theta,depth):
        other_point = (depth/np.tan(theta),depth)
        return [(0,0),other_point]
    
    def get_X_pixel(self,X_cartesian):
        '''
        Shape of Input must be (2,P)
        '''
        x_limit = self.x_limit
        y_limit = self.y_limit
        scale = self.scale
        X_pixel = ((np.array([[1,0],[0,-1]])@X_cartesian + np.array([[x_limit],[y_limit]]))*scale).astype(np.int32)

        return X_pixel


    def render_safe_set(self):

        img_height = self.img_height
        img_width = self.img_width

        ellipse_color = self.ellipse_color
        thickness = self.thickness
        isClosed = True

        # canvas = np.ones(shape=(img_height,img_width,3),dtype=np.int8)*255
        canvas = np.ones(shape=(img_height,img_width,3)) # Uncomment this in ROS and Comment the above


        for ellipse_points in self.ellipses_data.values():
            X_pixel = self.get_X_pixel(ellipse_points)
            cv2.polylines(canvas,[X_pixel.T.reshape(-1,1,2)],isClosed,ellipse_color,thickness)   
        
        
        # img_height = self.img_height
        # img_width = self.img_width
        # t1 = self.safeset['t1']
        # t2 = self.safeset['t2']
        # depth = self.safeset['depth']

        # canvas = np.ones(shape=(img_height,img_width,3))
        # # Define the color (BGR format, (0, 0, 255) is red)
        arrow_len = 0.25
        ref_arrow_len = 0.3
        # color = (0, 10, 0)
        robot_arrow_color = (0,255,0)
        reference_arrow_color = (255,0,0)
        thickness = int(img_width/200)

        # # Define the radius of the circle which is used to represent the robot point (IN PIXELS)
        radius = 2

        # right_line = self.get_points(t1,depth)
        # left_line = self.get_points(t2,depth)

        # cv2.line(canvas, self.get_X_pixel(*right_line[0]), self.get_X_pixel(*right_line[1]), color, thickness)
        # cv2.line(canvas, self.get_X_pixel(*left_line[0]), self.get_X_pixel(*left_line[1]), color, thickness)
        # cv2.line(canvas, self.get_X_pixel(*left_line[1]), self.get_X_pixel(*right_line[1]), color, thickness)

        X_bot_pixel = self.get_X_pixel(np.array([[self.x_local],[self.y_local]]))

        cv2.circle(canvas, (X_bot_pixel[0][0],X_bot_pixel[1][0]),radius, color=(0,0,255), thickness=2)

        x2 = self.x_local+arrow_len*np.cos(self.yaw_local)
        y2 = self.y_local+arrow_len*np.sin(self.yaw_local)


        X_bot_pixel2 = self.get_X_pixel(np.array([[x2],[y2]])) 
        # print("{:.2f},{:.2f}| Pixel1{}: | Pixel2 {}".format(x2,y2, X_bot_pixel,X_bot_pixel2))

        # The order is swapped because the parameter for cv2.arrowline is (x,y) 
        # but we already swapped once when getting X_pixel as we had used a rotation matrix

        cv2.arrowedLine(canvas,(X_bot_pixel[0][0],X_bot_pixel[1][0]),(X_bot_pixel2[0][0],X_bot_pixel2[1][0]),color = robot_arrow_color,thickness=int(thickness))

        x2_ref = self.x_local+ref_arrow_len*np.cos(self.reference_angle_local)
        y2_ref = self.y_local+ref_arrow_len*np.sin(self.reference_angle_local)
        X_ref_pixel2 = self.get_X_pixel(np.array([[x2_ref],[y2_ref]]))
        cv2.arrowedLine(canvas,(X_bot_pixel[0][0],X_bot_pixel[1][0]),(X_ref_pixel2[0][0],X_ref_pixel2[1][0]),color = reference_arrow_color,thickness=int(thickness))


        self.rendered_data = canvas

        cv2.imshow('Safe Set',canvas)
        # # cv2.imshow("RGB Image",rgb_rbesized)
        # # cv2.imshow("Depth Image",depth_resized)
        cv2.waitKey(1)




def main():
    rospy.init_node("controller")

    control_algo_config = {'gamma':0.1, #High gamma means the safety filter acts only near the boundary
                        'Q_u1':100,
                        'u1_limit':0.26,
                        'u2_limit':1.82,
                        'k':1,
                        'u1_ref':0.1
                        }


    robot_config = {'theta_min':30*np.pi/180, # Minimum Angle of the Camera
                    'theta_max':150*np.pi/180, # Maximum Angle of the Camera
                    'max_depth':5, # Maximum Depth Perceived by the Camera (Used in the Loss Function)
                    'min_depth':0.2, # Minimum Depth Perceived by the Camera
                    'r_limit'  :2.5, # Maximum Radial Distance to Depict in the Plots
                    'r0'       :0.15, # Robot Radius 
                    'l'        :-0.08, #Distance of axle from Centre
                    # 'resolution': 1920
                    }

    # Rendering Properties
    render_config = {'render_plt':True,
                    'y_limit': 6, # Indicates the maximum local region to be shown in the rendered image (= r_limit or r_max_ellipse+d_s)
                    'img_height': 300,
                    'ellipse_color':(255,0,0),
                    'thickness': 1,
                    'theta_ellipse_resolution':20
                    }

    reference_angle = 0*np.pi/180

    controller_obj = Controller(robot_config=robot_config,control_algo_config=control_algo_config,render_config=render_config,reference_angle=reference_angle)

    rate = rospy.Rate(1000)

    while not rospy.is_shutdown():
        # print('here')
        current_time = rospy.get_rostime()

        # Print the current ROS time
        # print("Current ROS Time: %f seconds" % current_time.to_sec())

        # # controller_obj.vel_publish()
        if controller_obj.safeset['received']==True:

            # print("Safe Set Configuration Received!")
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