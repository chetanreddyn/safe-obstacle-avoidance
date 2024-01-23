# safe-obstacle-avoidance

In this work, we aim to build a low level obstacle avoidance controller using depth images that does not rely on global localisation or pre-existing maps. Using Control Barrier Functions (CBFs), we provide provable guarantees to safety even in an unknown and uncertain environment. 

## Folder Structure:
- **ROS Packages**
  - **kth_rpl_obstacle_avoidance**
     - **msg** : Has the .msg files for specifying the safeset configuration msg type.
     - **src/scripts**
       - **Ellipsoid_Approximation** : Has the files to generate the safesets and the CBF-controller using ellipsoidal approximation for unsafe regions
       - **Triangular_SafeSet** : Has the files to generate the triagular safesets and the CBF-controller
     - **CMakeLists.txt**
     - **default.rviz**
     - **package.xml**
  - **turtlebot3_descriptions** : Package required to simulate the turtlebot environment
  - **turtlebot3_simulations**  : Package required to simulate the turtlebot environment
- **Other Files** : Has the files inclusing a notebook used outside ROS to develop the algorithm
- **Media**       : Has some videos and pictures from Gazebo
  
## Implementation
- Every 0.1 seconds (with a frequency of 10 Hz), a depth image is generated from the Intel RealSense Camera.
- A safe set is computed from this depth image and the configuration of this set is published to the ROS topic "Safe_Set_Config". This is done by the script kth_rpl_obstacle_avoidance/src/scripts
/safe_set_generator2.py.
- The controller node subscribes to the safe set configuration and the controls are synthesized by using CBF formulation and inequality contraints.
- We assume that a high level controller provides the reference direction and velocity. The synthesized controls are made to be as close as possible to these reference values while complying with the inequality constraints. We calculate this by formulating a Quadratic Program.

Instructions to Run the Code:
- After installing the ROS Packages, execute the commands as given below.
- `roslaunch turtlebot3_gazebo turtlebot3_house.launch` This launches the Gazebo environment and spawns the turtlebot.
- `roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch` This launches the teleop control, which helps control the turtlebot using the keyboard keys w,a,s,d,x.
- `rosrun kth_rpl_obstacle_avoidance safe_set_generator_ellipsoid.py` This starts the script to generate the safe set from the depth image using ellipsoid approximation.



## Results
- The Video shows an implementation of the CBF based controller successfully avoiding an obstacle. The blue arrow represents the reference direction. The control synthesis has a mathematical backing to the safety.



https://github.com/chetanreddy1412/safe-obstacle-avoidance/assets/60615610/f7d4e3cd-c5c5-4b7a-a759-1f2a176249a8




- The trajectories of Reference Velocity and Filtered Velocity Commands (which comply with safety constraints) are depicted below for a particular segment of the path:


https://github.com/chetanreddy1412/safe-obstacle-avoidance/assets/60615610/3afb914d-0c68-4009-8323-13c9b6310793



   <img width="624" alt="Screenshot 2024-01-10 at 4 59 04 PM" src="https://github.com/chetanreddy1412/safe-obstacle-avoidance/assets/60615610/5779bd82-8b26-4e00-8a8d-4e007edd8537">

  






