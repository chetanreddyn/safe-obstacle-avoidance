# safe-obstacle-avoidance

In this work, we aim to build a low level obstacle avoidance controller that does not rely on global localisation or pre-existing maps. It provides provable guarantees to safety even in an unknown and uncertain environment.  

## Implementation
- Every 0.1 seconds (with a frequency of 10 Hz), a depth image is generated from the Intel RealSense Camera.
- A safe set is computed from this depth image and the configuration of this set is published to the ROS topic "Safe_Set_Config". This is done by the script kth_rpl_obstacle_avoidance/src/scripts
/safe_set_generator2.py.
- The controller node subscribes to the safe set configuration and the controls are synthesized by using CBF formulation and inequality contraints.
- We assume that a high level controller provides the reference direction and velocity. The synthesized controls are made to be as close as possible to these reference values while complying with the inequality constraints. We calculate this by formulating a Quadratic Program.

## Results
- The Video shows an implementation of the CBF based controller successfully avoiding an obstacle. The blue arrow represents the reference direction. The control synthesis has a mathematical backing to the safety.


https://github.com/chetanreddy1412/safe-obstacle-avoidance/assets/60615610/03253815-5a0b-4eda-b857-a247d76853b1

- The trajectories of Reference Velocity and Filtered Velocity Commands (which comply with safety constraints) are depicted below for a particular segment of the path:


https://github.com/chetanreddy1412/safe-obstacle-avoidance/assets/60615610/3afb914d-0c68-4009-8323-13c9b6310793



   <img width="624" alt="Screenshot 2024-01-10 at 4 59 04 PM" src="https://github.com/chetanreddy1412/safe-obstacle-avoidance/assets/60615610/5779bd82-8b26-4e00-8a8d-4e007edd8537">

  






