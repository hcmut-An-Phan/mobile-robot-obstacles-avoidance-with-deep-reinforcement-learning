#ifndef TURTLEBOT3_WAFFLE_H_
#define TURTLEBOT3_WAFFLE_H_

#define NAME                             "Waffle Pi"

#define PI                               3.14159265359

#define WHEEL_RADIUS                     0.033           // meter
#define WHEEL_SEPARATION                 0.287           // meter 
#define TURNING_RADIUS                   0.1435          // meter 
#define ROBOT_RADIUS                     0.220           // meter 
#define ENCODER_MIN                      -2147483648     // raw
#define ENCODER_MAX                      2147483648      // raw

#define MAX_LINEAR_VELOCITY              (WHEEL_RADIUS * 2 * 3.14159265359 * 77 / 60) // m/s  
#define MAX_ANGULAR_VELOCITY             (MAX_LINEAR_VELOCITY / TURNING_RADIUS)       // rad/s

#define MIN_LINEAR_VELOCITY              -MAX_LINEAR_VELOCITY  
#define MIN_ANGULAR_VELOCITY             -MAX_ANGULAR_VELOCITY 

#endif  //TURTLEBOT3_WAFFLE_H_