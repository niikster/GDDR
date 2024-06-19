# System of Grasp Detection for Debris Removal
The software was meant to be used as a Bachelor project in Robotics to create a prototype system for debris removal tasks.
Right now only the subsystem of technical vision for grasp detection has been implemented.

> [!NOTE]
> The system is a __prototype__ and was tested on artificially made debris. The dataset is presented [here](https://app.roboflow.com/instance-segmentation-for-debris-removal).

## Overview
The software was developed using Ubuntu 20.04, ROS Noetic and Python for camera Asus Xtion Pro Live. 
It is used to determine feasibility of grasping the object of interest on its faces using a two-finger parallel gripper. 
The program returns set of points on the object's face from which the grasp can be executed.

## Requirements
Below are listed the requirements:

* Ubuntu 20.04
* ROS Noetic
* Python >= 3.6
* Camera Asus Xtion Pro Live

## Installation
Installing the package into the ROS workspace with Linux command line:

```
cd ~/catkin_ws/src
git clone https://github.com/niikster/GDDR.git
cd ..
catkin_make
```
Install ROS package for Asus Xtion Pro Live camera:

```
sudo apt-get install ros-noetic-openni2-launch
```

## Usage
The sequence of actions to enable the software is as follows:
1. Camera package;
2. Segmentation node;
3. Depth filtering node;
4. Depth data calibration package;
5. The node for determining the feasibility of grasp.
	
 The above sequence of actions can be summarized by the following commands in the terminal:
 ```
roslaunch openni2_launch openni2.launch
roslaunch GDDR le.launch
rosrun GDDR detect_grasp_node.py
```
Then the results can be visualized using rviz: `rosrun rviz rviz`.
The topics of interest start with `/y/` prefix.
