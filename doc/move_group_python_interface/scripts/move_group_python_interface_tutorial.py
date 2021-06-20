#!/usr/bin/env python

from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import numpy as np    #for calc traj w Slerp
from scipy.spatial.transform import Slerp
from tf.transformations import quaternion_from_euler
from tf.transformations import quaternion_slerp
from scipy.spatial.transform import Rotation as R


def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True


class MoveGroupPythonIntefaceTutorial(object):
  """MoveGroupPythonIntefaceTutorial"""
  def __init__(self):
    super(MoveGroupPythonIntefaceTutorial, self).__init__()


    ## First initialize `moveit_commander`_ and a `rospy`_ node:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

    ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
    ## kinematic model and the robot's current joint states
    robot = moveit_commander.RobotCommander()

    ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
    ## for getting, setting, and updating the robot's internal understanding of the
    ## surrounding world:
    scene = moveit_commander.PlanningSceneInterface()

    ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
    ## to a planning group (group of joints).  In this tutorial the group is the primary
    ## arm joints in the Fanuc robot, so we set the group's name to "fanuc_arm".
    ## If you are using a different robot, change this value to the name of your robot
    ## arm planning group.
    ## This interface can be used to plan and execute motions:
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
    ## trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)


    # We can get the name of the reference frame for this robot:
    planning_frame = move_group.get_planning_frame()
    print("============ Planning frame: %s" % planning_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = move_group.get_end_effector_link()
    print("============ End effector link: %s" % eef_link)

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print("============ Available Planning Groups:", robot.get_group_names())

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print("============ Printing robot state")
    print(robot.get_current_state())
    print("")


    # Misc variables
    #self.table_name = ''
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names




 ####################### GO TO POSE ##########################################################################

  def go_to_pose_goal(self, goal_position_x, goal_position_y, goal_position_z, roll, pitch, yaw, steps):

    move_group = self.move_group

    ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## We can plan a motion for this group to a desired pose for the
    ## end-effector:
    current_pose = self.move_group.get_current_pose().pose 
    final_pose = geometry_msgs.msg.Pose() 
    wpose = geometry_msgs.msg.Pose()
    waypoints = []
    
    #Using polynomial interpolation for 2 positions.
    #First we need the initial position of the eef, which is given by moveit_commander and final position(the box position*t(z,0.05))
    #initial position
    xstart = current_pose.position.x
    ystart = current_pose.position.y
    zstart = current_pose.position.z
    #final position
    xend = goal_position_x #from keyboard
    yend = goal_position_y
    zend = goal_position_z + 0.05
    T = 5 #5seconds
    #boundary conditions
    dss = 0 #initial velocity
    ddss = 0 #initial acceleration
    dse = 0 #final vel
    ddse = 0 #final acc
    #steps = 30*T
    #forming polynomial (as in the formula)
    time = np.linspace(0,T,steps)
    timed = np.array((np.power(time,5),np.power(time,4),np.power(time,3),np.power(time,2),time,np.ones((1,steps))))
    A = np.array([[0,0,0,0,0,1],[T**5, T**4, T**3,T**2, T, 1],[0,0,0,0,1,0],[5*T**4,4*T**3,3*T**2,2*T,1,0],[0,0,0,2,0,0],[20*T**3,12*T**2,6*T,2,0,0]])  
    inpt = np.array([0,1,dss,dse,ddss,ddse])
    inpt.shape = (6,1)
    sol = np.linalg.inv(A).dot(inpt)
    s = sol[0]*timed[0]+sol[1]*timed[1]+sol[2]*timed[2]+sol[3]*timed[3]+sol[4]*timed[4]+sol[5]*timed[5]
    #interpolation formula x,y,z pozition coord for waypoints
    x = (1-s)*xstart + s*xend
    y = (1-s)*ystart + s*yend
    z = (1-s)*zstart + s*zend
    x.shape = (steps, 1)  #so it will have steps lines, 1 column
    y.shape = (steps, 1)
    z.shape = (steps, 1)
    #print("x: ", x)
    #print("y: ", y)
    #print("z: ", z)

    #Using Slerp(Spherical Linear Interpoletion) so the eef will go in the most optimal trajectory(uniform angular vel) from starting orientation to ending orientation.
    #orientation quaternion
    quaternion = quaternion_from_euler(roll, pitch, yaw)   #transf euler in quat
    #print("quat: ", quaternion)
    final_pose.orientation.x = quaternion[0]
    final_pose.orientation.y = quaternion[1]
    final_pose.orientation.z = quaternion[2]
    final_pose.orientation.w = quaternion[3]
    #print("finalp: ", final_pose.orientation)
    quat0 = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
    quat1 = [final_pose.orientation.x, final_pose.orientation.y, final_pose.orientation.z, final_pose.orientation.w]
  
    wpose = self.move_group.get_current_pose().pose  #starting the trajectory from the initial pose
    #print("wpose0", wpose)
    waypoints.append(copy.deepcopy(move_group.get_current_pose().pose))

    for i in range (steps-1):       #i for numbering the moves
        wpose.position.x = x[i+1]
        wpose.position.y = y[i+1]
        wpose.position.z = z[i+1]
        quat = quaternion_slerp(quat0, quat1, float(i+1)/steps, 1, shortestpath=True) #fraction devided into steps(for)
        #print("test ", float(i+1)/steps)
        wpose.orientation.x = quat[0] 
        wpose.orientation.y = quat[1]
        wpose.orientation.z = quat[2]
        wpose.orientation.w = quat[3]
        waypoints.append(copy.deepcopy(wpose))  #add the wpose at the end of the 'waypoints' list 
        #print("wpose: ", wpose)

    #print("waypoints: ", waypoints)

    
    #move_group.set_pose_target(waypoints) 
    (plan, fraction) = move_group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   0.01,        # eef_step
                                   0.0,         # jump_threshold 
                                   1)           # avoid collision
    plan = move_group.go(wait=True)

    move_group.stop()

    move_group.clear_pose_targets()

    current_pose = self.move_group.get_current_pose().pose

    return all_close(final_pose, current_pose, 0.01)

 ######################################################################







  def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
    
    box_name = self.box_name
    #table_name = self.table_name
    scene = self.scene

    ## Ensuring Collision Updates Are Received
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## If the Python node dies before publishing a collision object update message, the message
    ## could get lost and the box will not appear. To ensure that the updates are
    ## made, we wait until we see the changes reflected in the
    ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
    ## For the purpose of this tutorial, we call this function after adding,
    ## removing, attaching or detaching an object in the planning scene. We then wait
    ## until the updates have been made or ``timeout`` seconds have passed
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
      # Test if the box is in attached objects
      attached_objects = scene.get_attached_objects([box_name])
      is_attached = len(attached_objects.keys()) > 0

      # Test if the box is in the scene.
      # Note that attaching the box will remove it from known_objects
      is_known = box_name in scene.get_known_object_names()
      #is_known = table_name in scene.get_known_object_names()

      # Test if we are in the expected state
      if (box_is_attached == is_attached) and (box_is_known == is_known):
        return True

      # Sleep so that we give other threads time on the processor
      rospy.sleep(0.1)
      seconds = rospy.get_time()

    # If we exited the while loop without returning then we timed out
    return False

  ######################### ADD TABLE

  #def add_table(self, timeout=4):

    #table_name = self.table_name
    #scene = self.scene

    ##Adding table to the Planning Scene
    ##Creating table in planning scene:
    #table_pose = geometry_msgs.msg.PoseStamped()
    #table_pose.header.frame_id = "world"
    #table_pose.pose.orientation.w=1.0
    #table_pose.pose.position.x = 0.5
    #table_pose.pose.position.y = 0.5
    #table_pose.pose.position.z = 0.5
    #table_name="table"
    #scene.add_table(table_name, table_pose, size=(0.4, 0.05 ,0.4))
    
    #self.table_name=table_name
    #return self.wait_for_state_update(table_is_known=True, timeout=timeout)



  def add_box(self, goal_position_x, goal_position_y, goal_position_z, roll, pitch, yaw, timeout=4):

    box_name = self.box_name
    scene = self.scene
 
    ## Adding Objects to the Planning Scene
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## First, we will create a box in the planning scene :
    #global box_pose
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = "world"
    box_name = "box"
    box_pose.pose.position.x = goal_position_x
    box_pose.pose.position.y = goal_position_y
    box_pose.pose.position.z = goal_position_z + 0.05
    quaternion = quaternion_from_euler(roll, pitch, yaw)   #transf euler in quat
    box_pose.pose.orientation.x = quaternion[0]
    box_pose.pose.orientation.y = quaternion[1]
    box_pose.pose.orientation.z = quaternion[2]
    box_pose.pose.orientation.w = quaternion[3]
    scene.add_box(box_name, box_pose, size=(0.1, 0.1 ,0.1))

    self.box_name=box_name
    return self.wait_for_state_update(box_is_known=True, timeout=timeout)


  def attach_box(self, timeout=4):

    box_name = self.box_name
    robot = self.robot
    scene = self.scene
    eef_link = self.eef_link
    group_names = self.group_names


    ## Attaching Objects to the Robot
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    grasping_group = 'manipulator'
    touch_links = robot.get_link_names(group=grasping_group)
    scene.attach_box(eef_link, box_name, touch_links=touch_links)

    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=timeout)


  def detach_box(self, timeout=4):

    box_name = self.box_name
    scene = self.scene
    eef_link = self.eef_link


    ## Detaching Objects from the Robot
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## We can also detach and remove the object from the planning scene:
    scene.remove_attached_object(eef_link, name=box_name)

    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_known=True, box_is_attached=False, timeout=timeout)


  def remove_box(self, timeout=4):

    box_name = self.box_name
    scene = self.scene


    ## Removing Objects from the Planning Scene
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## We can remove the box from the world.
    scene.remove_world_object(box_name)


    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)



def main():
 
  try:

    print("")
    print("----------------------------------------------------------")
    print("Welcome to the MoveIt MoveGroup Python Interface Tutorial")
    print("----------------------------------------------------------")
    print("Press Ctrl-D to exit at any time")
    print("")
    input("============ Press `Enter` to begin the tutorial by setting up the moveit_commander ...")
    tutorial = MoveGroupPythonIntefaceTutorial()


    roll = int(input('Insert roll Euler angle: '))
    pitch = int(input('Insert pitch Euler angle: '))
    yaw = int(input('Insert yaw Euler angle: '))

    goal_position_x = float(input('Insert the x coordinate for box_pose: '))
    goal_position_y = float(input('Insert the y coordinate for box_pose: '))
    goal_position_z = float(input('Insert the z coordinate for box_pose: '))

    #input("============ Press `Enter` to add table to the planning scene ...")
    #tutorial.add_table()

    input("============ Press `Enter` to add a box to the planning scene ...")
    tutorial.add_box(goal_position_x, goal_position_y, goal_position_z, roll, pitch, yaw)
    
    input("============ Press `Enter` to execute a movement using a pose goal ...")
    tutorial.go_to_pose_goal(goal_position_x, goal_position_y, goal_position_z, roll, pitch, yaw, 10)
    
    input("============ Press `Enter` to attach a Box to the Fanuc robot ...")
    tutorial.attach_box()
    
    goal_position_x = float(input('Insert the x coordinate for pose_goal: '))
    goal_position_y = float(input('Insert the y coordinate for pose_goal: '))
    goal_position_z = float(input('Insert the z coordinate for pose_goal: '))

    input("============ Press `Enter` to execute a movement using a pose goal ...")                    
    tutorial.go_to_pose_goal(goal_position_x, goal_position_y, goal_position_z, roll, pitch, yaw, 10)
    
    
    input("============ Press `Enter` to detach the box from the Fanuc robot ...")
    tutorial.detach_box()

    goal_position_x = float(input('Insert the x coordinate for retire_goal: '))
    goal_position_y = float(input('Insert the y coordinate for retire_goal: '))
    goal_position_z = float(input('Insert the z coordinate for retire_goal: '))
    
    input("============ Press `Enter` to execute a movement using a pose goal ...")
    tutorial.go_to_pose_goal(goal_position_x, goal_position_y, goal_position_z, roll, pitch, yaw, 10)
    
    input("============ Press `Enter` to remove the box from the planning scene ...")
    tutorial.remove_box()


    print("============ Python tutorial demo complete!")
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return

if __name__ == '__main__':
  main()


