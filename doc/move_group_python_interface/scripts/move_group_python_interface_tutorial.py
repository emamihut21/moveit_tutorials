#!/usr/bin/env python

from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import sin, cos
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import numpy as np    #for calc traj w Slerp
from scipy.spatial.transform import Slerp
from tf.transformations import quaternion_from_euler, euler_matrix, rotation_matrix, quaternion_slerp, quaternion_from_matrix



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

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True



class MoveGroupPythonIntefaceTutorial(object):
  """MoveGroupPythonIntefaceTutorial"""
  def __init__(self):
    super(MoveGroupPythonIntefaceTutorial, self).__init__()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

    robot = moveit_commander.RobotCommander()

    scene = moveit_commander.PlanningSceneInterface()

    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    planning_frame = move_group.get_planning_frame()
    print("============ Planning frame: %s" % planning_frame)

    eef_link = move_group.get_end_effector_link()
    print("============ End effector link: %s" % eef_link)

    group_names = robot.get_group_names()
    print("============ Available Planning Groups:", robot.get_group_names())

    print("============ Printing robot state")
    print(robot.get_current_state())
    print("")

    #self.table_name = ''
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names



 ####################### GO TO BOX POSE #########################
 
  def go_to_box_pose(self, box_position_x, box_position_y, box_position_z, box_roll, box_pitch, box_yaw):

    move_group = self.move_group
    robot = self.robot                                                 
    current_pose = self.move_group.get_current_pose().pose 
    
    Trans_z = np.matrix([[1, 0, 0, 0  ],       #Translation matrix on z 
                         [0, 1, 0, 0  ],
                         [0, 0, 1, 0.2],
                         [0, 0, 0, 1  ]])
  
    #box can be under eef => substract from z  #box can be above eef => substract from x

    #1st case. box under eef
    if (box_position_z < current_pose.position.z):
      #homogenous transformation matrix
      Rot_box = euler_matrix(box_roll, box_pitch, box_yaw)       #define transformation matrix from euler angles of the box
      print("Rot_box", Rot_box)
      Rot_box[:3,3] = np.array([box_position_x, box_position_y, box_position_z])     
      TransformationMatrix = Rot_box.dot(Trans_z)                   #Translation on z with 0.05 on the box frame for the box collision

    #2nd case. box above eef
    elif (box_position_z > current_pose.position.z):
      Rot_box = euler_matrix(box_roll, box_pitch, box_yaw)
      Rot_box[:3,3] = np.array([box_position_x, box_position_y, box_position_z])
      Rot_y = np.matrix([[ cos(90) , 0, sin(90),  0],       #rotation matrix around y
                         [ 0       , 1, 0,        0],
                         [ -sin(90), 0, cos(90),  0],
                         [0        , 0, 0,        1]])
      Rot_box = Rot_box.dot(Rot_y) #right multipl because it rotates around y of the box, not y of the robot
      TransformationMatrix = Rot_box.dot(Trans_z) 
      
    print("TF: ", TransformationMatrix)
    pose_box = geometry_msgs.msg.Pose()
    box_quaternions = quaternion_from_matrix(TransformationMatrix)
    pose_box.orientation.x = box_quaternions[0]
    pose_box.orientation.y = box_quaternions[1]
    pose_box.orientation.z = box_quaternions[2]
    pose_box.orientation.w = box_quaternions[3]
    pose_box.position.x = TransformationMatrix[0,3]
    pose_box.position.y = TransformationMatrix[1,3]
    pose_box.position.z = TransformationMatrix[2,3]
    print(pose_box)
    move_group.set_pose_target(pose_box)
    plan = move_group.go(wait=True)

    current_pose = self.move_group.get_current_pose().pose
    return all_close(pose_box, current_pose, 0.01)
    
   #################################################################



  ############## INTERPOLATION +WAYPOINTS #########################
   
  def go_to_pose_goal(self, goal_position_x, goal_position_y, goal_position_z, goal_roll, goal_pitch, goal_yaw, steps):

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
    xend = goal_position_x  #from keyboard
    yend = goal_position_y 
    zend = goal_position_z 
    T = 5 #5seconds
    #boundary conditions
    dss = 0 #initial velocity
    ddss = 0 #initial acceleration
    dse = 0 #final vel
    ddse = 0 #final acc
    #forming polynomial (as in the formula)
    time = np.linspace(0,T,steps)
    timed = np.array((np.power(time,5),np.power(time,4),np.power(time,3),np.power(time,2),time,np.ones((1,steps))))
    #print("timed", timed)
    A = np.array([[0,0,0,0,0,1],[T**5, T**4, T**3,T**2, T, 1],[0,0,0,0,1,0],[5*T**4,4*T**3,3*T**2,2*T,1,0],[0,0,0,2,0,0],[20*T**3,12*T**2,6*T,2,0,0]])  
    inpt = np.array([0,1,dss,dse,ddss,ddse])
    inpt.shape = (6,1)
    sol = np.linalg.inv(A).dot(inpt)
    s = sol[0]*timed[0]+sol[1]*timed[1]+sol[2]*timed[2]+sol[3]*timed[3]+sol[4]*timed[4]+sol[5]*timed[5]
    #interpolation formula x,y,z pozition coord for waypoints
    x = (1-s)*xstart + s*xend
    y = (1-s)*ystart + s*yend
    z = (1-s)*zstart + s*zend
    x.shape = (steps)  #so it will have steps lines, 1 column
    y.shape = (steps)
    z.shape = (steps)

    #Using Slerp(Spherical Linear Interpoletion) so the eef will go in the most optimal trajectory(uniform angular vel) from starting orientation to ending orientation.
    #orientation quaternion
    quaternion = quaternion_from_euler(goal_roll, goal_pitch, goal_yaw)   #transf euler in quat
    #print("quat: ", quaternion)
    final_pose.orientation.x = quaternion[0]
    final_pose.orientation.y = quaternion[1]
    final_pose.orientation.z = quaternion[2]
    final_pose.orientation.w = quaternion[3]
    print("finalp: ", final_pose.orientation)
    quat0 = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
    print("quat0 ",quat0)
    quat1 = [final_pose.orientation.x, final_pose.orientation.y, final_pose.orientation.z, final_pose.orientation.w]
    
    wpose = self.move_group.get_current_pose().pose  #starting the trajectory from the initial pose
    waypoints.append(copy.deepcopy(move_group.get_current_pose().pose))

    for i in range (steps-1):       #i for numbering the moves
        wpose.position.x = x[i+1]
        wpose.position.y = y[i+1]
        wpose.position.z = z[i+1]
        quat = quaternion_slerp(quat0, quat1, float(i+1)/(steps-1), 1, shortestpath=True) #fraction devided into steps(for)
        wpose.orientation.x = quat[0] 
        wpose.orientation.y = quat[1]
        wpose.orientation.z = quat[2]
        wpose.orientation.w = quat[3]
        waypoints.append(copy.deepcopy(wpose))  #add the wpose at the end of the 'waypoints' list 

    #print("last waypoint", waypoints[-1])
    #print("waypoints", waypoints)
    (plan, fraction) = move_group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   0.01,        # eef_step
                                   0.0,         # jump_threshold 
                                   1)           # avoid collision    
    rospy.sleep(5)
    #print("Planning fraction")
    move_group.stop()
    current_pose = self.move_group.get_current_pose().pose
    return all_close(final_pose, current_pose, 0.01)

   ######################################################################



  ######################### GO TO RETIRE POSE ###########################

  def go_to_retire_goal(self, retire_x, retire_y, retire_z):

    move_group = self.move_group
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = 7.3123010772e-14     #initial orientation from moveit_commander
    pose_goal.orientation.x = -0.707106781188
    pose_goal.orientation.y = -7.31230107717e-14
    pose_goal.orientation.z = -0.707106781185
    pose_goal.position.x = retire_x
    pose_goal.position.y = retire_y
    pose_goal.position.z = retire_z

    move_group.set_pose_target(pose_goal)
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    current_pose = self.move_group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)

   ######################################################################



  def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
    
    box_name = self.box_name
    #table_name = self.table_name
    scene = self.scene
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
    return False



  def add_box(self, box_position_x, box_position_y, box_position_z, box_roll, box_pitch, box_yaw, timeout=4):

    box_name = self.box_name
    scene = self.scene
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = "world"
    box_name = "box"
    box_pose.pose.position.x = box_position_x
    box_pose.pose.position.y = box_position_y
    box_pose.pose.position.z = box_position_z 
    quaternion = quaternion_from_euler(box_roll, box_pitch, box_yaw)   #transf euler in quat
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
    grasping_group = 'manipulator'
    touch_links = robot.get_link_names(group=grasping_group)
    scene.attach_box(eef_link, box_name, touch_links=touch_links)
    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=timeout)



  def detach_box(self, timeout=4):

    box_name = self.box_name
    scene = self.scene
    eef_link = self.eef_link
    scene.remove_attached_object(eef_link, name=box_name)
    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_known=True, box_is_attached=False, timeout=timeout)



  def remove_box(self, timeout=4):

    box_name = self.box_name
    scene = self.scene
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


    box_roll = int(input('Insert roll Euler angle: '))
    box_pitch = int(input('Insert pitch Euler angle: '))
    box_yaw = int(input('Insert yaw Euler angle: '))
    box_position_x = float(input('Insert the x coordinate for box_pose: '))
    box_position_y = float(input('Insert the y coordinate for box_pose: '))
    box_position_z = float(input('Insert the z coordinate for box_pose: '))

    input("============ Press `Enter` to add a box to the planning scene ...")
    tutorial.add_box(box_position_x, box_position_y, box_position_z, box_roll, box_pitch, box_yaw)
    
    input("============ Press `Enter` to execute a movement using the box pose ...")
    tutorial.go_to_box_pose(box_position_x, box_position_y, box_position_z, box_roll, box_pitch, box_yaw)
    
    input("============ Press `Enter` to attach a Box to the Fanuc robot ...")
    tutorial.attach_box()
    
    goal_roll = int(input('Insert roll Euler angle for the goal: '))
    goal_pitch = int(input('Insert pitch Euler angle for the goal: '))
    goal_yaw = int(input('Insert yaw Euler angle for the goal: '))
    goal_position_x = float(input('Insert the x coordinate for goal_pose: '))
    goal_position_y = float(input('Insert the y coordinate for goal_pose: '))
    goal_position_z = float(input('Insert the z coordinate for goal_pose: '))

    input("============ Press `Enter` to execute a movement using a pose goal ...")                    
    tutorial.go_to_pose_goal(goal_position_x, goal_position_y, goal_position_z, goal_roll, goal_pitch, goal_yaw, 10)
    
    
    input("============ Press `Enter` to detach the box from the Fanuc robot ...")
    tutorial.detach_box()

    retire_x = float(input('Insert the x coordinate for retire position: '))
    retire_y = float(input('Insert the y coordinate for retire position: '))
    retire_z = float(input('Insert the z coordinate for retire position: '))
    
    input("============ Press `Enter` to execute a movement for retire pose ...")
    tutorial.go_to_retire_goal(retire_x, retire_y, retire_z)
    
    input("============ Press `Enter` to remove the box from the planning scene ...")
    tutorial.remove_box()


    print("============ Python tutorial demo complete!")
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return

if __name__ == '__main__':
  main()


