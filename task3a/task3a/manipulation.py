#!/usr/bin/env python3

'''

# Team ID: eYRC#1912

# Theme: Krishi coBot

# Author List: Chirag Sharma, K P R B Revanth Sri Raj, CHandhan GS

# Filename: task2a_manipulation.py

# Functions: main, __init__, handle_tf_messages, handle_delta_joint_commands, convert_quaternion_to_euler, setup_waypoints_from_tf, display_all_waypoints, display_coordinates, load_collision_shapes, setup_kinematic_chain, calculate_jacobian_matrix, calculate_damped_pseudoinverse, calculate_joint_velocities, detect_singularity, handle_joint_states, handle_twist_commands, stop_robot, get_current_end_effector_pose, calculate_position_error, calculate_orientation_error, generate_twist_to_target, navigate_to_waypoints, calculate_twist, attach_object, attach_response_callback, detach_object, detach_response_callback, resolve_package_uri, get_link_pose, create_fcl_object, create_robot_collision_objects, update_collision_object_pose, testing, create_boundary_planes, publish_plane_markers, convert_to_kdl_pose, convert_to_kdl_inertia, convert_to_kdl_joint, add_children_to_tree, build_tree_from_urdf

# Global variables: None

This script combines ArUco marker detection and bad fruit detection using HSV color detection. It uses an RGB-D camera to spot both ArUco markers and bad fruits, then publishes their 3D positions as TF frames so the robot knows where everything is.

'''

import numpy as np
import tf_transformations
import urdf_parser_py
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
import rclpy
from geometry_msgs.msg import Twist, TransformStamped, TwistStamped
from control_msgs.msg import JointJog
from std_srvs.srv import SetBool
import trimesh
import fcl
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float32, Bool, String
from tf2_ros import TransformListener, Buffer
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
# from linkattacher_msgs.srv import AttachLink, DetachLink  # Commented out as per requirements
from ament_index_python.packages import get_package_share_directory
import os
import time
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from collections import defaultdict
import math

ADJACENT_PAIRS = {
    ("base_link_inertia", "shoulder_link"),
    ("base_link_inertia", "upper_arm_link"),
    ("base_link", "shoulder_link"),
    ("shoulder_link", "upper_arm_link"),
    ("upper_arm_link", "forearm_link"),
    ("forearm_link", "wrist_1_link"),
    ("wrist_1_link", "wrist_2_link"),
    ("wrist_2_link", "wrist_3_link"),
}



PLANE_SIZE = 1.0        # Size of collision planes
PLANE_THICKNESS = 0.01  # thickness of planes
PLANE_ALPHA = 0.3       # Transparency (0.0 = invisible, 1.0 = solid)



def convert_to_kdl_pose(pose):

    '''

    Purpose:

    ---

    Converts a URDF pose object into a PyKDL Frame for kinematic calculations

    Input Arguments:

    ---

    `pose` : URDF pose object

    The pose containing position (xyz) and orientation (rpy) information

    Returns:

    ---

    `frame` : kdl.Frame

    A PyKDL frame representing the pose in 3D space

    Example call:

    ---

    kdl_frame = _toKdlPose(urdf_link.origin)

    '''

    # URDF might have RPY OR XYZ unspecified. Both default to zeros
    rpy = pose.rpy if pose and pose.rpy and len(pose.rpy) == 3 else [0, 0, 0]
    xyz = pose.xyz if pose and pose.xyz and len(pose.xyz) == 3 else [0, 0, 0]

    return kdl.Frame(
          kdl.Rotation.RPY(*rpy),
          kdl.Vector(*xyz))


def convert_to_kdl_inertia(i):

    '''

    Purpose:

    ---

    Converts a URDF inertial object into a PyKDL RigidBodyInertia for dynamics calculations

    Input Arguments:

    ---

    `i` : URDF inertial object

    The inertial properties including mass, origin, and inertia tensor

    Returns:

    ---

    `inertia` : kdl.RigidBodyInertia

    A PyKDL rigid body inertia object for the link

    Example call:

    ---

    kdl_inertia = _toKdlInertia(urdf_link.inertial)

    '''

    # kdl specifies the inertia in the reference frame of the link, the urdf
    # specifies the inertia in the inertia reference frame
    origin = convert_to_kdl_pose(i.origin)
    inertia = i.inertia
    return origin.M * kdl.RigidBodyInertia(
            i.mass, origin.p,
            kdl.RotationalInertia(inertia.ixx, inertia.iyy, inertia.izz, inertia.ixy, inertia.ixz, inertia.iyz))

def convert_to_kdl_joint(jnt):

    '''

    Purpose:

    ---

    Converts a URDF joint object into a PyKDL Joint for kinematic chain construction

    Input Arguments:

    ---

    `jnt` : URDF joint object

    The joint definition including type, axis, and origin

    Returns:

    ---

    `joint` : kdl.Joint

    A PyKDL joint object representing the joint type and properties

    Example call:

    ---

    kdl_joint = _toKdlJoint(urdf_joint)

    '''

    fixed = lambda j,F: kdl.Joint(j.name, kdl.Joint.Fixed)
    rotational = lambda j,F: kdl.Joint(j.name, F.p, F.M * kdl.Vector(*j.axis), kdl.Joint.RotAxis)
    translational = lambda j,F: kdl.Joint(j.name, F.p, F.M * kdl.Vector(*j.axis), kdl.Joint.TransAxis)

    type_map = {
            'fixed': fixed,
            'revolute': rotational,
            'continuous': rotational,
            'prismatic': translational,
            'floating': fixed,
            'planar': fixed,
            'unknown': fixed,
            }

    return type_map[jnt.type](jnt, convert_to_kdl_pose(jnt.origin))

def add_children_to_tree(robot_model, root, tree):

    '''

    Purpose:

    ---

    Recursively adds child links to the KDL tree structure, building the kinematic chain

    Input Arguments:

    ---

    `robot_model` : URDF model object

    The parsed URDF model containing link and joint information

    `root` : URDF link object

    The current root link to add to the tree

    `tree` : kdl.Tree

    The KDL tree being constructed

    Returns:

    ---

    `bool`

    True if the segment was added successfully, False otherwise

    Example call:

    ---

    success = _add_children_to_tree(robot_model, root_link, kdl_tree)

    '''

    # constructs the optional inertia
    inert = kdl.RigidBodyInertia(0)
    if root.inertial:
        inert = convert_to_kdl_inertia(root.inertial)

    # constructs the kdl joint
    (parent_joint_name, parent_link_name) = robot_model.parent_map[root.name]
    parent_joint = robot_model.joint_map[parent_joint_name]

    # construct the kdl segment
    sgm = kdl.Segment(
        root.name,
        convert_to_kdl_joint(parent_joint),
        convert_to_kdl_pose(parent_joint.origin),
        inert)

    # add segment to tree
    if not tree.addSegment(sgm, parent_link_name):
        return False

    if root.name not in robot_model.child_map:
        return True

    children = [robot_model.link_map[l] for (j,l) in robot_model.child_map[root.name]]

    # recurslively add all children
    for child in children:
        if not add_children_to_tree(robot_model, child, tree):
            return False

    return True

def build_tree_from_urdf(robot_model, quiet=False):

    '''

    Purpose:

    ---

    Builds a complete PyKDL kinematic tree from a parsed URDF robot model

    Input Arguments:

    ---

    `robot_model` : URDF model object

    The parsed URDF model containing all robot links and joints

    `quiet` : bool, optional

    If True, suppresses warning messages about root link inertia (default: False)

    Returns:

    ---

    `tuple`

    A tuple containing (success_flag, kdl_tree) where success_flag is True if tree construction succeeded

    Example call:

    ---

    success, tree = treeFromUrdfModel(parsed_urdf_model)

    '''

    root = robot_model.link_map[robot_model.get_root()]

    if root.inertial and not quiet:
        print("The root link %s has an inertia specified in the URDF, but KDL does not support a root link with an inertia.  As a workaround, you can add an extra dummy link to your URDF." % root.name);

    ok = True
    tree = kdl.Tree(root.name)

    #  add all children
    for (joint,child) in robot_model.child_map[root.name]:
        if not add_children_to_tree(robot_model, robot_model.link_map[child], tree):
            ok = False
            break
  
    return (ok, tree)

class CartesianServoNode(Node):
    def __init__(self):

        '''

        Purpose:

        ---

        Initializes the CartesianServoNode with ROS2 publishers, subscribers, TF handling, and robot kinematics setup

        Input Arguments:

        ---

        None

        Returns:

        ---

        None

        Example call:

        ---

        node = CartesianServoNode()

        '''

        super().__init__('cartesian_servo_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.subscription = self.create_subscription(
            Twist,
            '/delta_twist_cmds',
            self.handle_twist_commands,
            10)
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.handle_joint_states,
            10)
        
        # Subscribe to /tf topic
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.handle_tf_messages,
            10
        )
        self.eef_pub = self.create_publisher(PoseStamped, '/tcp_pose_raw', 10)

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10)
        self.publisher_ = self.create_publisher(Float32, '/force_status', 10)
        
        # Manipulation state - auto-enabled (no longer waiting for detection_complete)
        self.manipulation_enabled = True  # Auto-enable manipulation
        self.manipulation_completed = False
        
        # Add marker publisher for plane visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/visualization_marker_array',
            10)
        
        self.joint_delta_sub = self.create_subscription(
            Float64MultiArray,
            '/delta_joint_cmds',
            self.handle_delta_joint_commands,
            10)
        
        self.transforms = {}

        # List of frame IDs we're interested in
        self.target_frames = [
            '1912_fertilizer_1', 
            'aruco_marker_6',
            '1912_bad_fruit_1',
            '1912_bad_fruit_2',
            '1912_bad_fruit_3'
        ]

        self.joint_positions = None
        self.twist = np.zeros(6)  # Initialize twist to zero
        self.state = 1.0
        self.show_planes = True  # Toggle for plane visibility
        
        # Flags for coordinate extraction
        self.coordinates_extracted = False
        self.extracted_coords = {
            '1912_fertilizer_1': None, 
            'aruco_marker_6': None,
            '1912_bad_fruit_1': None,
            '1912_bad_fruit_2': None,
            '1912_bad_fruit_3': None
        }
        
        # Waypoint navigation variables - will be populated after coordinate extraction
        self.waypoints = [
            # Waypoint 1: ArUco marker 3 / fertiliser_can (will be replaced with actual coordinates + attach action)
            {
                'position': [-0.2101, -0.5517, 0.6310],
                'orientation': [0.707, 0.028, 0.034, 0.707],
                'action': 'attach',
                'object_name': 'fertiliser_can'
            },
            # Waypoint 2: intermediate position
            {
                'position': [-0.2101, -0.3517, 0.6610],
                'orientation': [0.707, 0.028, 0.034, 0.707]
            },
            # Waypoint 3: ArUco marker 6 (will be replaced with actual coordinates) - DETACH fertiliser_can here
            {
                'position': [0.7101, 0.0086, 0.354],
                'orientation': [0.707, 0.707, 0.0, 0.033],
                'action': 'detach'
            },
            # Waypoints 4+: Bad fruits will be inserted here dynamically
            #'orientation': [0.029, 0.997, 0.045, 0.033] # to be used for all bad fruit way points.!!!
        ]



        self.current_waypoint_index = 0
        self.waypoint_reached = False
        self.waypoint_stop_start_time = None
        self.position_tolerance = 0.02 # 2.5cm tolerance (slightly relaxed for speed)
        self.orientation_tolerance = 0.12  # tolerance for orientation (slightly relaxed)
        self.stop_duration = 0.2  # 0.5 second stop at each waypoint (reduced from 1.0)
        self.navigation_active = False  # Start disabled, enabled when waypoints are set up
        
        # Can retrieval variables
        self.can_retrieval_active = False
        self.aruco6_position = None  # Store ArUco marker 6 position for can retrieval
        
        # Initialize boundary_plane_data first
        self.WORKSPACE_LIMITS = {
            'x_min': -10.5, 'x_max': 10.5,    # Left/Right walls
            'y_min': -10.5, 'y_max': 10.5,    # Front/Back walls  
            'z_min': -10.0, 'z_max': 10.0     # Floor/Ceiling
        }

        # self.WORKSPACE_LIMITS = {
        #     'x_min': -0.80, 
        #     'x_max': 0.8,    # Left/Right walls
        #     'y_min': -0.80, 'y_max': 0.8,    # Front/Back walls  
        #     'z_min': -0.00, 'z_max': 0.8     # Floor/Ceiling
        # }

        self.boundary_plane_data = []

        # URDF and KDL setup
        package_share_dir = get_package_share_directory('ur_simulation_gz')
        urdf_path = os.path.join(package_share_dir, 'ur5_arm.urdf')
        base_link = 'base_link'                  # Change to your robot base link
        end_link = 'tool0'     
        self.robot = URDF.from_xml_file(urdf_path)             # Change to your end-effector link
        self.kdl_chain = self.setup_kinematic_chain(urdf_path, base_link, end_link)
        self.link_geometries = self.load_collision_shapes()
        self.link_names = list(self.link_geometries.keys())
        # note: right now calculate_twist function takes time: 0.0030 seconds,
        #                          so here we set the timer to 0.0050 seconds 
        # so that collision check could occur at fast as possible

        self.create_timer(0.005, self.calculate_twist)
        self.joint_velocities = np.zeros(6)  # Initialize joint velocities to zero
        # Initialize boundary planes AFTER boundary_plane_data is initialized
        self.boundary_planes = self.create_boundary_planes()
        
        # Add this: Create robot collision objects once
        self.robot_collision_objects = {}
        self.create_robot_collision_objects()
        
        # self.publish_plane_markers()
        # Timer for publishing markers
        self.create_timer(1.0, self.publish_plane_markers)
        
        # Create service clients for link attacher - COMMENTED OUT AS PER REQUIREMENTS
        # self.attach_client = self.create_client(AttachLink, '/attach_link')
        # self.detach_client = self.create_client(DetachLink, '/detach_link')
        
        # Wait for services to be available
        # while not self.attach_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for /attach_link service...')
        # while not self.detach_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for /detach_link service...')
        
        # self.get_logger().info('Link attacher services are ready!')
        
        # Create service client for electromagnet control
        self.magnet_client = self.create_client(SetBool, '/magnet')
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Magnet service not available, waiting...')
        self.get_logger().info('Magnet service is ready!')
        
        # Subscribe to net force monitoring
        self.net_wrench_sub = self.create_subscription(
            Float32,
            '/net_wrench',
            self.net_wrench_callback,
            10)
        self.net_force = 0.0  # Store current net force
        
        # Create publishers for velocity control
        self.joint_vel_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.twist_vel_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        
        # Flag to track if object is attached
        self.object_attached = False
        self.attached_object_name = None  # Track which object is currently attached

    def setup_can_retrieval_waypoints(self):
        '''
        Purpose:
        ---
        Sets up waypoints to retrieve the fertilizer can and place it in trash bin

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''
        # Clear existing waypoints
        self.waypoints = []
        
        # Waypoint 1: Go to ArUco marker 6 (where can was dropped) - above position
        aruco6_above = self.aruco6_position.copy()
        aruco6_above[2] += 0.16  # 16cm above the can
        self.waypoints.append({
            'position': aruco6_above,
            'orientation': [0.707, 0.707, 0.0, 0.033],
            'action': 'approach'
        })
        
        # Waypoint 2: Pick up the can at ArUco marker 6
        self.waypoints.append({
            'position': self.aruco6_position.copy(),
            'orientation': [0.707, 0.707, 0.0, 0.033],
            'action': 'attach',
            'object_name': 'fertiliser_can'
        })
        
        # Waypoint 3: Lift up after picking
        aruco6_lift = self.aruco6_position.copy()
        aruco6_lift[2] += 0.16
        self.waypoints.append({
            'position': aruco6_lift,
            'orientation': [0.707, 0.707, 0.0, 0.033],
            'action': 'retreat'
        })
        
        # Waypoint 4: Go to trash bin
        self.waypoints.append({
            'position': [-0.806, 0.010, 0.382],
            'orientation': [-0.384, 0.726, 0.05, 0.008],
            'action': 'detach'
        })
        
        self.get_logger().info(f'✓ Created {len(self.waypoints)} waypoints for can retrieval')
        self.display_all_waypoints()
    
    def net_wrench_callback(self, msg):
        '''
        Purpose:
        ---
        Monitors the net force on the end-effector for safer grasping operations

        Input Arguments:
        ---
        msg : Float32
            Net force/wrench magnitude

        Returns:
        ---
        None
        '''
        self.net_force = msg.data
        # Log high forces for safety monitoring
        if self.net_force > 10.0:  # Threshold for warning
            self.get_logger().warn(f'High net force detected: {self.net_force:.2f} N')
    
    def control_magnet(self, state):
        '''
        Purpose:
        ---
        Controls the electromagnet to attach/detach objects

        Input Arguments:
        ---
        state : bool
            True to activate magnet, False to deactivate

        Returns:
        ---
        Future
            Service call future object
        '''
        request = SetBool.Request()
        request.data = state
        action = "Activating" if state else "Deactivating"
        self.get_logger().info(f'{action} electromagnet...')
        future = self.magnet_client.call_async(request)
        future.add_done_callback(lambda f: self.magnet_response_callback(f, state))
        return future
    
    def magnet_response_callback(self, future, state):
        '''
        Purpose:
        ---
        Callback to handle the response from the magnet service

        Input Arguments:
        ---
        future : Future
            The service response future
        state : bool
            The requested state (for logging)

        Returns:
        ---
        None
        '''
        try:
            response = future.result()
            if response.success:
                action = "activated" if state else "deactivated"
                self.get_logger().info(f'✓ Electromagnet {action} successfully!')
                self.object_attached = state
            else:
                self.get_logger().error(f'✗ Magnet control failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Magnet service call failed: {e}')

    def handle_tf_messages(self, msg):

        '''

        Purpose:

        ---

        Processes incoming TF (transform) messages to store coordinate transforms for target frames

        Input Arguments:

        ---

        `msg` : TFMessage

        The ROS2 TF message containing transform data from the /tf topic

        Returns:

        ---

        None

        Example call:

        ---

        Automatically called by ROS2 when TF messages are received

        '''

        for transform in msg.transforms:
            child_frame = transform.child_frame_id
            parent_frame = transform.header.frame_id
            
            # Only store transforms we're interested in that are relative to base_link
            if child_frame in self.target_frames and parent_frame == 'base_link':
                self.transforms[child_frame] = {
                    'parent': parent_frame,
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z,
                    'qx': transform.transform.rotation.x,
                    'qy': transform.transform.rotation.y,
                    'qz': transform.transform.rotation.z,
                    'qw': transform.transform.rotation.w,
                    'timestamp': transform.header.stamp
                }
        
        # Extract coordinates once when all frames are available
        if not self.coordinates_extracted and len(self.transforms) == len(self.target_frames):
            self.setup_waypoints_from_tf()

    def handle_delta_joint_commands(self, msg):

        '''

        Purpose:

        ---

        Callback to handle delta joint commands and forward them to the velocity controller

        Input Arguments:

        ---

        `msg` : Float64MultiArray

        The message containing delta joint velocities to be applied

        Returns:

        ---

        None

        Example call:

        ---

        Automatically called by ROS2 when delta joint command messages are received

        '''

        if msg.data is None:
            return
        msg2 = Float64MultiArray()
        msg2.data = list(msg.data)
        self.cmd_pub.publish(msg2)

    def convert_quaternion_to_euler(self, qx, qy, qz, qw):

        '''

        Purpose:

        ---

        Converts quaternion orientation to Euler angles (roll, pitch, yaw) in degrees for easier human understanding

        Input Arguments:

        ---

        `qx` : float

        X component of quaternion

        `qy` : float

        Y component of quaternion

        `qz` : float

        Z component of quaternion

        `qw` : float

        W component of quaternion

        Returns:

        ---

        `tuple`

        (roll, pitch, yaw) angles in degrees

        Example call:

        ---

        roll, pitch, yaw = self.convert_quaternion_to_euler(0.0, 0.0, 0.0, 1.0)

        '''

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    
    def setup_waypoints_from_tf(self):

        '''

        Purpose:

        ---

        Extracts coordinates from TF frames once all target frames are available and dynamically generates waypoints for fertilizer can and bad fruits

        Input Arguments:

        ---

        None

        Returns:

        ---

        None

        Example call:

        ---

        Automatically called when all TF frames are received

        '''

        if self.coordinates_extracted:
            return
        
        self.get_logger().info('🎯 Extracting coordinates from TF frames...')
        
        # Store extracted coordinates
        for frame_id in self.target_frames:
            if frame_id in self.transforms:
                tf = self.transforms[frame_id]
                self.extracted_coords[frame_id] = {
                    'position': [tf['x'], tf['y'], tf['z']],
                    'orientation': [tf['qx'], tf['qy'], tf['qz'], tf['qw']]
                }
                self.get_logger().info(f'✓ Extracted {frame_id}: pos=[{tf["x"]:.4f}, {tf["y"]:.4f}, {tf["z"]:.4f}]')
        
        # Update waypoint 1 (index 0) with 1912_fertilizer_1 coordinates - ONLY POSITION
        # This waypoint has 'attach' action already set for fertiliser_can
        if self.extracted_coords['1912_fertilizer_1'] is not None:
            self.waypoints[0]['position'] = self.extracted_coords['1912_fertilizer_1']['position'].copy()
            # Keep the original orientation: [0.707, 0.028, 0.034, 0.707]
            self.get_logger().info('✓ Updated waypoint 1 with 1912_fertilizer_1 position (keeping original orientation)')
        
        # Update waypoint 3 (index 2) with aruco_marker_6 coordinates - ONLY POSITION, ADD +0.15 to Z
        if self.extracted_coords['aruco_marker_6'] is not None:
            aruco6_pos = self.extracted_coords['aruco_marker_6']['position'].copy()
            aruco6_pos[2] += 0.16 # Add +0.15 to Z value
            self.waypoints[2]['position'] = aruco6_pos
            # Keep the original orientation: [0.707, 0.707, 0.0, 0.033]
            
            # Store the original ArUco marker 6 position for later can retrieval
            self.aruco6_position = self.extracted_coords['aruco_marker_6']['position'].copy()
            
            self.get_logger().info(f'✓ Updated waypoint 3 with aruco_marker_6 position (Z+0.15) (keeping original orientation)')
            self.get_logger().info(f'✓ Stored ArUco marker 6 position for can retrieval: {self.aruco6_position}')
        
        # Create waypoints for bad fruits (insert after waypoint 3)
        # Final waypoint (trash bin)
        trash_bin_waypoint = {
            'position': [-0.806, 0.010, 0.382],
            'orientation': [-0.384, 0.726, 0.05, 0.008],
            'action': 'detach'  # Mark as detach point
        }
        
        # Orientation to be used for ALL bad fruit waypoints
        bad_fruit_orientation = [0.029, 0.997, 0.045, 0.033]
        
        # Build bad fruit waypoints
        bad_fruit_waypoints = []
        for i, fruit_frame in enumerate(['1912_bad_fruit_1', '1912_bad_fruit_2', '1912_bad_fruit_3'], 1):
            if self.extracted_coords[fruit_frame] is not None:
                fruit_pos = self.extracted_coords[fruit_frame]['position'].copy()
                
                # Pseudo waypoint BEFORE (z + 0.2) - use specified orientation
                pseudo_before = {
                    'position': [fruit_pos[0], fruit_pos[1], fruit_pos[2] + 0.15],
                    'orientation': bad_fruit_orientation.copy(),
                    'action': 'approach'
                }
                
                # Actual waypoint (pickup location) - use specified orientation
                actual_waypoint = {
                    'position': fruit_pos.copy(),
                    'orientation': bad_fruit_orientation.copy(),
                    'action': 'attach',
                    'object_name': f'bad_fruit'
                }
                
                # Pseudo waypoint AFTER (z + 0.2) - use specified orientation
                pseudo_after = {
                    'position': [fruit_pos[0], fruit_pos[1], fruit_pos[2] + 0.16],
                    'orientation': bad_fruit_orientation.copy(),
                    'action': 'retreat'
                }
                
                # Add all three waypoints for this fruit
                bad_fruit_waypoints.extend([pseudo_before, actual_waypoint, pseudo_after])
                
                # Add trash bin waypoint after each fruit sequence
                bad_fruit_waypoints.append(trash_bin_waypoint.copy())
                
                self.get_logger().info(f'✓ Created waypoint sequence for {fruit_frame}')
        
        # Insert bad fruit waypoints after waypoint 3 (index 2)
        self.waypoints = self.waypoints[:3] + bad_fruit_waypoints
        
        self.coordinates_extracted = True
        self.get_logger().info(f'🎉 Waypoints updated! Total waypoints: {len(self.waypoints)}')
        self.display_all_waypoints()
    
    def display_all_waypoints(self):

        '''

        Purpose:

        ---

        Prints a formatted list of all waypoints with their positions and actions for debugging purposes

        Input Arguments:

        ---

        None

        Returns:

        ---

        None

        Example call:

        ---

        self.display_all_waypoints()

        '''

        self.get_logger().info('\n' + '='*80)
        self.get_logger().info('📋 COMPLETE WAYPOINT LIST')
        self.get_logger().info('='*80)
        for i, wp in enumerate(self.waypoints):
            pos = wp['position']
            action = wp.get('action', 'navigate')
            self.get_logger().info(f'Waypoint {i}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], action={action}')
        self.get_logger().info('='*80 + '\n')
    
    def display_coordinates(self):

        '''

        Purpose:

        ---

        Prints all stored TF transform coordinates in a formatted table for debugging and verification

        Input Arguments:

        ---

        None

        Returns:

        ---

        None

        Example call:

        ---

        self.display_coordinates()

        '''

        if not self.transforms:
            self.get_logger().info("No transforms received yet...")
            return
        
        print("\n" + "="*80)
        print("TRANSFORM COORDINATES (relative to base_link)")
        print("="*80)
        
        # Sort by frame name for consistent output
        for frame_id in sorted(self.transforms.keys()):
            tf = self.transforms[frame_id]
            
            # Get Euler angles from quaternion
            roll, pitch, yaw = self.convert_quaternion_to_euler(
                tf['qx'], tf['qy'], tf['qz'], tf['qw']
            )
            
            print(f"\n📍 {frame_id}")
            print(f"   Parent Frame: {tf['parent']}")
            print(f"   Position (x, y, z): ({tf['x']:.4f}, {tf['y']:.4f}, {tf['z']:.4f}) meters")
            print(f"   Quaternion (x, y, z, w): ({tf['qx']:.4f}, {tf['qy']:.4f}, {tf['qz']:.4f}, {tf['qw']:.4f})")
            print(f"   Euler Angles (roll, pitch, yaw): ({roll:.2f}°, {pitch:.2f}°, {yaw:.2f}°)")
            
            # Calculate distance from base_link
            distance = math.sqrt(tf['x']**2 + tf['y']**2 + tf['z']**2)
            print(f"   Distance from base_link: {distance:.4f} meters")
        
        print("\n" + "="*80)
        print(f"Total transforms tracked: {len(self.transforms)}")
        print("="*80 + "\n")

    def load_collision_shapes(self):

        '''

        Purpose:

        ---

        Extracts collision geometries from the URDF robot model for collision detection setup

        Input Arguments:

        ---

        None

        Returns:

        ---

        `dict`

        Dictionary mapping link names to their collision geometries

        Example call:

        ---

        geometries = self.load_collision_shapes()

        '''

        geoms = {}
        for link in self.robot.links:
            if link.collision and link.collision.geometry:
                geoms[link.name] = link.collision.geometry
        return geoms

    def setup_kinematic_chain(self, urdf_path, base_link, end_link):

        '''

        Purpose:

        ---

        Creates a PyKDL kinematic chain from URDF model for forward and inverse kinematics calculations

        Input Arguments:

        ---

        `urdf_path` : str

        Path to the URDF file

        `base_link` : str

        Name of the base link in the chain

        `end_link` : str

        Name of the end-effector link in the chain

        Returns:

        ---

        `kdl.Chain`

        PyKDL kinematic chain object

        Example call:

        ---

        chain = self.setup_kinematic_chain('/path/to/ur5.urdf', 'base_link', 'tool0')

        '''

        success, kdl_tree = build_tree_from_urdf(self.robot)
        if not success:
            raise RuntimeError("Failed to parse URDF into KDL tree.")
        chain = kdl_tree.getChain(base_link, end_link)
        return chain


    def calculate_jacobian_matrix(self, chain, joint_positions):

        '''

        Purpose:

        ---

        Computes the Jacobian matrix for the kinematic chain at given joint positions for velocity control

        Input Arguments:

        ---

        `chain` : kdl.Chain

        The kinematic chain

        `joint_positions` : list

        Current joint positions

        Returns:

        ---

        `kdl.Jacobian`

        The 6xN Jacobian matrix

        Example call:

        ---

        jacobian = self.compute_jacobian(self.kdl_chain, self.joint_positions)

        '''

        nj = chain.getNrOfJoints()
        joint_array = kdl.JntArray(nj)
        for i in range(nj):
            joint_array[i] = joint_positions[i]

        jacobian = kdl.Jacobian(nj)
        solver = kdl.ChainJntToJacSolver(chain)
        solver.JntToJac(joint_array, jacobian)
        return jacobian


    def calculate_damped_pseudoinverse(self, J, damping=0.01):

        '''

        Purpose:

        ---

        Computes the damped pseudoinverse of the Jacobian matrix for singularity-robust inverse kinematics

        Input Arguments:

        ---

        `J` : numpy.ndarray

        The Jacobian matrix

        `damping` : float, optional

        Damping factor to avoid singularities (default: 0.01)

        Returns:

        ---

        `numpy.ndarray`

        The damped pseudoinverse matrix

        Example call:

        ---

        J_pinv = self.calculate_damped_pseudoinverse(jacobian_matrix, damping=0.01)

        '''

        JT = J.T
        identity = np.eye(J.shape[0])
        return JT @ np.linalg.inv(J @ JT + damping**2 * identity)


    def calculate_joint_velocities(self, jacobian, twist_cmd):

        '''

        Purpose:

        ---

        Computes joint velocities from Cartesian twist command using inverse Jacobian

        Input Arguments:

        ---

        `jacobian` : kdl.Jacobian

        The Jacobian matrix

        `twist_cmd` : numpy.ndarray

        Desired Cartesian twist [vx, vy, vz, wx, wy, wz]

        Returns:

        ---

        `numpy.ndarray` or None

        Joint velocities array, or None if singularity detected

        Example call:

        ---

        joint_vels = self.calculate_joint_velocities(jacobian, twist_command)

        '''

        # Convert KDL Jacobian to numpy array
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(6)])
        singularity_state = self.detect_singularity(J)
        # print("Singularity State:", singularity_state)
        # if singularity_state == 3.0:
        #     return None  # signal to halt
        # elif singularity_state == 2.0:
        #     twist_cmd = twist_cmd * (1*10**-3)
        # # elif isinstance(singularity_state, float) and singularity_state < 1.0:
        # #     twist_cmd = twist_cmd * singularity_state  # scale twist
        # else:
        twist_cmd = twist_cmd
        J_pinv = self.calculate_damped_pseudoinverse(J)
        return J_pinv @ twist_cmd



    def detect_singularity(self, J, hard_stop_threshold=75.0, lower_threshold=50.0):

        '''

        Purpose:

        ---

        Checks if the Jacobian matrix indicates a kinematic singularity and returns appropriate scaling factor

        Input Arguments:

        ---

        `J` : numpy.ndarray

        The Jacobian matrix

        `hard_stop_threshold` : float, optional

        Condition number threshold for hard stop (default: 75.0)

        `lower_threshold` : float, optional

        Condition number threshold for scaling (default: 50.0)

        Returns:

        ---

        `float`

        3.0 for hard stop, 2.0 for scaling, 1.0 for normal operation

        Example call:

        ---

        state = self.detect_singularity(jacobian_matrix)

        '''

        _, singular_vals, _ = np.linalg.svd(J)
        cond_number = np.max(singular_vals) / np.min(singular_vals)
        # print(f"Condition number: {cond_number:.6f}")
        if cond_number > hard_stop_threshold:
            self.state = 3.0  # signal to halt
            return 3.0
        elif cond_number > lower_threshold:
            # scale down
            self.state = 2.0
            return 2.0  # scale factor (0 < scale < 1)
        else:
            # normal operation
            self.state = 1.0
            return 1.0  # normal operation
    def handle_joint_states(self, msg):

        '''

        Purpose:

        ---

        Callback to receive and store current joint positions from the robot

        Input Arguments:

        ---

        `msg` : JointState

        ROS2 JointState message containing joint positions

        Returns:

        ---

        None

        Example call:

        ---

        Automatically called when joint state messages are received

        '''

        self.joint_positions = msg.position
        # print("Joint Positions:", self.joint_positions)
        # print("Joint Names:", msg)

    def handle_twist_commands(self, msg):

        '''

        Purpose:

        ---

        Callback to receive Cartesian twist commands for robot motion control

        Input Arguments:

        ---

        `msg` : Twist

        ROS2 Twist message with linear and angular velocities

        Returns:

        ---

        None

        Example call:

        ---

        Automatically called when twist command messages are received

        '''

        self.twist = np.array([msg.linear.x, msg.linear.y, msg.linear.z,
                          msg.angular.x, msg.angular.y, msg.angular.z])
        
    def stop_robot(self):

        '''

        Purpose:

        ---

        Immediately stops the robot by setting joint velocities to zero and publishing halt status

        Input Arguments:

        ---

        None

        Returns:

        ---

        None

        Example call:

        ---

        self.stop_robot()

        '''

        self.joint_velocities = np.zeros(len(self.joint_positions))
        vel_msg = Float64MultiArray()
        vel_msg.data = self.joint_velocities.tolist()
        self.cmd_pub.publish(vel_msg)
        publish_force_status = Float32()
        publish_force_status.data = self.state
        self.publisher_.publish(publish_force_status)

    def get_current_end_effector_pose(self):

        '''

        Purpose:

        ---

        Gets the current end-effector pose using forward kinematics from TF transforms

        Input Arguments:

        ---

        None

        Returns:

        ---

        `dict` or None

        Dictionary with 'position' and 'orientation' keys, or None if transform lookup fails

        Example call:

        ---

        current_pose = self.get_current_end_effector_pose()

        '''

        try:
            # Get transform from base_link to tool0 (end-effector)
            transform = self.get_link_pose('tool0', 'base_link', tf=True)
            if transform is None:
                return None
            
            translation, rotation = transform
            return {
                'position': translation,
                'orientation': rotation  # [x, y, z, w] quaternion
            }
        except Exception as e:
            self.get_logger().warn(f'Failed to get end-effector pose: {str(e)}')
            return None

    def calculate_position_error(self, current_pos, target_pos):

        '''

        Purpose:

        ---

        Calculates the Euclidean distance between current and target positions

        Input Arguments:

        ---

        `current_pos` : list

        Current position [x, y, z]

        `target_pos` : list

        Target position [x, y, z]

        Returns:

        ---

        `float`

        Euclidean distance between positions

        Example call:

        ---

        error = self.calculate_position_error([0, 0, 0], [1, 1, 1])

        '''

        return np.linalg.norm(np.array(current_pos) - np.array(target_pos))

    def calculate_orientation_error(self, current_quat, target_quat):

        '''

        Purpose:

        ---

        Calculates the orientation error between two quaternions as an angle in radians

        Input Arguments:

        ---

        `current_quat` : list

        Current orientation quaternion [x, y, z, w]

        `target_quat` : list

        Target orientation quaternion [x, y, z, w]

        Returns:

        ---

        `float`

        Angular error in radians

        Example call:

        ---

        orient_error = self.calculate_orientation_error([0, 0, 0, 1], [0, 0, 0.1, 0.995])

        '''

        # Normalize quaternions
        current_quat = np.array(current_quat) / np.linalg.norm(current_quat)
        target_quat = np.array(target_quat) / np.linalg.norm(target_quat)
        
        # Calculate the dot product
        dot_product = np.abs(np.dot(current_quat, target_quat))
        
        # Clamp to avoid numerical issues
        dot_product = np.clip(dot_product, 0.0, 1.0)
        
        # Calculate the angle difference
        angle_error = 2 * np.arccos(dot_product)
        return angle_error

    def generate_twist_to_target(self, current_pose, target_waypoint):

        '''

        Purpose:

        ---

        Generates a Cartesian twist command to move from current pose towards target waypoint

        Input Arguments:

        ---

        `current_pose` : dict

        Current end-effector pose with 'position' and 'orientation' keys

        `target_waypoint` : dict

        Target waypoint with 'position' and 'orientation' keys

        Returns:

        ---

        `numpy.ndarray`

        6D twist command [vx, vy, vz, wx, wy, wz]

        Example call:

        ---

        twist = self.generate_twist_to_target(current_pose, waypoint)

        '''

        current_pos = np.array(current_pose['position'])
        target_pos = np.array(target_waypoint['position'])
        
        # Position error
        pos_error = target_pos - current_pos
        
        # Orientation error - convert quaternions to angular velocity
        current_quat = np.array(current_pose['orientation'])  # [x,y,z,w]
        target_quat = np.array(target_waypoint['orientation'])  # [x,y,z,w]
        
        # Calculate rotation needed
        current_rot = Rotation.from_quat(current_quat)
        target_rot = Rotation.from_quat(target_quat)
        
        # Relative rotation from current to target
        relative_rot = target_rot * current_rot.inv()
        
        # Convert to axis-angle representation for angular velocity
        rotvec = relative_rot.as_rotvec()
        
        # Scale gains for position and orientation (INCREASED for faster movement)
        linear_gain = 2.0  # Increased from 1.0 to 2.0
        angular_gain = 2.0  # Increased from 1.0 to 2.0
        
        # Create twist command
        twist = np.zeros(6)
        twist[0:3] = linear_gain * pos_error  # Linear velocity
        twist[3:6] = angular_gain * rotvec     # Angular velocity
        
        # Limit maximum velocities (INCREASED for faster movement)
        max_linear_vel = 0.8 # Increased from 0.2 to 0.5 m/s
        max_angular_vel = 1.5 # Increased from 0.5 to 1.0 rad/s
        
        linear_vel_norm = np.linalg.norm(twist[0:3])
        if linear_vel_norm > max_linear_vel:
            twist[0:3] = twist[0:3] * (max_linear_vel / linear_vel_norm)
            
        angular_vel_norm = np.linalg.norm(twist[3:6])
        if angular_vel_norm > max_angular_vel:
            twist[3:6] = twist[3:6] * (max_angular_vel / angular_vel_norm)
        
        return twist

    def navigate_to_waypoints(self, current_pose):

        '''

        Purpose:

        ---

        Main waypoint navigation logic that moves the robot through predefined waypoints with actions

        Input Arguments:

        ---

        `current_pose` : dict

        Current end-effector pose with 'position' and 'orientation' keys

        Returns:

        ---

        None

        Example call:

        ---

        self.navigate_to_waypoints(current_pose)

        '''

        if self.current_waypoint_index >= len(self.waypoints):
            # All waypoints reached
            if self.navigation_active:
                self.get_logger().info('All waypoints reached! Navigation complete.')
                self.navigation_active = False
                self.twist = np.zeros(6)  # Stop the robot
            return
        
        current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Calculate errors
        pos_error = self.calculate_position_error(
            current_pose['position'], 
            current_waypoint['position']
        )
        orient_error = self.calculate_orientation_error(
            current_pose['orientation'], 
            current_waypoint['orientation']
        )
        
        # Check if waypoint is reached
        if pos_error < self.position_tolerance and orient_error < self.orientation_tolerance:
            if not self.waypoint_reached:
                # Just reached the waypoint
                self.waypoint_reached = True
                self.waypoint_stop_start_time = time.time()
                self.twist = np.zeros(6)  # Stop the robot
                self.get_logger().info(f'Reached waypoint {self.current_waypoint_index + 1}. Stopping for {self.stop_duration} seconds.')
                
                # Handle actions based on waypoint action field
                action = current_waypoint.get('action', None)
                
                if action == 'attach':
                    # Attach bad fruit using electromagnet
                    object_name = current_waypoint.get('object_name', 'bad_fruit')
                    self.get_logger().info(f'At waypoint {self.current_waypoint_index + 1} - Attempting to attach {object_name}...')
                    # self.attach_object(object_name, 'body')  # Commented out - using magnet instead
                    self.control_magnet(True)  # Activate electromagnet
                    self.attached_object_name = object_name
                
                elif action == 'detach':
                    # Detach at trash bin or ArUco marker 6 using electromagnet
                    if self.object_attached and self.attached_object_name:
                        self.get_logger().info(f'At waypoint {self.current_waypoint_index + 1} - Attempting to detach {self.attached_object_name}...')
                        # self.detach_object(self.attached_object_name, 'body')  # Commented out - using magnet instead
                        self.control_magnet(False)  # Deactivate electromagnet
                        self.attached_object_name = None
                        
                        # Log completion when object is placed
                        waypoint_name = current_waypoint.get('name', '')
                        if 'aruco_marker_6' in waypoint_name or self.current_waypoint_index == 2:
                            if not self.manipulation_completed:
                                self.manipulation_completed = True
                                self.get_logger().info('✓ Fertilizer can placed at ArUco marker 6. Manipulation complete!')
                
            elif time.time() - self.waypoint_stop_start_time >= self.stop_duration:
                # Stop duration completed, move to next waypoint
                self.current_waypoint_index += 1
                self.waypoint_reached = False
                self.waypoint_stop_start_time = None
                
                if self.current_waypoint_index < len(self.waypoints):
                    self.get_logger().info(f'Moving to waypoint {self.current_waypoint_index + 1}')
                else:
                    self.get_logger().info('All waypoints completed!')
            else:
                # Still in stop duration
                self.twist = np.zeros(6)
        else:
            # Move towards current waypoint
            if not self.waypoint_reached:
                self.twist = self.generate_twist_to_target(current_pose, current_waypoint)
                
                # Log progress occasionally
                if hasattr(self, '_last_log_time'):
                    if time.time() - self._last_log_time > 2.0:  # Log every 2 seconds
                        self.get_logger().info(
                            f'Moving to waypoint {self.current_waypoint_index + 1}: '
                            f'pos_error={pos_error:.3f}m, orient_error={orient_error:.3f}rad'
                        )
                        self._last_log_time = time.time()
                else:
                    self._last_log_time = time.time()


    def calculate_twist(self):

        '''

        Purpose:

        ---

        Main control loop that computes joint velocities from twist commands, performs waypoint navigation, and handles collision checking

        Input Arguments:

        ---

        None

        Returns:

        ---

        None

        Example call:

        ---

        Called periodically by ROS2 timer to update robot control

        '''

        if self.joint_positions is None:
            self.get_logger().info('Waiting for joint states...')
            return

        # Get current end-effector pose
        current_pose = self.get_current_end_effector_pose()
        if current_pose is None:
            return

        # Perform waypoint navigation
        if self.navigation_active:
            self.navigate_to_waypoints(current_pose)

        # start_time = time.time()
        if self.testing():
            pass
            # self.get_logger().warn("⚠️ Predicted collision detected. Halting motion.")
        #     self.halting()
        #     print(self.joint_velocities)
        #     return
        
        jacobian = self.calculate_jacobian_matrix(self.kdl_chain, self.joint_positions)
        self.joint_velocities = self.calculate_joint_velocities(jacobian, self.twist)
        publish_force_status = Float32()
        publish_force_status.data = self.state
        self.publisher_.publish(publish_force_status)
        
        # if self.joint_velocities is None:
        #     self.get_logger().warn("⚠️ Near singularity detected. Halting motion.")
        #     self.stop_robot()
        #     return
        
        vel_msg = Float64MultiArray()
        vel_msg.data = self.joint_velocities.tolist()
        self.cmd_pub.publish(vel_msg)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print("Elapsed time", elapsed_time)

    # COMMENTED OUT - Using electromagnet control instead of link attacher
    # def attach_object(self, model1_name, link1_name, model2_name='ur5', link2_name='wrist_3_link'):

    #     '''

    #     Purpose:

    #     ---

    #     Attaches an object to the robot gripper using the link attacher service

    #     Input Arguments:

    #     ---

    #     `model1_name` : str

    #     Name of the object model to attach

    #     `link1_name` : str

    #     Name of the link on the object

    #     `model2_name` : str, optional

    #     Name of the robot model (default: 'ur5')

    #     `link2_name` : str, optional

    #     Name of the robot link to attach to (default: 'wrist_3_link')

    #     Returns:

    #     ---

    #     None

    #     Example call:

    #     ---

    #     self.attach_object('bad_fruit', 'body', 'ur5', 'wrist_3_link')

    #     '''

    #     request = AttachLink.Request()
    #     request.model1_name = model1_name
    #     request.link1_name = link1_name
    #     request.model2_name = model2_name
    #     request.link2_name = link2_name
        
    #     self.get_logger().info(f'Attaching {model1_name}/{link1_name} to {model2_name}/{link2_name}...')
    #     future = self.attach_client.call_async(request)
    #     future.add_done_callback(self.attach_response_callback)
    
    # def attach_response_callback(self, future):

    #     '''

    #     Purpose:

    #     ---

    #     Callback to handle the response from the attach link service

    #     Input Arguments:

    #     ---

    #     `future` : Future

    #     The service response future

    #     Returns:

    #     ---

    #     None

    #     Example call:

    #     ---

    #     Automatically called when attach service responds

    #     '''

    #     try:
    #         response = future.result()
    #         if response.success:
    #             self.get_logger().info('✓ Object attached successfully!')
    #             self.object_attached = True
    #         else:
    #             self.get_logger().error(f'✗ Attach failed: {response.message}')
    #     except Exception as e:
    #         self.get_logger().error(f'Service call failed: {e}')
    
    # def detach_object(self, model1_name, link1_name, model2_name='ur5', link2_name='wrist_3_link'):

    #     '''

    #     Purpose:

    #     ---

    #     Detaches an object from the robot gripper using the link attacher service

    #     Input Arguments:

    #     ---

    #     `model1_name` : str

    #     Name of the object model to detach

    #     `link1_name` : str

    #     Name of the link on the object

    #     `model2_name` : str, optional

    #     Name of the robot model (default: 'ur5')

    #     `link2_name` : str, optional

    #     Name of the robot link to detach from (default: 'wrist_3_link')

    #     Returns:

    #     ---

    #     None

    #     Example call:

    #     ---

    #     self.detach_object('bad_fruit', 'body', 'ur5', 'wrist_3_link')

    #     '''

    #     request = DetachLink.Request()
    #     request.model1_name = model1_name
    #     request.link1_name = link1_name
    #     request.model2_name = model2_name
    #     request.link2_name = link2_name
        
    #     self.get_logger().info(f'Detaching {model1_name}/{link1_name} from {model2_name}/{link2_name}...')
    #     future = self.detach_client.call_async(request)
    #     future.add_done_callback(self.detach_response_callback)
    
    # def detach_response_callback(self, future):

    #     '''

    #     Purpose:

    #     ---

    #     Callback to handle the response from the detach link service

    #     Input Arguments:

    #     ---

    #     `future` : Future

    #     The service response future

    #     Returns:

    #     ---

    #     None

    #     Example call:

    #     ---

    #     Automatically called when detach service responds

    #     '''

    #     try:
    #         response = future.result()
    #         if response.success:
    #             self.get_logger().info('✓ Object detached successfully!')
    #             self.object_attached = False
    #         else:
    #             self.get_logger().error(f'✗ Detach failed: {response.message}')
    #     except Exception as e:
    #         self.get_logger().error(f'Service call failed: {e}')

    
    def resolve_package_uri(self, uri):

        '''

        Purpose:

        ---

        Resolves package:// URIs to absolute file paths for mesh loading

        Input Arguments:

        ---

        `uri` : str

        The package URI to resolve

        Returns:

        ---

        `str`

        Absolute file path or original URI if not a package URI

        Example call:

        ---

        mesh_path = self.resolve_package_uri('package://pymoveit2/ur5/meshes/link.stl')

        '''

        if uri.startswith("package://"):
            package_name, rel_path = uri[len("package://"):].split("/", 1)
            package_path = get_package_share_directory(package_name)
            return os.path.join(package_path, rel_path)
        return uri

    def get_link_pose(self, target_frame, source_frame='base_link' ,tf=False):

        '''

        Purpose:

        ---

        Gets the transform between two frames using TF buffer

        Input Arguments:

        ---

        `target_frame` : str

        Name of the target frame

        `source_frame` : str, optional

        Name of the source frame (default: 'base_link')

        `tf` : bool, optional

        If True, return translation and rotation separately (default: False)

        Returns:

        ---

        `numpy.ndarray` or tuple or None

        4x4 transformation matrix, or (translation, rotation) tuple, or None if lookup fails

        Example call:

        ---

        transform = self.get_link_pose('tool0', 'base_link')

        '''

        try:
            # now = self.get_clock().now()
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                source_frame, target_frame, rclpy.time.Time())
            translation = [trans.transform.translation.x,
                        trans.transform.translation.y,
                        trans.transform.translation.z]
            rotation = [trans.transform.rotation.x,
                        trans.transform.rotation.y,
                        trans.transform.rotation.z,
                        trans.transform.rotation.w]
            # Convert quaternion to rotation matrix using scipy
            if tf:
                return translation, rotation
            r = Rotation.from_quat(rotation)
            matrix = np.eye(4)
            matrix[:3, :3] = r.as_matrix()
            matrix[0:3, 3] = translation
            return matrix
        except Exception as e:
            self.get_logger().warn(f'Failed to get transform for {target_frame}: {str(e)}')
            return None

    def create_fcl_object(self, geometry, pose_matrix):
        """
        Purpose:
        ---
        Creates an FCL collision object from URDF geometry and pose matrix

        Input Arguments:
        ---
        `geometry` : urdf_parser_py.urdf.Mesh or other geometry type
            The geometry specification from URDF
        `pose_matrix` : numpy.ndarray
            4x4 transformation matrix representing the pose

        Returns:
        ---
        `fcl.CollisionObject` or None
            FCL collision object if geometry is a mesh, None otherwise

        Example call:
        ---
        collision_obj = self.create_fcl_object(mesh_geometry, pose_matrix)
        """
        if isinstance(geometry, urdf_parser_py.urdf.Mesh):
            mesh_path = self.resolve_package_uri(geometry.filename)
            print(f"Loading mesh from: {mesh_path}")
            mesh = trimesh.load_mesh(mesh_path)
            vertices = mesh.vertices
            triangles = mesh.faces
            model = fcl.BVHModel()
            model.beginModel(len(vertices), len(triangles))
            model.addSubModel(vertices, triangles)
            model.endModel()
            transform = fcl.Transform(pose_matrix[0:3, 0:3], pose_matrix[0:3, 3])
            return fcl.CollisionObject(model, transform)
        return None
            
    def create_robot_collision_objects(self):
        """Create FCL collision objects for robot links once during initialization"""
       
        for link_name in self.link_geometries:
            geometry = self.link_geometries[link_name]
            # Create the collision object with identity transform initially
            identity_pose = np.eye(4)
            obj = self.create_fcl_object(geometry, identity_pose)
            if obj:
                self.robot_collision_objects[link_name] = obj
                self.get_logger().info(f"Created collision object for link: {link_name}")
    
        self.get_logger().info(f"Created {len(self.robot_collision_objects)} robot collision objects")

    def update_collision_object_pose(self, collision_object, pose_matrix):
        """Update the transform of an existing FCL collision object"""
        new_transform = fcl.Transform(pose_matrix[0:3, 0:3], pose_matrix[0:3, 3])
        collision_object.setTransform(new_transform)

    def testing(self):
        """
        Purpose:
        ---
        Performs collision detection by predicting future robot poses and checking for self-collision and boundary collisions

        Input Arguments:
        ---
        None (uses current joint positions and velocities from class attributes)

        Returns:
        ---
        `bool`
            True if collision is detected, False otherwise

        Example call:
        ---
        collision = self.testing()
        if collision:
            self.stop_robot()
        """
        if self.joint_positions is None:
            self.get_logger().info('Waiting for joint states...')
            return

        delta_twist_joint_velocity = self.joint_velocities
        current_positions = list(self.joint_positions)
        predicted_positions = [i for i in current_positions]
        delta_t = 0.1#0.4
        
        for i, val in enumerate(delta_twist_joint_velocity):
            predicted_positions[i] = current_positions[i] + delta_t * delta_twist_joint_velocity[i]

        # 1. Get current link poses from TF
        current_poses = {}
        for link in self.link_names:
            tf_pose = self.get_link_pose(link)
            if tf_pose is not None:
                current_poses[link] = tf_pose

        # 2. Compute FK for predicted joint positions
        fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        joint_array = kdl.JntArray(self.kdl_chain.getNrOfJoints())
        for i in range(min(len(predicted_positions), joint_array.rows())):
            joint_array[i] = predicted_positions[i]

        predicted_global_poses = {}
        for i in range(self.kdl_chain.getNrOfSegments()):
            frame = kdl.Frame()
            fk_solver.JntToCart(joint_array, frame, i + 1)
            pose = np.eye(4)
            for r in range(3):
                for c in range(3):
                    pose[r, c] = frame.M[r, c]
                pose[r, 3] = frame.p[r]
            link_name = self.kdl_chain.getSegment(i).getName()
            predicted_global_poses[link_name] = pose
            
            # Add debugging to see link positions
            # if 'wrist' in link_name:
            #     self.get_logger().info(f"Link {link_name} position: {pose[0:3, 3]}")
        # print("predicted_global_poses:", predicted_global_poses)
        tcp_matrix = predicted_global_poses['tool0']
        eef_pose_msg = PoseStamped()
        eef_pose_msg.header.stamp = self.get_clock().now().to_msg()
        eef_pose_msg.header.frame_id = "base_link"
        eef_pose_msg.pose.position.x = tcp_matrix[0, 3]
        eef_pose_msg.pose.position.y = tcp_matrix[1, 3]
        eef_pose_msg.pose.position.z = tcp_matrix[2, 3]
        r = Rotation.from_matrix(tcp_matrix[:3, :3])
        q = r.as_quat()
        eef_pose_msg.pose.orientation.x = q[0]
        eef_pose_msg.pose.orientation.y = q[1]
        eef_pose_msg.pose.orientation.z = q[2]
        eef_pose_msg.pose.orientation.w = q[3]
        self.eef_pub.publish(eef_pose_msg)
        # print("Published predicted TCP pose:", eef_pose_msg.pose)
        # 3. Combine base→current_link and current_link→predicted_link to get predicted global pose
        # predicted_global_poses = {}
        # for link in self.link_names:
            # print("\nProcessing link:", link)
            # print("Current pose:\n", current_poses.get(link, None))
            # print("Predicted relative pose:\n", predicted_relative_poses.get(link, None))
            # if link in current_poses and link in predicted_relative_poses:
                # predicted_global_poses[link] = np.dot(current_poses[link], predicted_relative_poses[link])
            # elif link in current_poses:
            # print("Predicted global pose:\n", predicted_global_poses[link])


        # 4. Update existing collision objects with predicted poses (MODIFIED)
        valid_collision_objects = []
        valid_links = []
        
        for link_name in self.robot_collision_objects:
            # if (link_name in predicted_global_poses and 
            #     link_name in self.robot_collision_objects):
                
                # Update the existing collision object's pose
            pose = predicted_global_poses[link_name]
            self.update_collision_object_pose(self.robot_collision_objects[link_name], pose)
            
            valid_collision_objects.append(self.robot_collision_objects[link_name])
            valid_links.append(link_name)

        # 5. Collision check (using updated objects instead of newly created ones)
        collision_detected = False
        
        # Check robot self-collision
        for i in range(len(valid_collision_objects)):
            for j in range(i + 1, len(valid_collision_objects)):
                l1 = valid_links[i]
                l2 = valid_links[j]
                if (l1, l2) in ADJACENT_PAIRS or (l2, l1) in ADJACENT_PAIRS:
                    continue
                req = fcl.CollisionRequest()
                res = fcl.CollisionResult()
                fcl.collide(valid_collision_objects[i], valid_collision_objects[j], req, res)
                if res.is_collision:
                    self.get_logger().warn(f"Predicted self-collision between {l1} and {l2} in next step. Stopping robot.")
                    collision_detected = True
                    break
            if collision_detected:
                break
        
        # Check collision with boundary planes
        if not collision_detected:
            for i, robot_link in enumerate(valid_collision_objects):
                link_name = valid_links[i]
                
                if 'base' in link_name:
                    continue

                # if 'upper_arm_link' in link_name:
                #     continue
                    
                for j, plane in enumerate(self.boundary_planes):
                    req = fcl.CollisionRequest()
                    res = fcl.CollisionResult()
                    fcl.collide(robot_link, plane, req, res)
                    if res.is_collision:
                        plane_name = "Plane_"+str(j)
                        self.get_logger().warn(f"Predicted collision between {link_name} and {plane_name} in next step. Stopping robot.")
                        collision_detected = True
                        break
                if collision_detected:
                    break
        
        return collision_detected

    def create_boundary_planes(self):
        """Create 6 simple boundary planes around UR5"""
        planes = []
        # BOUNDARY PLANE CONFIGURATION - Modify these values as needed
        # The following values are the offsets
        # self.WORKSPACE_LIMITS = {
        #     'x_min': -0.5, 'x_max': 0.5,    # Left/Right walls
        #     'y_min': -0.5, 'y_max': 0.5,    # Front/Back walls  
        #     'z_min': -0.0, 'z_max': 1.0     # Floor/Ceiling
        # }


        
        # Use class parameters for plane definitions
        plane_configs = [
            ([-1, 0, 0], self.WORKSPACE_LIMITS['x_min']),    #Plane 0  # Left wall 
            ([1, 0, 0], self.WORKSPACE_LIMITS['x_max']),   #Plane 1  # Right wall  
            ([0, -1, 0], self.WORKSPACE_LIMITS['y_min']),    #Plane 2  # Front wall 
            ([0, 1, 0], self.WORKSPACE_LIMITS['y_max']),   #Plane 3  # Back wall 
            ([0, 0, -1], self.WORKSPACE_LIMITS['z_min']),    #Plane 4  # Floor 
            ([0, 0, 1], self.WORKSPACE_LIMITS['z_max'])    #Plane 5  # Ceiling 
        ]
        
        for i, (normal, offset) in enumerate(plane_configs):
            normal = np.array(normal, dtype=float)
            plane = fcl.Plane(normal, abs(offset))
            transform = fcl.Transform()
            plane_obj = fcl.CollisionObject(plane, transform)
            planes.append(plane_obj)
        
        self.get_logger().info(f"Created {len(planes)} boundary planes")
        return planes

    def publish_plane_markers(self):
        """Publish visualization markers for boundary planes"""
        marker_array = MarkerArray()
        
        if not self.show_planes:
            return
        
        # Use class parameters for marker definitions
        planes = [
            # [position, scale, color, name]
            ([self.WORKSPACE_LIMITS['x_min'], 0, PLANE_SIZE/2.0], [PLANE_THICKNESS, PLANE_SIZE, PLANE_SIZE], [1, 0, 0, PLANE_ALPHA], "Left Wall"),
            ([self.WORKSPACE_LIMITS['x_max'], 0, PLANE_SIZE/2.0], [PLANE_THICKNESS, PLANE_SIZE, PLANE_SIZE], [1, 0, 0, PLANE_ALPHA], "Right Wall"),
            ([0, self.WORKSPACE_LIMITS['y_min'], PLANE_SIZE/2.0], [PLANE_SIZE, PLANE_THICKNESS, PLANE_SIZE], [0, 1, 0, PLANE_ALPHA], "Front Wall"),
            ([0, self.WORKSPACE_LIMITS['y_max'], PLANE_SIZE/2.0], [PLANE_SIZE, PLANE_THICKNESS, PLANE_SIZE], [0, 1, 0, PLANE_ALPHA], "Back Wall"),
            ([0, 0, self.WORKSPACE_LIMITS['z_min']],              [PLANE_SIZE, PLANE_SIZE, PLANE_THICKNESS], [0, 0, 1, PLANE_ALPHA], "Floor"),
            ([0, 0, self.WORKSPACE_LIMITS['z_max']],              [PLANE_SIZE, PLANE_SIZE, PLANE_THICKNESS], [0, 0, 1, PLANE_ALPHA], "Ceiling")
        ]
        
        for i, (pos, scale, color, name) in enumerate(planes):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "boundary_planes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            
            # No rotation needed - keep default orientation
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            
            # Scale
            marker.scale.x = float(scale[0])
            marker.scale.y = float(scale[1])
            marker.scale.z = float(scale[2])
            
            # Color
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(color[3])
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    """
    Purpose:
    ---
    Main entry point for the Cartesian servo manipulation node

    Input Arguments:
    ---
    `args` : list, optional
        Command line arguments passed to rclpy.init()

    Returns:
    ---
    None

    Example call:
    ---
    main()
    """
    rclpy.init(args=args)
    node = CartesianServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()