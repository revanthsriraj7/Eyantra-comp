'''
# Team ID:          < eYRC#1912 >
# Theme:            < Krishi coBot >
# Author List:      < K P R B Revanth Sriraj, Chirag Sharma, Chandan GS >
# Filename:         < detection.py >
# Functions:        < CameraIntrinsics.is_valid, DetectionResult.has_valid_3d, OpenCVDepthNode.init, 
#                      OpenCVDepthNode.setup_params, OpenCVDepthNode.load_params, OpenCVDepthNode.setup_camera, 
#                      OpenCVDepthNode.setup_detection, OpenCVDepthNode.setup_aruco_detector, 
#                      OpenCVDepthNode.setup_subs, OpenCVDepthNode.setup_windows, 
#                      OpenCVDepthNode.convert_pixel_to_world, OpenCVDepthNode.fix_coordinates, OpenCVDepthNode.get_param_or_default, 
#                      OpenCVDepthNode.publish_fruit_position, OpenCVDepthNode.detect_aruco_markers, 
#                      OpenCVDepthNode.process_images, OpenCVDepthNode.convert_images, 
#                      OpenCVDepthNode.find_bad_fruits, OpenCVDepthNode.process_detection, OpenCVDepthNode.get_depth, 
#                      OpenCVDepthNode.get_depth_at_point, OpenCVDepthNode.show_results, OpenCVDepthNode.draw_detection, 
#                      OpenCVDepthNode.draw_detection_on_depth, OpenCVDepthNode.make_depth_colormap, OpenCVDepthNode.save_frame, 
#                      OpenCVDepthNode.print_stats, OpenCVDepthNode.create_depth_colormap, main >
# Global variables: < None >
'''

#This code was made by referring example code and boiler code

# Combined Detection - ArUco Markers + Bad Fruits
# This code detects both ArUco markers and bad fruits using camera and depth sensor
# Uses ArUco marker detection for objects like fertilizer can (marker ID 3)
# Uses HSV color detection to find grey fruits with green spots

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from typing import Optional, Tuple, List
import traceback
from dataclasses import dataclass
import math
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from rclpy.duration import Duration


@dataclass
class CameraIntrinsics:
    # stores camera parameters for converting pixels to 3d coordinates
    width: int
    height: int
    cx: float  # center point x
    cy: float  # center point y  
    fx: float  # focal lenght x
    fy: float  # focal lenght y
    
    def is_valid(self) -> bool:
        # checks if camera parameters are ok
        return all([
            self.width > 0,
            self.height > 0,
            self.fx > 0,
            self.fy > 0,
            0 <= self.cx <= self.width,
            0 <= self.cy <= self.height
        ])


@dataclass  
class DetectionResult:
    # stores info about one detected bad fruit
    pixel_x: int
    pixel_y: int
    bbox: Tuple[int, int, int, int]  # bounding box coordinates
    confidence: float
    depth: float
    world_x: Optional[float] = None  # 3d position in real world
    world_y: Optional[float] = None
    world_z: Optional[float] = None
    frame_id: Optional[str] = None
    
    @property
    def has_valid_3d(self) -> bool:
        # check if we got valid 3d coordinates or not
        return all(coord is not None for coord in [self.world_x, self.world_y, self.world_z])


class OpenCVDepthNode(Node):
    # main class that does all the fruit detection stuff
    
    def __init__(self):
        super().__init__('opencv_bad_fruit_depth_node')
        
        # setup all the ros2 components we need
        self.bridge = CvBridge()  # converts images between ros and opencv
        self.tf_broadcaster = TransformBroadcaster(self)  # publishes 3d positions
        self.tf_buffer = Buffer()  # for coordinate transformations
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_intrinsics: Optional[CameraIntrinsics] = None
        self.camera_info_received = False  # flag to check if we got camera info
        
        # load all parameters from config
        self.setup_params()
        self.load_params()
        
        # setup coordinate system (had issues with frames being above camera)
        self.coordinate_transform = self.get_param_or_default('coordinate_transform', 'flip_y')
        
        # setup camera stuff
        self.setup_camera()
        
        # setup opencv detection with hsv ranges
        self.setup_detection()
        
        # setup ArUco marker detector
        self.setup_aruco_detector()
        
        # setup subscribers for rgb and depth images  
        self.setup_subs()
        
        # setup opencv windows if we want to see whats happening
        if self.enable_visualization:
            self.setup_windows()
        
        # keep track of how many frames we processed
        self.frame_count = 0
        self.detection_stats = {'total': 0, 'valid_3d': 0, 'invalid_depth': 0}
        self.aruco_stats = {'total': 0, 'valid_3d': 0}
        
        detection_types = []
        if self.aruco_enable:
            detection_types.append("ArUco markers")
        detection_types.append("bad fruits")
        
        self.get_logger().info(f"Combined detection node started successfully!")
        self.get_logger().info(f"Detecting: {' + '.join(detection_types)}")
    
    def setup_params(self):
        # setup all the parameters with default values
        self.declare_parameter('confidence_threshold', 0.05)
        self.declare_parameter('max_detections', 20)  # dont detect too many at once
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('target_frame', 'base_link')  # where to publish fruit positions
        self.declare_parameter('enable_visualization', True)  # show opencv windows or not
        self.declare_parameter('sync_slop', 0.1)  # how much time difference allowed between rgb and depth
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('use_camera_info', True)  # get intrinsics from camera_info topic
        
        # ArUco detection parameters
        self.declare_parameter('aruco.enable', True)  # enable ArUco marker detection
        self.declare_parameter('aruco.min_marker_area', 500)  # minimum marker area to consider valid
        

        #camera intrinsics values
        # camera calibration values (fallback defaults if camera_info not available)
        self.declare_parameter('camera.width', 1280)
        self.declare_parameter('camera.height', 720)
        self.declare_parameter('camera.cx', 642.724365234375)  # principal point x
        self.declare_parameter('camera.cy', 361.9780578613281)  # principal point y
        self.declare_parameter('camera.fx', 915.3003540039062)  # focal length x
        self.declare_parameter('camera.fy', 914.0320434570312)  # focal length y
        
        # coordinate system transformation (was having issues with frames)
        self.declare_parameter('coordinate_transform', 'camera_to_ros')
    
    def load_params(self):
        # get all parameter values from config
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.enable_visualization = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        self.sync_slop = self.get_parameter('sync_slop').get_parameter_value().double_value
        self.queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.use_camera_info = self.get_parameter('use_camera_info').get_parameter_value().bool_value
        
        # ArUco parameters
        self.aruco_enable = self.get_parameter('aruco.enable').get_parameter_value().bool_value
        self.min_marker_area = self.get_parameter('aruco.min_marker_area').get_parameter_value().integer_value
        
        self.get_logger().info(f"loaded params: conf={self.confidence_threshold}, max_detect={self.max_detections}, use_camera_info={self.use_camera_info}, aruco={self.aruco_enable}")
    
    def setup_camera(self):
        # setup camera calibration values
        try:
            # if use_camera_info is enabled, subscribe to camera_info topic
            if self.use_camera_info:
                self.get_logger().info("waiting for camera_info from /camera/camera/color/camera_info...")
                self.camera_info_sub = self.create_subscription(
                    CameraInfo,
                    '/camera/camera/color/camera_info',
                    self.camera_info_callback,
                    10
                )
                # don't proceed until we get camera info
                return
            
            # otherwise use hardcoded parameters
            self.camera_intrinsics = CameraIntrinsics(
                width=self.get_parameter('camera.width').get_parameter_value().integer_value,
                height=self.get_parameter('camera.height').get_parameter_value().integer_value,
                cx=self.get_parameter('camera.cx').get_parameter_value().double_value,
                cy=self.get_parameter('camera.cy').get_parameter_value().double_value,
                fx=self.get_parameter('camera.fx').get_parameter_value().double_value,
                fy=self.get_parameter('camera.fy').get_parameter_value().double_value
            )
            
            if not self.camera_intrinsics.is_valid():
                raise ValueError("camera parameters are wrong!")
            
            self.camera_info_received = True
            self.get_logger().info(f"camera setup (hardcoded): {self.camera_intrinsics.width}x{self.camera_intrinsics.height}, "
                                  f"focal=({self.camera_intrinsics.fx:.1f}, {self.camera_intrinsics.fy:.1f}), "
                                  f"center=({self.camera_intrinsics.cx:.1f}, {self.camera_intrinsics.cy:.1f})")
        except Exception as e:
            self.get_logger().error(f"camera setup failed: {e}")
            raise
    
    def camera_info_callback(self, msg: CameraInfo):
        # callback to receive camera intrinsics from camera_info topic
        if self.camera_info_received:
            return  # already got it, no need to process again
        
        try:
            # extract intrinsics from K matrix [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.camera_intrinsics = CameraIntrinsics(
                width=msg.width,
                height=msg.height,
                cx=msg.k[2],  # K[0,2]
                cy=msg.k[5],  # K[1,2]
                fx=msg.k[0],  # K[0,0]
                fy=msg.k[4]   # K[1,1]
            )
            
            if not self.camera_intrinsics.is_valid():
                raise ValueError("camera parameters from camera_info are invalid!")
            
            self.camera_info_received = True
            self.get_logger().info(f"camera setup (from camera_info): {self.camera_intrinsics.width}x{self.camera_intrinsics.height}, "
                                  f"focal=({self.camera_intrinsics.fx:.1f}, {self.camera_intrinsics.fy:.1f}), "
                                  f"center=({self.camera_intrinsics.cx:.1f}, {self.camera_intrinsics.cy:.1f})")
            
            # unsubscribe after getting the info (we only need it once)
            self.destroy_subscription(self.camera_info_sub)
            
        except Exception as e:
            self.get_logger().error(f"failed to parse camera_info: {e}")
            # fall back to hardcoded values
            self.use_camera_info = False
            self.setup_camera()
    
    def setup_detection(self):
        # setup hsv color ranges for detecting bad fruits
        try:
            # these hsv values were found using trackbars (took forever to tune!)
            # gray fruit body: 
            self.grey_lower = np.array([14, 6, 73])
            self.grey_upper = np.array([124, 39, 124])
            
            # green moldy spots:
            self.green_lower = np.array([49, 67, 182])
            self.green_upper = np.array([59, 125, 196])
            
            # minimum sizes for detection
            self.min_contour_area = 500  # fruit must be atleast this big
            self.min_green_pixels = 50   # must have some green mold
            
            self.get_logger().info('hsv ranges set:')
            self.get_logger().info(f'grey fruit: [9,18,81] to [179,32,200]')
            self.get_logger().info(f'green mold: [38,245,153] to [179,253,200]')
            
        except Exception as e:
            self.get_logger().error(f"opencv setup failed: {e}")
            raise
    
    def setup_aruco_detector(self):
        # setup ArUco marker detector for detecting markers like fertilizer can
        if not self.aruco_enable:
            self.get_logger().info("ArUco detection disabled")
            return
        
        try:
            # Use 4x4_50 dictionary (standard for competition)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            
            # Tune parameters to reduce false detections
            aruco_params.minMarkerPerimeterRate = 0.03
            aruco_params.maxMarkerPerimeterRate = 4.0
            aruco_params.minCornerDistanceRate = 0.05
            aruco_params.minDistanceToBorder = 3
            aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            
            self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            
            self.get_logger().info("ArUco detector configured (DICT_4X4_50)")
            
        except Exception as e:
            self.get_logger().error(f"ArUco detector setup failed: {e}")
            self.aruco_enable = False
    
    def setup_subs(self):
        # setup subscribers for rgb and depth camera feeds
        try:
            self.rgb_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
            self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
            
            # synchronize rgb and depth images (they come at slightly different times)
            self.ts = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub],
                queue_size=self.queue_size,
                slop=self.sync_slop  # allow small time difference
            )
            self.ts.registerCallback(self.process_images)
            
            self.get_logger().info(f"camera subscribers ready, queue={self.queue_size}, slop={self.sync_slop}s")
        except Exception as e:
            self.get_logger().error(f"subscriber setup failed: {e}")
            raise
    
    def setup_windows(self):
        # create opencv windows to see what the camera sees
        try:
            cv2.namedWindow("OpenCV Bad Fruit Detection", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grey Mask", cv2.WINDOW_NORMAL)  # shows grey detection
            cv2.namedWindow("Green Mask", cv2.WINDOW_NORMAL)  # shows green detection
            self.get_logger().info("opencv windows created")
        except Exception as e:
            self.get_logger().error(f"couldnt create windows: {e}")
            self.enable_visualization = False  # disable if failed

    def convert_pixel_to_world(self, u: int, v: int, depth: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        # convert pixel position + depth to real world 3d coordinates
        # this uses camera calibration math (pinhole camera model)
        
        if not self.camera_intrinsics:
            self.get_logger().error("camera not setup yet!")
            return None, None, None
        
        # check if depth value is valid (not zero or nan)
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None, None, None
        
        # make sure pixel is within image bounds
        if not (0 <= u < self.camera_intrinsics.width and 0 <= v < self.camera_intrinsics.height):
            self.get_logger().warn(f"pixel ({u}, {v}) is outside image!")
            return None, None, None
        
        try:
            # do the math to convert pixel to 3d point
            X = (u - self.camera_intrinsics.cx) * depth / self.camera_intrinsics.fx
            Y = (v - self.camera_intrinsics.cy) * depth / self.camera_intrinsics.fy
            Z = depth
            
            # fix coordinate system issues (frames were appearing above camera)
            X, Y, Z = self.fix_coordinates(X, Y, Z)
            
            return X, Y, Z
        except Exception as e:
            self.get_logger().error(f"3d math failed: {e}")
            return None, None, None
    
    def fix_coordinates(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        # fix coordinate system so frames appear in right place
        # had big issues with fruits appearing above camera instead of where they actually are
        
        if self.coordinate_transform == 'none':
            # dont change anything
            return x, y, z
        
        elif self.coordinate_transform == 'flip_y':
            # just flip y axis (this fixed most of our problems)
            return x, -y, z
        
        elif self.coordinate_transform == 'camera_to_ros':
            # full transformation from camera coords to ros coords
            # camera: x=right, y=down, z=forward  
            # ros: x=forward, y=left, z=up
            return z, -x, -y  
        
        elif self.coordinate_transform == 'camera_to_ros_alt':
            # alternative transformation (if camera mounted different)
            return z, x, -y
        
        else:
            # default just flip y
            self.get_logger().warn(f"dont know transform '{self.coordinate_transform}', using flip_y")
            return x, -y, z
    
    def get_param_or_default(self, param_name: str, default_value):
        # try to get parameter value, if it doesnt exist use default
        try:
            param = self.get_parameter(param_name)
            if param.type_ == Parameter.Type.STRING:
                return param.get_parameter_value().string_value
            elif param.type_ == Parameter.Type.DOUBLE:
                return param.get_parameter_value().double_value
            elif param.type_ == Parameter.Type.INTEGER:
                return param.get_parameter_value().integer_value
            elif param.type_ == Parameter.Type.BOOL:
                return param.get_parameter_value().bool_value
            else:
                return default_value
        except:
            return default_value  # parameter not found, use default

    def publish_fruit_position(self, child_frame, x, y, z, timestamp):
        # publish 3d position of detected fruit as tf frame
        # this is so the robot arm can know where the fruit is
        
        # figure out which frame to publish relative to
        parent_frame = getattr(self, 'target_frame', None) or self.camera_frame

        # start with coordinates in camera frame
        px, py, pz = float(x), float(y), float(z)

        # if we want to publish relative to different frame, need to transform
        if parent_frame != self.camera_frame:
            try:
                # lookup transformation between frames
                trans = self.tf_buffer.lookup_transform(
                    parent_frame,              
                    self.camera_frame,         
                    Time.from_msg(timestamp),
                    timeout=Duration(seconds=0.2)
                )

                # get translation and rotation from transform
                tx = trans.transform.translation.x
                ty = trans.transform.translation.y
                tz = trans.transform.translation.z
                qx = trans.transform.rotation.x
                qy = trans.transform.rotation.y
                qz = trans.transform.rotation.z
                qw = trans.transform.rotation.w

                # convert quaternion to rotation matrix (math is confusing but works)
                r00 = 1 - 2*(qy*qy + qz*qz)
                r01 = 2*(qx*qy - qz*qw)
                r02 = 2*(qx*qz + qy*qw)
                r10 = 2*(qx*qy + qz*qw)
                r11 = 1 - 2*(qx*qx + qz*qz)
                r12 = 2*(qy*qz - qx*qw)
                r20 = 2*(qx*qz - qy*qw)
                r21 = 2*(qy*qz + qx*qw)
                r22 = 1 - 2*(qx*qx + qy*qy)

                # apply transformation: new_point = rotation * old_point + translation
                px_new = r00*px + r01*py + r02*pz + tx
                py_new = r10*px + r11*py + r12*pz + ty
                pz_new = r20*px + r21*py + r22*pz + tz
                px, py, pz = px_new, py_new, pz_new
            except Exception as e:
                self.get_logger().warn(f"tf lookup failed {self.camera_frame}->{parent_frame}: {e}. using camera frame instead")
                parent_frame = self.camera_frame

        # create transform message
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # set position
        t.transform.translation.x = px
        t.transform.translation.y = py
        t.transform.translation.z = pz

        # set orientation (just keep it simple, no rotation)
        angle_rad = -math.radians(0.0)
        half_angle = angle_rad / 2.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = math.sin(half_angle)
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = math.cos(half_angle)

        # actually publish the transform
        self.tf_broadcaster.sendTransform(t)
    
    def process_images(self, rgb_msg: Image, depth_msg: Image):
        # this gets called when both rgb and depth images arrive at same time
        # main processing happens here
        try:
            # wait until we have camera intrinsics before processing
            if not self.camera_info_received:
                return  # skip processing until camera info is received
            
            self.frame_count += 1  # keep track of how many frames we processed
            
            # convert ros images to opencv format
            rgb_frame, depth_frame = self.convert_images(rgb_msg, depth_msg)
            if rgb_frame is None or depth_frame is None:
                return  # skip if conversion failed
            
            # detect ArUco markers first (like fertilizer can)
            aruco_detections = self.detect_aruco_markers(rgb_frame, depth_frame, rgb_msg.header.stamp)
            
            # run the actual fruit detection
            detections = self.find_bad_fruits(rgb_frame, depth_frame, rgb_msg.header.stamp)
            
            # update our statistics
            self.detection_stats['total'] += len(detections)
            self.detection_stats['valid_3d'] += sum(1 for d in detections if d.has_valid_3d)
            self.detection_stats['invalid_depth'] += sum(1 for d in detections if not d.has_valid_3d)
            
            # show results in opencv windows if enabled
            if self.enable_visualization:
                self.show_results(rgb_frame, depth_frame, detections, aruco_detections)
            
            # print stats every 100 frames
            if self.frame_count % 100 == 0:
                self.print_stats()
                
        except Exception as e:
            self.get_logger().error(f"callback failed: {e}")
            self.get_logger().error(f"error details: {traceback.format_exc()}")
    
    def convert_images(self, rgb_msg: Image, depth_msg: Image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # convert ros image messages to opencv arrays
        try:
            # convert color image to bgr format (opencv uses bgr not rgb)
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            
            # depth images can have different encodings, handle each one
            if depth_msg.encoding == '32FC1':
                # already in meters as float32
                depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            elif depth_msg.encoding == '16UC1':
                # in millimeters as uint16, convert to meters
                depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
                depth_frame = depth_frame.astype(np.float32) / 1000.0  
            elif depth_msg.encoding == 'mono16':
                # also in millimeters usually
                depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='mono16')
                depth_frame = depth_frame.astype(np.float32) / 1000.0  
            else:
                self.get_logger().warn(f"unknown depth encoding: {depth_msg.encoding}, trying passthrough")
                depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # make sure rgb and depth are same size (they should be aligned)
            if rgb_frame.shape[:2] != depth_frame.shape[:2]:
                self.get_logger().error(f"image sizes dont match! rgb={rgb_frame.shape[:2]} depth={depth_frame.shape[:2]}")
                return None, None
            
            return rgb_frame, depth_frame
            
        except Exception as e:
            self.get_logger().error(f"image conversion failed: {e}")
            return None, None

    def detect_aruco_markers(self, rgb_frame: np.ndarray, depth_frame: np.ndarray, timestamp) -> List[Tuple[int, int, int, float]]:
        # detect ArUco markers in the frame
        # returns list of (marker_id, center_x, center_y, depth)
        
        if not self.aruco_enable:
            return []
        
        detections = []
        
        try:
            # convert to grayscale for ArUco detection
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            
            # detect ArUco markers
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            # process detected markers
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    # get corners of this marker
                    marker_corners = corners[i][0]
                    
                    # calculate marker area to filter small detections
                    x_coords = marker_corners[:, 0]
                    y_coords = marker_corners[:, 1]
                    width = np.max(x_coords) - np.min(x_coords)
                    height = np.max(y_coords) - np.min(y_coords)
                    area = width * height
                    
                    # skip if marker is too small (likely false positive)
                    if area < self.min_marker_area:
                        self.get_logger().debug(f"skipping small ArUco ID {marker_id}, area={area:.0f}")
                        continue
                    
                    # calculate center point
                    center_x = int(np.mean(marker_corners[:, 0]))
                    center_y = int(np.mean(marker_corners[:, 1]))
                    
                    # get depth at center with robust method
                    depth = self.get_depth(depth_frame, center_x, center_y)
                    
                    # convert to 3D world coordinates
                    world_x, world_y, world_z = self.convert_pixel_to_world(center_x, center_y, depth)
                    
                    # publish transform if valid
                    if world_x is not None and world_y is not None and world_z is not None:
                        # set child frame ID based on marker ID
                        # marker ID 3 is for fertilizer can (as per competition rules)
                        if marker_id == 3:
                            child_frame = "1912_fertiliser_can"
                        else:
                            child_frame = f"1912_aruco_marker_{marker_id}"
                        
                        self.publish_fruit_position(child_frame, world_x, world_y, world_z, timestamp)
                        
                        self.aruco_stats['valid_3d'] += 1
                        
                        self.get_logger().info(
                            f"ArUco ID {marker_id}: pixel=({center_x}, {center_y}), depth={depth:.3f}m, "
                            f"3D=({world_x:.3f}, {world_y:.3f}, {world_z:.3f}), area={area:.0f}"
                        )
                    
                    self.aruco_stats['total'] += 1
                    detections.append((marker_id, center_x, center_y, depth))
                    
                    # draw marker center on frame
                    cv2.circle(rgb_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # add text label
                    if world_x is not None:
                        if marker_id == 3:
                            text = f"Fertiliser Can: D={depth:.2f}m"
                        else:
                            text = f"ArUco {marker_id}: D={depth:.2f}m"
                        cv2.putText(rgb_frame, text, (center_x - 80, center_y - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # draw detected marker outlines
                cv2.aruco.drawDetectedMarkers(rgb_frame, corners, ids)
        
        except Exception as e:
            self.get_logger().error(f"ArUco detection failed: {e}")
        
        return detections

    def find_bad_fruits(self, rgb_frame: np.ndarray, depth_frame: np.ndarray, timestamp) -> List[DetectionResult]:
        # main detection function - finds bad fruits using hsv color detection
        detections = []
        
        try:
            # convert bgr image to hsv (hue saturation value) for better color detection
            hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
            
            # create mask for grey fruit bodies using our calibrated hsv range
            mask = cv2.inRange(hsv, self.grey_lower, self.grey_upper)
            
            # clean up the mask with morphological operations (removes noise)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise
            
            # find contours (outlines) of detected objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # save masks for debugging visualization
            self.current_grey_mask = mask
            self.current_green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            detection_count = 0
            for contour in contours:
                # skip if object is too small
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue
                
                # get bounding rectangle around the object
                x, y, w, h = cv2.boundingRect(contour)
                
                # look inside this rectangle for green mold
                roi_hsv = hsv[y:y+h, x:x+w]  # region of interest
                
                # check for green color inside the fruit
                green_mask = cv2.inRange(roi_hsv, self.green_lower, self.green_upper)
                green_pixels = cv2.countNonZero(green_mask)
                
                # only count as bad fruit if it has both grey body AND green mold
                if green_pixels > self.min_green_pixels:
                    detection_count += 1
                    if detection_count > self.max_detections:
                        break  # dont detect too many at once
                    
                    # process this detection and calculate 3d position
                    detection = self.process_detection(
                        x, y, w, h, area, green_pixels, depth_frame, detection_count, timestamp
                    )
                    if detection:
                        detections.append(detection)
            
        except Exception as e:
            self.get_logger().error(f"detection failed: {e}")
        
        return detections
    
    def process_detection(self, x: int, y: int, w: int, h: int, area: float, 
                                green_pixels: int, depth_frame: np.ndarray, 
                                detection_id: int, timestamp) -> Optional[DetectionResult]:
        # process one detected fruit and calculate its 3d position
        try:
            # find center of the bounding box
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            # get depth at center point (use robust method to handle noisy depth data)
            depth_value = self.get_depth(depth_frame, center_x, center_y)
            
            # convert pixel + depth to real world 3d coordinates
            world_x, world_y, world_z = self.convert_pixel_to_world(center_x, center_y, depth_value)
            
            # calculate confidence based on size and amount of green mold
            confidence = min(1.0, (area / 5000.0) * (green_pixels / 200.0))
            
            # create detection object with all the info
            detection = DetectionResult(
                pixel_x=center_x,
                pixel_y=center_y,
                bbox=(x, y, x + w, y + h),
                confidence=confidence,
                depth=depth_value,
                world_x=world_x,
                world_y=world_y,
                world_z=world_z
            )
            
            # if we got valid 3d position, publish it as tf frame
            if detection.has_valid_3d:
                detection.frame_id = f"1912_bad_fruit_{detection_id}"  # our team number
                self.publish_fruit_position(detection.frame_id, world_x, world_y, world_z, timestamp)
                
                self.get_logger().debug(
                    f"found {detection.frame_id}: pixel({center_x}, {center_y}), "
                    f"depth={depth_value:.3f}m, 3d=({world_x:.3f}, {world_y:.3f}, {world_z:.3f}), "
                    f"size={area:.0f}, green_pixels={green_pixels}"
                )
            else:
                self.get_logger().debug(f"detection #{detection_id}: bad depth at ({center_x}, {center_y})")
            
            return detection
            
        except Exception as e:
            self.get_logger().error(f"processing detection failed: {e}")
            return None
    
    def get_depth(self, depth_frame: np.ndarray, x: int, y: int, window_size: int = 5) -> float:
        # get depth value that handles noisy depth data
        # instead of using just one pixel, look at small area around it and take median
        
        h, w = depth_frame.shape[:2]
        
        # make sure coordinates are inside image
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # create small window around the point
        half_window = window_size // 2
        x1 = max(0, x - half_window)
        x2 = min(w, x + half_window + 1)
        y1 = max(0, y - half_window)
        y2 = min(h, y + half_window + 1)
        
        # get all depth values in this window
        depth_window = depth_frame[y1:y2, x1:x2]
        
        # remove invalid depths (nan, inf, zero, negative)
        valid_depths = depth_window[
            ~(np.isnan(depth_window) | np.isinf(depth_window) | (depth_window <= 0))
        ]
        
        if len(valid_depths) == 0:
            return 0.0  # no valid depth found
        
        # use median instead of mean to avoid outliers messing up the result
        return float(np.median(valid_depths))
    
    def get_depth_at_point(self, depth_frame: np.ndarray, x: int, y: int) -> float:
        # old function name kept for compatibility (just calls the robust version)
        return self.get_depth(depth_frame, x, y, window_size=1)
    
    def show_results(self, rgb_frame: np.ndarray, depth_frame: np.ndarray, detections: List[DetectionResult], aruco_detections: List = []):
        # show detection results in opencv windows (helps with debugging)
        try:
            # make copies so we dont mess up original images
            rgb_vis = rgb_frame.copy()
            depth_colormap = self.make_depth_colormap(depth_frame)
            
            # draw bounding boxes and info on detected fruits
            for detection in detections:
                self.draw_detection(rgb_vis, detection)
                self.draw_detection_on_depth(depth_colormap, detection)
            
            # add some info text at top
            info_text = f"Frame: {self.frame_count} | Fruits: {len(detections)} | ArUco: {len(aruco_detections)} | 3D: {sum(1 for d in detections if d.has_valid_3d)}"
            cv2.putText(rgb_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # show the images in windows
            cv2.imshow("OpenCV Bad Fruit Detection", rgb_vis)
            cv2.imshow("Depth Image", depth_colormap)
            
            # show debug masks if we have them
            if hasattr(self, 'current_grey_mask') and hasattr(self, 'current_green_mask'):
                cv2.imshow("Grey Mask", self.current_grey_mask)
                cv2.imshow("Green Mask", self.current_green_mask)
            
            # check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # escape key
                self.get_logger().info("escape pressed, shutting down")
                rclpy.shutdown()
            elif key == ord('s'):  # s key to save current frame
                self.save_frame(rgb_vis, depth_colormap)
                
        except Exception as e:
            self.get_logger().error(f"visualization failed: {e}")
    
    def draw_detection(self, image: np.ndarray, detection: DetectionResult):
        # draw bounding box and info text on rgb image
        x1, y1, x2, y2 = detection.bbox
        
        # green box if we got 3d position, red if depth failed
        color = (0, 255, 0) if detection.has_valid_3d else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # mark center point with small circle
        cv2.circle(image, (detection.pixel_x, detection.pixel_y), 4, (0, 0, 255), -1)
        
        # add text info above bounding box
        y_offset = y1 - 5
        texts = [
            f"conf: {detection.confidence:.2f}",
            f"pixel: ({detection.pixel_x},{detection.pixel_y})",
            f"depth: {detection.depth:.3f}m"
        ]
        
        if detection.has_valid_3d:
            texts.append(f"3d: ({detection.world_x:.3f}, {detection.world_y:.3f}, {detection.world_z:.3f})")
            if detection.frame_id:
                texts.append(f"id: {detection.frame_id}")
        else:
            texts.append("bad depth!")
        
        # draw each line of text
        for i, text in enumerate(texts):
            y_pos = y_offset - (len(texts) - i) * 15
            cv2.putText(image, text, (x1, max(15, y_pos)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_detection_on_depth(self, depth_image: np.ndarray, detection: DetectionResult):
        # mark detected fruit on depth image
        cv2.circle(depth_image, (detection.pixel_x, detection.pixel_y), 4, (0, 0, 255), -1)
        cv2.putText(depth_image, f"{detection.depth:.2f}m",
                   (detection.pixel_x + 5, detection.pixel_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def make_depth_colormap(self, depth_frame: np.ndarray) -> np.ndarray:
        # convert depth image to colored version so its easier to see
        try:
            # work on copy so we dont mess up original
            depth_viz = depth_frame.copy()
            
            # fix invalid depth values (set to zero)
            depth_viz[np.isnan(depth_viz)] = 0
            depth_viz[np.isinf(depth_viz)] = 0
            depth_viz[depth_viz < 0] = 0
            
            # limit max depth for better visualization (ignore super far stuff)
            max_depth = np.percentile(depth_viz[depth_viz > 0], 95) if np.any(depth_viz > 0) else 5.0
            depth_viz = np.clip(depth_viz, 0, max_depth)
            
            # convert to 0-255 range for colormap
            if max_depth > 0:
                depth_normalized = (depth_viz / max_depth * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_viz, dtype=np.uint8)
            
            # apply jet colormap (blue=close, red=far)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            return depth_colormap
            
        except Exception as e:
            self.get_logger().error(f"depth colormap failed: {e}")
            return np.zeros((depth_frame.shape[0], depth_frame.shape[1], 3), dtype=np.uint8)
    
    def save_frame(self, rgb_frame: np.ndarray, depth_colormap: np.ndarray):
        # save current frame when user presses 's' key (useful for debugging)
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"/tmp/detection_rgb_{timestamp}.jpg", rgb_frame)
            cv2.imwrite(f"/tmp/detection_depth_{timestamp}.jpg", depth_colormap)
            self.get_logger().info(f"saved frames: {timestamp}")
        except Exception as e:
            self.get_logger().error(f"couldnt save frames: {e}")
    
    def print_stats(self):
        # print detection stats every 100 frames
        total = self.detection_stats['total']
        valid = self.detection_stats['valid_3d']
        invalid = self.detection_stats['invalid_depth']
        
        aruco_total = self.aruco_stats['total']
        aruco_valid = self.aruco_stats['valid_3d']
        
        if total > 0 or aruco_total > 0:
            if total > 0:
                success_rate = (valid / total) * 100
                self.get_logger().info(
                    f"Fruit stats (last 100 frames): found={total}, got_3d={valid} ({success_rate:.1f}%), "
                    f"bad_depth={invalid}"
                )
            
            if aruco_total > 0:
                self.get_logger().info(
                    f"ArUco stats (last 100 frames): found={aruco_total}, got_3d={aruco_valid}"
                )
            
            self.get_logger().info(f"Total frames processed: {self.frame_count}")
        
        # reset counters for next 100 frames
        self.detection_stats = {'total': 0, 'valid_3d': 0, 'invalid_depth': 0}
        self.aruco_stats = {'total': 0, 'valid_3d': 0}
    
    def create_depth_colormap(self, depth_frame: np.ndarray) -> np.ndarray:
        # old function name (kept for compatibility)
        return self.make_depth_colormap(depth_frame)


def main(args=None):
    # main function that starts everything
    rclpy.init(args=args)
    node = OpenCVDepthNode()
    
    try:
        rclpy.spin(node)  # keep running until ctrl+c
    except KeyboardInterrupt:
        pass  # exit gracefully when user presses ctrl+c
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()  # close all opencv windows
        rclpy.shutdown()


if __name__ == "__main__":
    main()