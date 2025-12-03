#!/usr/bin/env python3
"""
HSV Color Calibration Tool for Bad Fruit Detection
Team ID: eYRC#1912
Theme: Krishi coBot

This tool helps calibrate HSV ranges for detecting:
1. Grey fruit bodies
2. Green moldy spots on fruits

Usage:
    ros2 run task3a hsv_calibration

Controls:
    - Adjust sliders to find optimal HSV ranges
    - Press 's' to save current values
    - Press 'r' to reset to default values
    - Press 'ESC' to exit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from pathlib import Path


class HSVCalibrationNode(Node):
    def __init__(self):
        super().__init__('hsv_calibration_node')
        
        self.bridge = CvBridge()
        self.current_frame = None
        
        # Subscribe to RGB camera feed
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Default HSV ranges (from your detection.py)
        self.grey_lower = [9, 18, 81]
        self.grey_upper = [179, 32, 200]
        self.green_lower = [38, 245, 153]
        self.green_upper = [179, 253, 200]
        
        # Create windows
        self.setup_windows()
        
        # Create trackbars
        self.create_trackbars()
        
        self.get_logger().info("HSV Calibration Tool Started!")
        self.get_logger().info("Adjust sliders to tune detection")
        self.get_logger().info("Press 's' to save, 'r' to reset, 'ESC' to quit")
        
    def setup_windows(self):
        """Create all OpenCV windows"""
        cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Grey Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Green Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Combined Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Sliders - Grey", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Sliders - Green", cv2.WINDOW_NORMAL)
        
    def create_trackbars(self):
        """Create HSV range trackbars"""
        
        # Grey fruit trackbars
        cv2.createTrackbar("Grey H Min", "HSV Sliders - Grey", self.grey_lower[0], 179, lambda x: None)
        cv2.createTrackbar("Grey H Max", "HSV Sliders - Grey", self.grey_upper[0], 179, lambda x: None)
        cv2.createTrackbar("Grey S Min", "HSV Sliders - Grey", self.grey_lower[1], 255, lambda x: None)
        cv2.createTrackbar("Grey S Max", "HSV Sliders - Grey", self.grey_upper[1], 255, lambda x: None)
        cv2.createTrackbar("Grey V Min", "HSV Sliders - Grey", self.grey_lower[2], 255, lambda x: None)
        cv2.createTrackbar("Grey V Max", "HSV Sliders - Grey", self.grey_upper[2], 255, lambda x: None)
        
        # Green mold trackbars
        cv2.createTrackbar("Green H Min", "HSV Sliders - Green", self.green_lower[0], 179, lambda x: None)
        cv2.createTrackbar("Green H Max", "HSV Sliders - Green", self.green_upper[0], 179, lambda x: None)
        cv2.createTrackbar("Green S Min", "HSV Sliders - Green", self.green_lower[1], 255, lambda x: None)
        cv2.createTrackbar("Green S Max", "HSV Sliders - Green", self.green_upper[1], 255, lambda x: None)
        cv2.createTrackbar("Green V Min", "HSV Sliders - Green", self.green_lower[2], 255, lambda x: None)
        cv2.createTrackbar("Green V Max", "HSV Sliders - Green", self.green_upper[2], 255, lambda x: None)
        
    def get_trackbar_values(self):
        """Read current values from trackbars"""
        # Grey ranges
        grey_h_min = cv2.getTrackbarPos("Grey H Min", "HSV Sliders - Grey")
        grey_h_max = cv2.getTrackbarPos("Grey H Max", "HSV Sliders - Grey")
        grey_s_min = cv2.getTrackbarPos("Grey S Min", "HSV Sliders - Grey")
        grey_s_max = cv2.getTrackbarPos("Grey S Max", "HSV Sliders - Grey")
        grey_v_min = cv2.getTrackbarPos("Grey V Min", "HSV Sliders - Grey")
        grey_v_max = cv2.getTrackbarPos("Grey V Max", "HSV Sliders - Grey")
        
        # Green ranges
        green_h_min = cv2.getTrackbarPos("Green H Min", "HSV Sliders - Green")
        green_h_max = cv2.getTrackbarPos("Green H Max", "HSV Sliders - Green")
        green_s_min = cv2.getTrackbarPos("Green S Min", "HSV Sliders - Green")
        green_s_max = cv2.getTrackbarPos("Green S Max", "HSV Sliders - Green")
        green_v_min = cv2.getTrackbarPos("Green V Min", "HSV Sliders - Green")
        green_v_max = cv2.getTrackbarPos("Green V Max", "HSV Sliders - Green")
        
        return {
            'grey_lower': [grey_h_min, grey_s_min, grey_v_min],
            'grey_upper': [grey_h_max, grey_s_max, grey_v_max],
            'green_lower': [green_h_min, green_s_min, green_v_min],
            'green_upper': [green_h_max, green_s_max, green_v_max]
        }
    
    def reset_trackbars(self):
        """Reset trackbars to default values"""
        cv2.setTrackbarPos("Grey H Min", "HSV Sliders - Grey", self.grey_lower[0])
        cv2.setTrackbarPos("Grey H Max", "HSV Sliders - Grey", self.grey_upper[0])
        cv2.setTrackbarPos("Grey S Min", "HSV Sliders - Grey", self.grey_lower[1])
        cv2.setTrackbarPos("Grey S Max", "HSV Sliders - Grey", self.grey_upper[1])
        cv2.setTrackbarPos("Grey V Min", "HSV Sliders - Grey", self.grey_lower[2])
        cv2.setTrackbarPos("Grey V Max", "HSV Sliders - Grey", self.grey_upper[2])
        
        cv2.setTrackbarPos("Green H Min", "HSV Sliders - Green", self.green_lower[0])
        cv2.setTrackbarPos("Green H Max", "HSV Sliders - Green", self.green_upper[0])
        cv2.setTrackbarPos("Green S Min", "HSV Sliders - Green", self.green_lower[1])
        cv2.setTrackbarPos("Green S Max", "HSV Sliders - Green", self.green_upper[1])
        cv2.setTrackbarPos("Green V Min", "HSV Sliders - Green", self.green_lower[2])
        cv2.setTrackbarPos("Green V Max", "HSV Sliders - Green", self.green_upper[2])
        
        self.get_logger().info("Reset to default values")
        
    def save_values(self):
        """Save current HSV values to file"""
        values = self.get_trackbar_values()
        
        # Save to JSON file
        config_path = Path.home() / 'hsv_calibration.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(values, f, indent=4)
            self.get_logger().info(f"Saved calibration to {config_path}")
            
            # Also print values for easy copy-paste into detection.py
            print("\n" + "="*60)
            print("Copy these values to your detection.py:")
            print("="*60)
            print(f"self.grey_lower = np.array({values['grey_lower']})")
            print(f"self.grey_upper = np.array({values['grey_upper']})")
            print(f"self.green_lower = np.array({values['green_lower']})")
            print(f"self.green_upper = np.array({values['green_upper']})")
            print("="*60 + "\n")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save: {e}")
    
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Process and display
            self.process_frame()
            
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")
    
    def process_frame(self):
        """Apply HSV filtering and show results"""
        if self.current_frame is None:
            return
        
        # Get current trackbar values
        values = self.get_trackbar_values()
        
        # Convert to HSV
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        
        # Create masks
        grey_lower = np.array(values['grey_lower'])
        grey_upper = np.array(values['grey_upper'])
        green_lower = np.array(values['green_lower'])
        green_upper = np.array(values['green_upper'])
        
        grey_mask = cv2.inRange(hsv, grey_lower, grey_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((3, 3), np.uint8)
        grey_mask_clean = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel)
        grey_mask_clean = cv2.morphologyEx(grey_mask_clean, cv2.MORPH_OPEN, kernel)
        
        green_mask_clean = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask_clean = cv2.morphologyEx(green_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Find contours for grey fruits
        contours_grey, _ = cv2.findContours(grey_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create visualization
        combined = self.current_frame.copy()
        detection_count = 0
        
        # Process each grey fruit contour
        for contour in contours_grey:
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum area threshold
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check for green mold in this region
            roi_green = green_mask_clean[y:y+h, x:x+w]
            green_pixels = cv2.countNonZero(roi_green)
            
            # If has both grey body and green mold, it's a bad fruit
            if green_pixels > 50:  # Minimum green pixels
                detection_count += 1
                
                # Draw on combined image
                cv2.rectangle(combined, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(combined, f"Bad Fruit {detection_count}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(combined, f"Area: {int(area)}", (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(combined, f"Green: {green_pixels}", (x, y+h+40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add info text
        info_text = [
            f"Detections: {detection_count}",
            f"Grey HSV: [{values['grey_lower'][0]},{values['grey_lower'][1]},{values['grey_lower'][2]}] - [{values['grey_upper'][0]},{values['grey_upper'][1]},{values['grey_upper'][2]}]",
            f"Green HSV: [{values['green_lower'][0]},{values['green_lower'][1]},{values['green_lower'][2]}] - [{values['green_upper'][0]},{values['green_upper'][1]},{values['green_upper'][2]}]",
            "Press 's' to save | 'r' to reset | ESC to quit"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(combined, text, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show all windows
        cv2.imshow("Original Image", self.current_frame)
        cv2.imshow("Grey Mask", grey_mask_clean)
        cv2.imshow("Green Mask", green_mask_clean)
        cv2.imshow("Combined Detection", combined)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.get_logger().info("Exiting...")
            rclpy.shutdown()
        elif key == ord('s'):
            self.save_values()
        elif key == ord('r'):
            self.reset_trackbars()


def main(args=None):
    rclpy.init(args=args)
    node = HSVCalibrationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
