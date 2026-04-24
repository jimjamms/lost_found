import os
import threading
import time
import math
import random
import numpy as np
import cv2

# Force CPU mode for PyTorch to prevent NVIDIA driver crashes
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PointStamped
from nav2_msgs.action import NavigateToPose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class KeyboardScout(Node):
    def __init__(self):
        super().__init__('keyboard_scout_node')
        
        # 1. Vision & Memory
        self.model = YOLO('yolov8n.pt')
        self.model.to('cpu')
        self.bridge = CvBridge()
        self.target_list = ['backpack', 'cup', 'bottle', 'person']
        self.object_memory = {item: [] for item in self.target_list}
        self.FORGET_THRESHOLD = 20
        
        self.latest_depth_msg = None
        self.camera_intrinsics = None
        
        # 2. States & Stall Monitor
        self.state = "WANDERING"
        self.target_label = None
        self.target_index = 0
        self.home_pose = None
        self.is_navigating = False
        self.is_scanning = False
        
        # Stall tracking
        self.last_pos = None
        self.last_rot = None
        self.last_activity_time = time.time()
        self.goal_handle = None 

        # 3. ROS Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        video_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.create_subscription(CameraInfo, '/rgb/camera_info', self.info_cb, video_qos)
        self.create_subscription(Image, '/depth_to_rgb/image_raw', self.depth_cb, video_qos)
        self.create_subscription(Image, '/rgb/image_raw', self.rgb_cb, video_qos)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 4. Threads & Timers
        self.create_timer(5.0, self.wander_logic)
        self.create_timer(1.0, self.check_if_stalled) 
        threading.Thread(target=self.keyboard_logic_loop, daemon=True).start()

        self.get_logger().info("WIDE PATROL ONLINE: Wandering within 3.0m of home.")

    def check_if_stalled(self):
        if not self.is_navigating or self.is_scanning or self.state != "WANDERING":
            self.last_activity_time = time.time()
            return
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            curr_pos = (t.transform.translation.x, t.transform.translation.y)
            curr_rot = (t.transform.rotation.z, t.transform.rotation.w)
            if self.last_pos is not None and self.last_rot is not None:
                dist = math.hypot(curr_pos[0] - self.last_pos[0], curr_pos[1] - self.last_pos[1])
                rot_diff = abs(curr_rot[0] - self.last_rot[0])
                if dist > 0.05 or rot_diff > 0.02:
                    self.last_activity_time = time.time()
            self.last_pos, self.last_rot = curr_pos, curr_rot
            if time.time() - self.last_activity_time > 5.0:
                self.get_logger().warn("WANDER STALL: Robot stuck. Re-routing...")
                if self.goal_handle: self.goal_handle.cancel_goal_async()
                self.is_navigating = False
                self.last_activity_time = time.time()
        except: pass

    def rgb_cb(self, msg):
        try:
            if msg.encoding == "bgra8":
                img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                cv_img = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, 'passthrough') if self.latest_depth_msg else None
        except: return
        results = self.model(cv_img, stream=True, conf=0.45, verbose=False)
        frame_detections = []
        for r in results:
            for box in r.boxes:
                label = self.model.names[int(box.cls[0])]
                if label in self.target_list:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.rectangle(cv_img, (x1,y1), (x2,y2), (0,255,0), 2)
                    if depth_img is not None and self.camera_intrinsics:
                        z = float(depth_img[cy, cx])
                        z_m = z / 1000.0 if z > 20 else z 
                        if 0.3 < z_m < 5.0:
                            loc = self.get_map_coords(cx, cy, z_m, msg.header)
                            if loc:
                                frame_detections.append((label, (x1, y1, x2, y2)))
                                self.save_to_map(label, loc)
        self.verify_visibility(msg.header, frame_detections)
        cv2.imshow("Kinect AI Feed", cv_img)
        cv2.waitKey(1)

    def verify_visibility(self, header, frame_detections):
        if not self.camera_intrinsics: return
        for label, instances in self.object_memory.items():
            to_remove = []
            for i, data in enumerate(instances):
                try:
                    p = PointStamped()
                    p.header.frame_id = 'map'
                    p.point.x, p.point.y = data[0], data[1]
                    p_cam = self.tf_buffer.transform(p, header.frame_id, timeout=rclpy.duration.Duration(seconds=0.05))
                    if 0.5 < p_cam.point.z < 4.5:
                        fx, cx_p, fy, cy_p = self.camera_intrinsics.k[0], self.camera_intrinsics.k[2], self.camera_intrinsics.k[4], self.camera_intrinsics.k[5]
                        u, v = int((p_cam.point.x * fx / p_cam.point.z) + cx_p), int((p_cam.point.y * fy / p_cam.point.z) + cy_p)
                        if 100 < u < (self.camera_intrinsics.width - 100) and 100 < v < (self.camera_intrinsics.height - 100):
                            is_there = any((d_label == label and x1 < u < x2 and y1 < v < y2) for d_label, (x1, y1, x2, y2) in frame_detections)
                            if not is_there: data[2] += 1
                            else: data[2] = 0
                            if data[2] >= self.FORGET_THRESHOLD: to_remove.append(i)
                except: continue
            for idx in sorted(to_remove, reverse=True):
                self.get_logger().warn(f"FORGETTING: {label} removed (disappeared from view).")
                instances.pop(idx)

    def wander_logic(self):
        """Wanders within a strict 3.0m radius of the start position."""
        if self.state != "WANDERING" or self.is_navigating or self.is_scanning: return
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            curr = (t.transform.translation.x, t.transform.translation.y)
            if self.home_pose is None:
                self.home_pose = curr
                self.get_logger().info(f"HOME BASE set at: {curr[0]:.2f}, {curr[1]:.2f}")

            # Pick a random point within 3.0 meters of HOME BASE
            radius = 3.0
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0.5, radius)
            
            tx = self.home_pose[0] + (dist * math.cos(angle))
            ty = self.home_pose[1] + (dist * math.sin(angle))
            
            self.get_logger().info(f"NAV: Wandering to wide waypoint [{tx:.1f}, {ty:.1f}]")
            self.send_nav_goal(tx, ty)
        except: pass

    def perform_360_scan(self, x, y):
        self.is_scanning = True
        self.get_logger().info("SCAN: Starting 360 degree rotation.")
        angles = [0.0, 1.57, 3.14, 4.71, 0.0]
        def next_spin(idx):
            if idx >= len(angles): 
                self.get_logger().info("SCAN: Rotation complete.")
                self.is_scanning = False; return
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'
            goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
            goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(angles[idx]/2), math.cos(angles[idx]/2)
            self.nav_client.send_goal_async(goal).add_done_callback(lambda f: f.result().get_result_async().add_done_callback(lambda _: next_spin(idx+1)))
        next_spin(0)

    def send_nav_goal(self, x, y):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x, goal.pose.pose.position.y, goal.pose.pose.orientation.w = float(x), float(y), 1.0
        self.is_navigating = True
        self.last_activity_time = time.time()
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal).add_done_callback(self.goal_resp_cb)

    def goal_resp_cb(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted: 
            self.get_logger().error("NAV: Goal rejected.")
            self.is_navigating = False; return
        self.goal_handle.get_result_async().add_done_callback(self.nav_done_cb)

    def nav_done_cb(self, future):
        status = future.result().status
        self.is_navigating = False
        if status == 4:
            self.get_logger().info("NAV: Waypoint reached.")
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            if self.state == "WANDERING": self.perform_360_scan(t.transform.translation.x, t.transform.translation.y)
            elif self.state == "GUIDING":
                print(f"\n[GUIDE] Arrived at saved {self.target_label} location.")
                ans = input(f"Is {self.target_label} here? (yes/no/nevermind): ").lower().strip()
                if ans == 'nevermind' or 'y' in ans: 
                    self.get_logger().info("GUIDE: Finished. Returning to wander.")
                    self.state = "WANDERING"
                else: 
                    if self.target_index < len(self.object_memory[self.target_label]): 
                        self.get_logger().warn(f"GUIDE: Target missing from coordinate.")
                        self.object_memory[self.target_label].pop(self.target_index)
                    self.start_guidance()

    def get_map_coords(self, cx, cy, z, header):
        fx, cx_p, fy, cy_p = self.camera_intrinsics.k[0], self.camera_intrinsics.k[2], self.camera_intrinsics.k[4], self.camera_intrinsics.k[5]
        p = PointStamped()
        p.header = header
        p.point.x, p.point.y, p.point.z = (cx - cx_p)*z/fx, (cy - cy_p)*z/fy, z
        try:
            res = self.tf_buffer.transform(p, 'map', timeout=rclpy.duration.Duration(seconds=0.05))
            return (res.point.x, res.point.y)
        except: return None

    def save_to_map(self, label, loc):
        for inst in self.object_memory[label]:
            if math.hypot(inst[0]-loc[0], inst[1]-loc[1]) < 1.2:
                inst[2] = 0
                return
        self.object_memory[label].append([loc[0], loc[1], 0])
        self.get_logger().info(f"!!! NEW TARGET DETECTED: {label} !!!")

    def keyboard_logic_loop(self):
        while rclpy.ok():
            if self.state == "WANDERING":
                input("\n>>> [IDLE] Press ENTER to search."); 
                self.state = "MENU"; self.handle_menu()
            time.sleep(0.1)

    def handle_menu(self):
        while True:
            seen = {k: len(v) for k, v in self.object_memory.items() if len(v) > 0}
            print(f"\n--- SEARCH MENU ---")
            if not seen: print("Memory empty.")
            else:
                for obj, count in seen.items(): print(f"- {obj}: {count} found")
            choice = input("\nTarget name (or 'nevermind' to resume): ").strip().lower()
            if choice == 'nevermind': 
                self.get_logger().info("MENU: Resuming wander.")
                self.state = "WANDERING"; return
            if choice in seen: 
                self.target_label, self.target_index = choice, 0
                self.get_logger().info(f"GUIDE: Heading to {choice}.")
                self.start_guidance(); return

    def start_guidance(self):
        locs = self.object_memory.get(self.target_label, [])
        if self.target_index < len(locs): 
            self.state = "GUIDING"
            self.send_nav_goal(locs[self.target_index][0], locs[self.target_index][1])
        else: self.handle_menu()

    def info_cb(self, msg): self.camera_intrinsics = msg
    def depth_cb(self, msg): self.latest_depth_msg = msg

def main():
    rclpy.init(); node = KeyboardScout()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()