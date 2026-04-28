import os
import threading
import time
import math
import random
import numpy as np
import cv2

# Force CPU mode for PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from ultralytics import YOLO
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PointStamped
from nav2_msgs.action import NavigateToPose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from voice_query import VoiceIO, extract_item, format_answer, is_cancel_query, wants_guidance

class KeyboardScout(Node):
    def __init__(self, user_radius):
        super().__init__('keyboard_scout_node')
        
        self.patrol_radius = user_radius
        
        # 1. Vision & Memory
        self.model = YOLO('yolov8n.pt')
        self.model.to('cpu')
        self.bridge = CvBridge()
        self.target_list = ['backpack', 'cup', 'bottle', 'person']
        self.object_memory = {item: [] for item in self.target_list}
        self.voice = VoiceIO(enabled=True)
        
        self.latest_depth_msg = None
        self.camera_intrinsics = None
        self.latest_map = None
        self.front_obstacle_detected = False
        
        # 2. States & Navigation
        self.state = "WANDERING"
        self.target_label = None
        self.target_index = 0
        self.home_pose = None
        self.is_navigating = False
        self.last_angle = 0.0 
        
        # Stall tracking
        self.last_pos = None
        self.last_rot = None
        self.last_activity_time = time.time()
        self.goal_handle = None 

        # 3. ROS Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        video_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.create_subscription(CameraInfo, '/rgb/camera_info', self.info_cb, video_qos)
        self.create_subscription(Image, '/depth_to_rgb/image_raw', self.depth_cb, video_qos)
        self.create_subscription(Image, '/rgb/image_raw', self.rgb_cb, video_qos)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.map_cb, map_qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, video_qos)
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 4. Threads & Timers
        self.create_timer(5.0, self.wander_logic)
        self.create_timer(1.0, self.watchdog_logic) 
        threading.Thread(target=self.keyboard_logic_loop, daemon=True).start()

        self.get_logger().info(f"SCOUT ONLINE: Voice query enabled for radius {self.patrol_radius}m.")

    def scan_cb(self, msg):
        num_readings = len(msg.ranges)
        center_idx, cone_width = num_readings // 2, num_readings // 4 
        start_idx, end_idx = center_idx - (cone_width // 2), center_idx + (cone_width // 2)
        front_ranges = [r for r in msg.ranges[start_idx:end_idx] if msg.range_min < r < msg.range_max]
        self.front_obstacle_detected = True if (front_ranges and min(front_ranges) < 0.8) else False

    def watchdog_logic(self):
        if not self.is_navigating:
            self.last_activity_time = time.time()
            return
            
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            curr_pos = (t.transform.translation.x, t.transform.translation.y)
            curr_rot = (t.transform.rotation.z, t.transform.rotation.w)

            # Check if we are close enough to the item to stop and turn
            if self.state == "GUIDING":
                target = self.object_memory[self.target_label][self.target_index]
                if math.hypot(curr_pos[0] - target[0], curr_pos[1] - target[1]) < 1.0:
                    self.get_logger().info("PROXIMITY: Target near. Stopping to face the object.")
                    if self.goal_handle: self.goal_handle.cancel_goal_async()
                    self.is_navigating = False
                    self.face_item_then_prompt(curr_pos, target)
                    return

            if self.front_obstacle_detected and self.state == "WANDERING":
                if self.goal_handle: self.goal_handle.cancel_goal_async()
                self.is_navigating = False
                return

            if self.last_pos is not None:
                if math.hypot(curr_pos[0]-self.last_pos[0], curr_pos[1]-self.last_pos[1]) > 0.05 or abs(curr_rot[0]-self.last_rot[0]) > 0.02:
                    self.last_activity_time = time.time()

            if time.time() - self.last_activity_time > 5.0:
                self.handle_stall(curr_pos)
            self.last_pos, self.last_rot = curr_pos, curr_rot
        except: pass

    def handle_stall(self, curr_pos):
        if self.goal_handle: self.goal_handle.cancel_goal_async()
        self.is_navigating = False
        self.last_activity_time = time.time()
        if self.state == "GUIDING":
            angle = random.uniform(0, 2*math.pi)
            self.send_nav_goal(curr_pos[0] + 1.2*math.cos(angle), curr_pos[1] + 1.2*math.sin(angle))
            threading.Timer(5.0, self.start_guidance).start()

    def face_item_then_prompt(self, curr_pos, target_pos):
        """Calculates the orientation to face the item and sends a rotation goal."""
        angle_to_target = math.atan2(target_pos[1] - curr_pos[1], target_pos[0] - curr_pos[0])
        
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = float(curr_pos[0])
        goal.pose.pose.position.y = float(curr_pos[1])
        # Convert yaw to quaternion
        goal.pose.pose.orientation.z = math.sin(angle_to_target / 2.0)
        goal.pose.pose.orientation.w = math.cos(angle_to_target / 2.0)
        
        self.nav_client.wait_for_server()
        self.get_logger().info(f"GUIDE: Rotating to face the {self.target_label}...")
        self.nav_client.send_goal_async(goal).add_done_callback(
            lambda _: threading.Timer(1.5, self.trigger_arrival_prompt).start()
        )

    def trigger_arrival_prompt(self):
        total = len(self.object_memory[self.target_label])
        print(f"\n[GUIDE] Arrived at {self.target_label} (Object {self.target_index+1} of {total})")
        ans = input("Is this correct? (yes/no/nevermind): ").lower().strip()
        if 'y' in ans or 'never' in ans: self.state = "WANDERING"
        else: self.target_index += 1; self.start_guidance()

    def wander_logic(self):
        if self.state != "WANDERING" or self.is_navigating: return
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            curr = (t.transform.translation.x, t.transform.translation.y)
            if self.home_pose is None: self.home_pose = curr
            for _ in range(15):
                new_angle = (self.last_angle + random.uniform(math.pi/2, 3*math.pi/2)) % (2*math.pi)
                dist = random.uniform(1.2, self.patrol_radius) 
                tx, ty = curr[0] + dist*math.cos(new_angle), curr[1] + dist*math.sin(new_angle)
                if math.hypot(tx - self.home_pose[0], ty - self.home_pose[1]) > self.patrol_radius:
                    ang = math.atan2(ty-self.home_pose[1], tx-self.home_pose[0])
                    tx, ty = self.home_pose[0] + self.patrol_radius*math.cos(ang), self.home_pose[1] + self.patrol_radius*math.sin(ang)
                if self.is_location_clear(tx, ty):
                    self.last_angle = new_angle
                    self.send_nav_goal(tx, ty); return
        except: pass

    def is_location_clear(self, x, y):
        if self.latest_map is None: return True
        inf = self.latest_map.info
        gx, gy = int((x - inf.origin.position.x) / inf.resolution), int((y - inf.origin.position.y) / inf.resolution)
        if 0 <= gx < inf.width and 0 <= gy < inf.height:
            return self.latest_map.data[gy * inf.width + gx] < 50
        return False

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
        if not self.goal_handle.accepted: self.is_navigating = False; return
        self.goal_handle.get_result_async().add_done_callback(self.nav_done_cb)

    def nav_done_cb(self, future):
        if future.result().status == 4:
            self.is_navigating = False
            if self.state == "GUIDING":
                # Manual fallback if proximity didn't trigger
                try:
                    t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                    self.face_item_then_prompt((t.transform.translation.x, t.transform.translation.y), self.object_memory[self.target_label][self.target_index])
                except: self.trigger_arrival_prompt()

    def rgb_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, 'passthrough') if self.latest_depth_msg else None
        except: return
        if self.state == "WANDERING":
            results = self.model(img, stream=True, conf=0.45, verbose=False)
            for r in results:
                for box in r.boxes:
                    label = self.model.names[int(box.cls[0])]
                    if label in self.target_list:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if depth is not None and self.camera_intrinsics:
                            z = float(depth[cy, cx]) / 1000.0 if float(depth[cy, cx]) > 20 else float(depth[cy, cx])
                            if 0.3 < z < 5.0:
                                loc = self.get_map_coords(cx, cy, z, msg.header)
                                if loc: self.save_to_map(label, loc)
        cv2.imshow("Kinect AI Feed", img); cv2.waitKey(1)

    def get_map_coords(self, cx, cy, z, header):
        if not self.camera_intrinsics: return None
        fx, cx_p, fy, cy_p = self.camera_intrinsics.k[0], self.camera_intrinsics.k[2], self.camera_intrinsics.k[4], self.camera_intrinsics.k[5]
        p = PointStamped()
        p.header = header
        p.point.x, p.point.y, p.point.z = (cx - cx_p)*z/fx, (cy - cy_p)*z/fy, float(z)
        try:
            res = self.tf_buffer.transform(p, 'map', timeout=rclpy.duration.Duration(seconds=0.05))
            return (res.point.x, res.point.y)
        except: return None

    def save_to_map(self, label, loc):
        for inst in self.object_memory[label]:
            if math.hypot(inst[0]-loc[0], inst[1]-loc[1]) < 0.6: return
        self.object_memory[label].append([loc[0], loc[1]])
        self.get_logger().info(f"!!! MEMORY: Found {label} !!!")

    def keyboard_logic_loop(self):
        while rclpy.ok():
            if self.state == "WANDERING":
                input("\n>>> [WANDERING] Press ENTER to search.")
                if self.goal_handle: self.goal_handle.cancel_goal_async()
                self.is_navigating = False; self.state = "MENU"; self.handle_menu()
            time.sleep(0.1)

    def handle_menu(self):
        while True:
            seen = {k: len(v) for k, v in self.object_memory.items() if len(v) > 0}
            print("\n--- VOICE SEARCH ---")
            if seen:
                for obj, count in seen.items():
                    print(f"- {obj}: {count} instance(s)")
            else:
                print("Memory empty.")

            query = self.voice.listen("Ask: 'Have you seen a bottle?' (or 'nevermind'): ").strip()
            if is_cancel_query(query):
                self.state = "WANDERING"
                return

            item = extract_item(query, self.target_list)
            locations = self.object_memory.get(item, []) if item else []
            self.voice.speak(format_answer(item, locations))

            if item not in seen:
                continue

            if wants_guidance(query):
                self.target_label, self.target_index = item, 0
                self.start_guidance()
                return

            guide_ans = self.voice.listen(f"Should I guide you to the {item}? (yes/no): ").strip().lower()
            if guide_ans.startswith('y'):
                self.target_label, self.target_index = item, 0
                self.start_guidance()
                return

    def start_guidance(self):
        locs = self.object_memory.get(self.target_label, [])
        if self.target_index < len(locs): self.state = "GUIDING"; self.send_nav_goal(locs[self.target_index][0], locs[self.target_index][1])
        else: print(f"\n[!] Ran out of {self.target_label} instances."); self.handle_menu()

    def map_cb(self, msg): self.latest_map = msg
    def info_cb(self, msg): self.camera_intrinsics = msg
    def depth_cb(self, msg): self.latest_depth_msg = msg

def main():
    try:
        val = input("Enter patrol radius in meters: ")
        user_radius = float(val)
    except:
        user_radius = 3.0
    rclpy.init()
    node = KeyboardScout(user_radius)
    try: rclpy.spin(node)
    except: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
