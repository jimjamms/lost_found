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
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
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
        
        self.latest_depth_msg = None
        self.camera_intrinsics = None
        self.latest_map = None
        
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
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 4. Threads & Timers
        self.create_timer(5.0, self.wander_logic)
        self.create_timer(1.0, self.watchdog_logic) 
        threading.Thread(target=self.keyboard_logic_loop, daemon=True).start()

        self.get_logger().info("OBSTACLE-AWARE SCOUT ONLINE: Scanning costmap to avoid collisions.")

    def map_cb(self, msg):
        self.latest_map = msg

    def is_location_clear(self, x, y):
        """Checks if a coordinate is free of obstacles using the costmap."""
        if self.latest_map is None:
            return True # Proceed if no map yet, Navigation Stack will still try to avoid
        
        # Convert map coordinates to grid indices
        origin_x = self.latest_map.info.origin.position.x
        origin_y = self.latest_map.info.origin.position.y
        res = self.latest_map.info.resolution
        
        grid_x = int((x - origin_x) / res)
        grid_y = int((y - origin_y) / res)
        
        if 0 <= grid_x < self.latest_map.info.width and 0 <= grid_y < self.latest_map.info.height:
            index = grid_y * self.latest_map.info.width + grid_x
            cost = self.latest_map.data[index]
            # Cost > 50 usually means occupied or near an obstacle in Nav2
            return cost < 50
        return False

    def watchdog_logic(self):
        if not self.is_navigating:
            self.last_activity_time = time.time()
            return
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            curr_pos = (t.transform.translation.x, t.transform.translation.y)
            curr_rot = (t.transform.rotation.z, t.transform.rotation.w)

            if self.state == "GUIDING":
                target = self.object_memory[self.target_label][self.target_index]
                if math.hypot(curr_pos[0] - target[0], curr_pos[1] - target[1]) < 1.0:
                    if self.goal_handle: self.goal_handle.cancel_goal_async()
                    self.is_navigating = False
                    self.face_object_and_prompt(curr_pos, target)
                    return

            if self.last_pos is not None:
                if math.hypot(curr_pos[0]-self.last_pos[0], curr_pos[1]-self.last_pos[1]) > 0.05 or abs(curr_rot[0]-self.last_rot[0]) > 0.02:
                    self.last_activity_time = time.time()

            if time.time() - self.last_activity_time > 5.0:
                self.handle_stall(curr_pos)

            self.last_pos, self.last_rot = curr_pos, curr_rot
        except Exception: pass

    def handle_stall(self, curr_pos):
        if self.goal_handle: self.goal_handle.cancel_goal_async()
        self.is_navigating = False
        self.last_activity_time = time.time()

        if self.state == "WANDERING":
            self.get_logger().warn("WANDER STALL: Obstacle detected or stuck. Picking new clear waypoint.")
        elif self.state == "GUIDING":
            self.get_logger().warn("GUIDE STALL: Stuck. Moving to open area...")
            # Try to find a clear spot to move to first
            for _ in range(10):
                angle = random.uniform(0, 2*math.pi)
                safe_x = curr_pos[0] + (1.2 * math.cos(angle))
                safe_y = curr_pos[1] + (1.2 * math.sin(angle))
                if self.is_location_clear(safe_x, safe_y):
                    self.send_nav_goal(safe_x, safe_y)
                    break
            threading.Timer(5.0, self.start_guidance).start()

    def wander_logic(self):
        if self.state != "WANDERING" or self.is_navigating: return
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            curr = (t.transform.translation.x, t.transform.translation.y)
            if self.home_pose is None: self.home_pose = curr

            # Try up to 15 times to find a coordinate that isn't inside a wall/obstacle
            for _ in range(15):
                new_angle = (self.last_angle + random.uniform(math.pi/2, 3*math.pi/2)) % (2*math.pi)
                dist = random.uniform(1.2, 3.0) 
                tx = curr[0] + (dist * math.cos(new_angle))
                ty = curr[1] + (dist * math.sin(new_angle))

                # Global boundary clamp
                if math.hypot(tx - self.home_pose[0], ty - self.home_pose[1]) > 3.0:
                    angle_h = math.atan2(ty - self.home_pose[1], tx - self.home_pose[0])
                    tx, ty = self.home_pose[0] + 3.0*math.cos(angle_h), self.home_pose[1] + 3.0*math.sin(angle_h)

                if self.is_location_clear(tx, ty):
                    self.last_angle = new_angle
                    self.get_logger().info(f"NAV: Clear path found. Moving to [{tx:.1f}, {ty:.1f}]")
                    self.send_nav_goal(tx, ty)
                    return
            
            self.get_logger().warn("NAV: Could not find clear waypoint in 15 tries. Retrying soon...")
        except: pass

    def face_object_and_prompt(self, curr_pos, target):
        angle_to_obj = math.atan2(target[1] - curr_pos[1], target[0] - curr_pos[0])
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(curr_pos[0]), float(curr_pos[1])
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(angle_to_obj/2.0), math.cos(angle_to_obj/2.0)
        self.nav_client.send_goal_async(goal).add_done_callback(lambda _: threading.Timer(1.5, self.trigger_arrival_prompt).start())

    def trigger_arrival_prompt(self):
        total = len(self.object_memory[self.target_label])
        print(f"\n[GUIDE] Facing {self.target_label} (Object {self.target_index + 1} of {total})")
        ans = input("Is this the correct item? (yes/no/nevermind): ").lower().strip()
        if ans == 'nevermind' or 'yes' in ans or 'y' == ans: self.state = "WANDERING"
        else: 
            self.target_index += 1
            self.start_guidance()

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
                try:
                    t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                    self.face_object_and_prompt((t.transform.translation.x, t.transform.translation.y), self.object_memory[self.target_label][self.target_index])
                except: self.trigger_arrival_prompt()

    def rgb_cb(self, msg):
        try:
            if msg.encoding == "bgra8":
                img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                cv_img = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, 'passthrough') if self.latest_depth_msg else None
        except: return
        if self.state == "WANDERING":
            results = self.model(cv_img, stream=True, conf=0.45, verbose=False)
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
                                if loc: self.save_to_map(label, loc)
        cv2.imshow("Kinect AI Feed", cv_img)
        cv2.waitKey(1)

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
            if math.hypot(inst[0]-loc[0], inst[1]-loc[1]) < 0.6: return
        self.object_memory[label].append([loc[0], loc[1]])
        self.get_logger().info(f"!!! MEMORY: Found {label} !!!")

    def keyboard_logic_loop(self):
        while rclpy.ok():
            if self.state == "WANDERING":
                input("\n>>> [WANDERING] Press ENTER to search.")
                if self.goal_handle: self.goal_handle.cancel_goal_async()
                self.is_navigating = False
                self.state = "MENU"; self.handle_menu()
            time.sleep(0.1)

    def handle_menu(self):
        while True:
            seen = {k: len(v) for k, v in self.object_memory.items() if len(v) > 0}
            print(f"\n--- SEARCH MENU ---")
            if not seen: print("Memory empty."); return
            for obj, count in seen.items(): print(f"- {obj}: {count} instance(s)")
            choice = input("\nTarget name (or 'nevermind'): ").strip().lower()
            if choice == 'nevermind': self.state = "WANDERING"; return
            if choice in seen: 
                self.target_label, self.target_index = choice, 0
                self.start_guidance(); return

    def start_guidance(self):
        locs = self.object_memory.get(self.target_label, [])
        if self.target_index < len(locs): 
            self.state = "GUIDING"
            self.send_nav_goal(locs[self.target_index][0], locs[self.target_index][1])
        else:
            print(f"\n[!] Ran out of {self.target_label} instances in memory.")
            self.handle_menu()

    def info_cb(self, msg): self.camera_intrinsics = msg
    def depth_cb(self, msg): self.latest_depth_msg = msg

def main():
    rclpy.init(); node = KeyboardScout()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()