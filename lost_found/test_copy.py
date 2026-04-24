import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import math
import random
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PointStamped
from nav2_msgs.action import NavigateToPose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class AutonomousObjectScout(Node):
    def __init__(self):
        super().__init__('object_scout_node')
        
        # 1. Vision Setup
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()
        self.latest_depth_msg = None
        
        # --- TARGET FILTER ---
        self.target_list = {'backpack', 'cup', 'bottle', 'person'} 

        # 2. QoS Profile for Azure Kinect
        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 3. Navigation State
        self.is_navigating = False
        self.is_wandering = True 
        self.at_home = True
        self.home_pose = None

        # 4. TF2 Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 5. Camera Subscriptions
        self.camera_intrinsics = None
        self.create_subscription(CameraInfo, '/rgb/camera_info', self.info_cb, video_qos)
        self.create_subscription(Image, '/depth_to_rgb/image_raw', self.depth_cb, video_qos)
        self.create_subscription(Image, '/rgb/image_raw', self.rgb_cb, video_qos)

        # 6. Action Client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 7. Wander Timer (8s)
        self.create_timer(8.0, self.wander_logic)
        
        self.get_logger().info(f"AI SCOUT ONLINE. Target Whitelist: {self.target_list}")

    def info_cb(self, msg):
        self.camera_intrinsics = msg

    def depth_cb(self, msg):
        self.latest_depth_msg = msg

    def rgb_cb(self, msg):
        """Processes RGB frames and ONLY prints if object is in target_list."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except:
            return

        results = self.model(cv_image, stream=True, conf=0.45, verbose=False)

        for r in results:
            for box in r.boxes:
                label = self.model.names[int(box.cls[0])]
                
                # --- FILTERED DETECTION PRINT ---
                if label in self.target_list:
                    self.get_logger().info(f"!!! TARGET SPOTTED: {label} !!!")

                # Visual feedback on window
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if label in self.target_list else (255, 0, 0)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(cv_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Kinect AI Feed", cv_image)
        cv2.waitKey(1)

    def wander_logic(self):
        """Wander Loop (1.5m max) with active movement logging."""
        if self.is_navigating or not self.is_wandering:
            return

        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('map', 'base_link', now, timeout=rclpy.duration.Duration(seconds=0.1))
            curr_x, curr_y = trans.transform.translation.x, trans.transform.translation.y

            if self.home_pose is None:
                self.home_pose = (curr_x, curr_y)
                self.get_logger().info(f"HOME BASE set at: {curr_x:.2f}, {curr_y:.2f}")

            if self.at_home:
                # WANDER
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(0.8, 1.5)
                rx = self.home_pose[0] + (dist * math.cos(angle))
                ry = self.home_pose[1] + (dist * math.sin(angle))
                
                self.get_logger().info(f"NAV: Leaving Home to wander at [{rx:.1f}, {ry:.1f}]")
                self.send_nav_goal(rx, ry)
                self.at_home = False
            else:
                # RETURN HOME
                self.get_logger().info("NAV: Wander complete. Traveling back HOME...")
                self.send_nav_goal(self.home_pose[0], self.home_pose[1])
                self.at_home = True
        except Exception:
            self.get_logger().warn("NAV: Waiting for map localization...", throttle_duration_sec=10.0)

    def send_nav_goal(self, x, y):
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.w = 1.0 
        self.is_navigating = True
        self.nav_to_pose_client.send_goal_async(goal).add_done_callback(self.goal_cb)

    def goal_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("NAV: Goal rejected!")
            self.is_navigating = False
            return
        goal_handle.get_result_async().add_done_callback(self.done_cb)

    def done_cb(self, future):
        self.get_logger().info("NAV: Goal reached successfully.")
        self.is_navigating = False

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousObjectScout()
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