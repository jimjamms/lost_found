import os
import threading
import time
import math
import random
import numpy as np
import cv2
import re
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

ITEM_ALIASES = {
   "bottle": ("bottle", "water bottle", "drink bottle", "hydro flask", "flask"),
   "backpack": ("backpack", "back pack", "bag", "bookbag", "book bag", "school bag"),
   "cup": ("cup", "mug", "coffee cup"),
   "person": ("person", "human", "someone"),
   "umbrella": ("umbrella", "parasol"),
   "handbag": ("handbag", "purse"),
   "laptop": ("laptop", "computer", "macbook"),
   "cell phone": ("cell phone", "phone", "smartphone", "mobile")
}

GUIDE_WORDS = (
   "guide", "take me", "show me", "bring me", "lead me", "go to",
   "find it", "where is", "where's", "take me to"
)
YES_WORDS = ("yes", "y", "yeah", "yep", "sure", "ok", "okay", "please")
CANCEL_WORDS = ("nevermind", "never mind", "cancel", "stop", "resume")
DORA_WAKE_WORDS = ("hey dora", "hello dora", "dora")
WAKE_WORDS = DORA_WAKE_WORDS + ("hey robot", "hello robot", "robot", "scout")

def normalize_text(text):
   return re.sub(r"\s+", " ", text.lower()).strip()

def extract_item(query, targets):
   query = normalize_text(query)
   target_set = set(targets)
   for item, aliases in ITEM_ALIASES.items():
       if item not in target_set: continue
       for alias in aliases:
           if re.search(rf"\b{re.escape(alias)}s?\b", query): return item
   for item in targets:
       if re.search(rf"\b{re.escape(item)}s?\b", query): return item
   return None

def is_cancel_query(query):
   query = normalize_text(query)
   return any(word in query for word in CANCEL_WORDS)

def wants_guidance(query):
   query = normalize_text(query)
   return any(word in query for word in GUIDE_WORDS)

def is_yes_query(query):
   query = normalize_text(query)
   return any(re.search(rf"\b{re.escape(word)}\b", query) for word in YES_WORDS)

def strip_wake_words(query):
   query = normalize_text(query)
   for word in WAKE_WORDS:
       query = re.sub(rf"^\s*{re.escape(word)}[,\s]*", "", query).strip()
   return query

def has_wake_word(query):
   query = normalize_text(query)
   return any(re.search(rf"^\s*{re.escape(word)}\b", query) for word in WAKE_WORDS)

def format_location(location):
   if isinstance(location, str): return location
   if isinstance(location, (list, tuple)) and len(location) >= 2:
       return f"coordinates X {location[0]:.1f}, Y {location[1]:.1f}"
   return "the saved location"

class VoiceIO:
   def __init__(self, enabled=True):
       self.enabled = enabled
       self.recognizer = None
       self.microphone = None
       self.tts_engine = None
       self.last_error = "None"
       self.last_heard = ""
       self.voice_phase = "Idle"
       self.current_prompt = ""
       if not enabled: return
       try:
           import speech_recognition as sr
           self.recognizer = sr.Recognizer()
           self.recognizer.pause_threshold = 1.4
           self.recognizer.phrase_threshold = 0.25
           self.recognizer.non_speaking_duration = 0.6
           self.recognizer.dynamic_energy_threshold = True
           self.microphone = sr.Microphone()
           print("[VOICE] Speech recognition initialized.")
       except Exception as exc:
           self.recognizer, self.microphone = None, None
           print(f"[VOICE] Speech recognition unavailable: {type(exc).__name__}: {exc}")
       try:
           import pyttsx3
           self.tts_engine = pyttsx3.init()
           self.tts_engine.setProperty("rate", 175)
           self.select_female_voice()
           print("[VOICE] Text-to-speech initialized.")
       except Exception as exc:
           self.tts_engine = None
           print(f"[VOICE] Text-to-speech unavailable: {type(exc).__name__}: {exc}")

   def select_female_voice(self):
       if not self.tts_engine:
           return
       try:
           voices = self.tts_engine.getProperty("voices")
       except Exception as exc:
           print(f"[VOICE] Could not list voices: {type(exc).__name__}: {exc}")
           return
       preferred_names = (
           "samantha", "victoria", "karen", "moira", "tessa", "veena",
           "zira", "hazel", "susan", "female"
       )
       for voice in voices:
           voice_text = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
           if any(name in voice_text for name in preferred_names):
               self.tts_engine.setProperty("voice", voice.id)
               print(f"[VOICE] Using voice: {getattr(voice, 'name', voice.id)}")
               return
       print("[VOICE] No clearly female voice found. Using system default voice.")

   def listen(self, prompt="Ask me about an item: ", typed_fallback=True, phrase_time_limit=8):
       self.current_prompt = prompt
       if self.recognizer and self.microphone:
           self.voice_phase = "Preparing microphone"
           print(f"\n[VOICE] {prompt} (Listening...)")
           try:
               with self.microphone as source:
                   self.voice_phase = "Calibrating microphone"
                   print("[VOICE] Calibrating microphone noise...")
                   self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
                   self.voice_phase = "Listening now"
                   print("[VOICE] Listening now.")
                   audio = self.recognizer.listen(source, timeout=6, phrase_time_limit=phrase_time_limit)
               self.voice_phase = "Recognizing speech"
               print("[VOICE] Audio captured. Sending to Google recognizer...")
               text = self.recognizer.recognize_google(audio)
               print(f"[VOICE] Heard: {text}")
               self.last_error = "None"
               self.last_heard = text
               self.voice_phase = "Speech recognized"
               self.current_prompt = ""
               return text
           except Exception as exc:
               self.last_error = f"{type(exc).__name__}: {exc}"
               self.voice_phase = "Speech failed"
               print(f"[VOICE] Speech failed: {type(exc).__name__}: {exc}")
               if typed_fallback:
                   self.voice_phase = "Typing fallback"
                   print("[VOICE] Switching to typing.")
               else:
                   print("[VOICE] No speech command heard.")
                   self.current_prompt = ""
                   return ""
       if typed_fallback:
           typed = input(prompt)
           self.last_heard = typed
           self.voice_phase = "Typed input received"
           self.current_prompt = ""
           return typed
       self.current_prompt = ""
       return ""

   def beep(self, count=1):
       for _ in range(count):
           print("\a", end="", flush=True)
           time.sleep(0.08)

   def speak(self, message):
       self.voice_phase = "Speaking"
       print(f"\nRobot: {message}")
       if self.tts_engine:
           try:
               self.tts_engine.say(message)
               self.tts_engine.runAndWait()
           except Exception as exc:
               print(f"[VOICE] Text-to-speech failed: {type(exc).__name__}: {exc}")
       self.voice_phase = "Idle"



class VoiceScout(Node):
   def __init__(self, user_radius):
       super().__init__('voice_scout_node')
      
       self.patrol_radius = user_radius
       self.voice = VoiceIO(enabled=True)
      
       # Vision & Memory
       self.model = YOLO('yolov8n.pt')
       self.model.to('cpu')
       self.bridge = CvBridge()
       self.target_list = ['backpack', 'cup', 'bottle', 'umbrella', 'handbag', 'laptop', 'cell phone']
       self.object_memory = {item: [] for item in self.target_list}
       self.memory_strikes = {}
      
       self.latest_depth_msg = None
       self.camera_intrinsics = None
       self.latest_map = None
       self.front_obstacle_detected = False
       self.obstacle_hit_count = 0
       self.obstacle_clear_count = 0
       self.stopped_for_obstacle = False
       self.obstacle_stop_time = 0.0
       self.obstacle_resume_delay = 1.2
      
       # States & Navigation
       self.state = "WANDERING"
       self.target_label = None
       self.target_index = 0
       self.home_pose = None
       self.is_navigating = False
       self.last_angle = 0.0
       self.current_wander_target = (0.0, 0.0)
       self.status_message = "Starting up"
       self.last_detection_label = "None"
       self.last_detection_time = 0.0
       self.current_request = ""
      
       self.last_pos = None
       self.last_rot = None
       self.last_activity_time = time.time()
       self.nav_start_time = time.time()
       self.goal_handle = None
       
       # Long-term idle
       self.macro_pos = None
       self.macro_time = time.time()

       # ROS Setup
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

       # Threads & Timers
       self.create_timer(1.0, self.wander_logic)
       self.create_timer(0.2, self.watchdog_logic)
       self.create_timer(0.5, self.status_screen_cb)
       threading.Thread(target=self.interaction_loop, daemon=True).start()
       self.get_logger().info(f"SCOUT ONLINE: Anti-Flicker Memory Logic Active (Radius: {self.patrol_radius}m).")

   def set_status(self, message):
       self.status_message = message
       self.get_logger().info(message)

   def get_robot_pose(self):
       try:
           t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
           return (t.transform.translation.x, t.transform.translation.y)
       except Exception:
           return None

   # UI
   def status_screen_cb(self):
       panel = np.zeros((360, 620, 3), dtype=np.uint8)
       panel[:] = (28, 31, 34)
       colors = {
           "WANDERING": (80, 200, 120),
           "LISTENING": (70, 180, 255),
           "GUIDING": (80, 140, 255)
       }
       display_state = "LISTENING" if self.voice.voice_phase == "Listening now" else self.state
       state_color = colors.get(display_state, (180, 180, 180))
       cv2.putText(panel, "VOICE SCOUT", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
       cv2.circle(panel, (36, 78), 9, state_color, -1)
       cv2.putText(panel, f"State: {display_state}", (56, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
       cv2.putText(panel, f"Robot status: {self.status_message[:44]}", (24, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
       cv2.putText(panel, f"Voice status: {self.voice.voice_phase[:44]}", (24, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 180, 255), 1)
       cv2.putText(panel, f"Prompt: {self.voice.current_prompt[:48] or 'None'}", (24, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
       cv2.putText(panel, f"Raw heard: {self.voice.last_heard[:46] or 'None'}", (24, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
       cv2.putText(panel, f"Request: {self.current_request[:48] or 'None'}", (24, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
       cv2.putText(panel, f"Nav: {self.is_navigating} | Obstacle: {self.front_obstacle_detected} | Target: {self.target_label or 'None'}", (24, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
       cv2.putText(panel, f"Voice error: {self.voice.last_error[:42]}", (24, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 210, 255), 1)
       memory_text = " | ".join(f"{item}:{len(locs)}" for item, locs in self.object_memory.items() if locs)
       cv2.putText(panel, f"Memory: {memory_text or 'No objects saved yet'}", (24, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 240, 200), 1)
       cv2.imshow("Voice Scout Status", panel)
       cv2.waitKey(1)

   def scan_cb(self, msg):
       num_readings = len(msg.ranges)
       start_idx, end_idx = num_readings // 4, 3 * num_readings // 4
       relevant_ranges = msg.ranges[start_idx:end_idx]
       valid_ranges = [r for r in relevant_ranges if msg.range_min < r < msg.range_max]
       obstacle_now = bool(valid_ranges and min(valid_ranges) < 0.55)

       if obstacle_now:
           self.obstacle_hit_count += 1
           self.obstacle_clear_count = 0
       else:
           self.obstacle_clear_count += 1
           self.obstacle_hit_count = 0

       if self.obstacle_hit_count >= 3:
           self.front_obstacle_detected = True
       elif self.obstacle_clear_count >= 4:
           self.front_obstacle_detected = False

   def watchdog_logic(self):
       # Long idle
       try:
           t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
           curr_pos = (t.transform.translation.x, t.transform.translation.y)
           
           if self.state == "WANDERING":
               if self.macro_pos is None:
                   self.macro_pos = curr_pos
                   self.macro_time = time.time()
               elif time.time() - self.macro_time > 12.0:
                   if math.hypot(curr_pos[0] - self.macro_pos[0], curr_pos[1] - self.macro_pos[1]) < 0.5:
                       self.get_logger().warn("LONG IDLE DETECTED: Robot hasn't moved much in 12s. Forcing new nav goal.")
                       self.stop_navigation()
                       self.last_angle = random.uniform(0, 2*math.pi)
                       self.wander_logic()
                   self.macro_pos = curr_pos
                   self.macro_time = time.time()
       except Exception: pass

       if not self.is_navigating:
           if self.stopped_for_obstacle and not self.front_obstacle_detected:
               if time.time() - self.obstacle_stop_time > self.obstacle_resume_delay:
                   self.stopped_for_obstacle = False
                   self.status_message = "Path clear. Resuming."
                   if self.state == "GUIDING" and self.target_label:
                       self.voice.speak("The path is clear. I will keep guiding you.")
                       self.start_guidance()
                   elif self.state == "WANDERING":
                       self.wander_logic()
           self.last_activity_time = time.time()
           return

       if self.front_obstacle_detected:
           self.get_logger().error("SAFETY STOP: Side/Front obstacle detected!")
           self.status_message = "Obstacle detected. Stopped."
           if not self.stopped_for_obstacle:
               self.voice.beep(2)
               if self.state == "GUIDING":
                   self.voice.speak("Wait! There's something in our way. Let me find another way around! ¡Démosle la vuelta!")
                   threading.Timer(1.5, self.start_guidance).start()
               else:
                   self.voice.speak("I see something in the way. I will wait until the path is clear.")
           self.stopped_for_obstacle = True
           self.obstacle_stop_time = time.time()
           self.stop_navigation()
           self.last_activity_time = time.time()
           return
       try:
           t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
           curr_pos = (t.transform.translation.x, t.transform.translation.y)
           curr_rot_z = t.transform.rotation.z
           if self.state == "WANDERING":
               dist_to_goal = math.hypot(curr_pos[0] - self.current_wander_target[0],
                                         curr_pos[1] - self.current_wander_target[1])
               if dist_to_goal < 0.75:
                   self.stop_navigation()
                   self.wander_logic()
                   return
           if self.state == "GUIDING":
               target = self.object_memory[self.target_label][self.target_index]
               if math.hypot(curr_pos[0] - target[0], curr_pos[1] - target[1]) < 1.0:
                   self.stop_navigation()
                   self.face_item_then_prompt(curr_pos, target)
                   return
           if time.time() - self.nav_start_time < 2.0:
               self.last_activity_time = time.time()
               self.last_pos, self.last_rot = curr_pos, curr_rot_z
               return
           if self.last_pos is not None:
               move_dist = math.hypot(curr_pos[0]-self.last_pos[0], curr_pos[1]-self.last_pos[1])
               rot_diff = abs(curr_rot_z - self.last_rot)
               if move_dist > 0.02 or rot_diff > 0.01:
                   self.last_activity_time = time.time()
           if time.time() - self.last_activity_time > 2.5:
               self.get_logger().warn("STALL Detected. Recovering...")
               self.handle_stall(curr_pos)
           self.last_pos, self.last_rot = curr_pos, curr_rot_z
       except Exception: pass

   def stop_navigation(self):
       if self.goal_handle: self.goal_handle.cancel_goal_async()
       self.is_navigating = False
       self.last_pos = None

   def describe_location(self, location):
       robot_pos = self.get_robot_pose()
       if not robot_pos:
           return format_location(location)
       dx = location[0] - robot_pos[0]
       dy = location[1] - robot_pos[1]
       dist = math.hypot(dx, dy)
       angle = math.degrees(math.atan2(dy, dx))
       if -45 <= angle < 45:
           direction = "east"
       elif 45 <= angle < 135:
           direction = "north"
       elif angle >= 135 or angle < -135:
           direction = "west"
       else:
           direction = "south"
       return f"about {dist:.1f} meters {direction} of me"

   def describe_item_locations(self, item, locations):
       if not locations:
           return f"I have not seen a {item} yet."
       closest_index = self.choose_closest_location_index(locations)
       closest = locations[closest_index]
       closest_text = self.describe_location(closest)
       if len(locations) == 1:
           return f"I found one {item}, {closest_text}."
       return f"I found {len(locations)} {item}s. The closest one is {closest_text}."

   def choose_closest_location_index(self, locations):
       robot_pos = self.get_robot_pose()
       if not robot_pos:
           return 0
       return min(
           range(len(locations)),
           key=lambda i: math.hypot(locations[i][0] - robot_pos[0], locations[i][1] - robot_pos[1])
       )

   def handle_stall(self, curr_pos):
       self.stop_navigation()
       self.last_activity_time = time.time()
       if self.state == "WANDERING":
           escape_angle = (self.last_angle + math.pi + random.uniform(-0.5, 0.5)) % (2*math.pi)
           self.send_nav_goal(curr_pos[0] + 0.8*math.cos(escape_angle), curr_pos[1] + 0.8*math.sin(escape_angle))
       else:
           angle = random.uniform(0, 2*math.pi)
           self.send_nav_goal(curr_pos[0] + 0.6*math.cos(angle), curr_pos[1] + 0.6*math.sin(angle))
           threading.Timer(2.0, self.start_guidance).start()

   def wander_logic(self):
       if self.state != "WANDERING" or self.is_navigating: return
       try:
           t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
           curr = (t.transform.translation.x, t.transform.translation.y)
           if self.home_pose is None: self.home_pose = curr
           for _ in range(25):
               new_angle = (self.last_angle + random.uniform(math.pi/3, 5*math.pi/3)) % (2*math.pi)
               dist = random.uniform(2.5, self.patrol_radius)
               tx, ty = curr[0] + dist*math.cos(new_angle), curr[1] + dist*math.sin(new_angle)
               if math.hypot(tx - self.home_pose[0], ty - self.home_pose[1]) > self.patrol_radius:
                   ang = math.atan2(ty-self.home_pose[1], tx-self.home_pose[0])
                   tx, ty = self.home_pose[0] + (self.patrol_radius * 0.9)*math.cos(ang), \
                            self.home_pose[1] + (self.patrol_radius * 0.9)*math.sin(ang)
               if self.is_location_clear(tx, ty):
                   self.last_angle = new_angle
                   self.send_nav_goal(tx, ty)
                   return  
           self.get_logger().warn("NAV: Could not find clear wander target. Forcing blind fallback goal.")
           fallback_ang = random.uniform(0, 2*math.pi)
           self.send_nav_goal(curr[0] + 1.0*math.cos(fallback_ang), curr[1] + 1.0*math.sin(fallback_ang))
       except Exception: pass

   def is_location_clear(self, x, y):
       if self.latest_map is None: return True
       inf = self.latest_map.info
       for dx in np.arange(-0.35, 0.36, 0.1):
           for dy in np.arange(-0.35, 0.36, 0.1):
               gx = int((x + dx - inf.origin.position.x) / inf.resolution)
               gy = int((y + dy - inf.origin.position.y) / inf.resolution)
               if 0 <= gx < inf.width and 0 <= gy < inf.height:
                   if self.latest_map.data[gy * inf.width + gx] > 40: return False
               else: return False
       return True

   def send_nav_goal(self, x, y, yaw=None):
       self.current_wander_target = (x, y)
       self.nav_start_time = time.time()
       self.last_activity_time = time.time()
      
       goal = NavigateToPose.Goal()
       goal.pose.header.frame_id = 'map'
       goal.pose.header.stamp = self.get_clock().now().to_msg()
       goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
      
       if yaw is not None:
           goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
           goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
       else:
           goal.pose.pose.orientation.w = 1.0
          
       if not self.nav_client.wait_for_server(timeout_sec=2.0):
           self.is_navigating = False
           self.status_message = "Nav2 action server is not ready."
           self.get_logger().error("NAV: navigate_to_pose action server is not available.")
           return False

       self.get_logger().info(
           f"NAV: Sending {self.state} goal to X {x:.2f}, Y {y:.2f}."
       )
       self.status_message = f"Navigating to X {x:.1f}, Y {y:.1f}"
       self.is_navigating = True
       self.nav_client.send_goal_async(goal).add_done_callback(self.goal_resp_cb)
       return True

   def face_item_then_prompt(self, curr_pos, target_pos):
       angle_to_target = math.atan2(target_pos[1] - curr_pos[1], target_pos[0] - curr_pos[0])
       goal = NavigateToPose.Goal()
       goal.pose.header.frame_id = 'map'
       goal.pose.header.stamp = self.get_clock().now().to_msg()
       goal.pose.pose.position.x, goal.pose.pose.position.y = float(curr_pos[0]), float(curr_pos[1])
       goal.pose.pose.orientation.z = math.sin(angle_to_target / 2.0)
       goal.pose.pose.orientation.w = math.cos(angle_to_target / 2.0)
      
       if not self.nav_client.wait_for_server(timeout_sec=2.0):
           self.get_logger().error("NAV: Cannot face item because navigate_to_pose action server is not available.")
           self.trigger_arrival_prompt()
           return
       self.nav_client.send_goal_async(goal).add_done_callback(
           lambda _: threading.Timer(1.5, self.trigger_arrival_prompt).start()
       )

   def trigger_arrival_prompt(self):
       msg = f"I have arrived at the {self.target_label}. Is this the one you wanted?"
       self.voice.speak(msg)
       ans = self.voice.listen("Is this correct? (yes/no/nevermind): ").lower().strip()
      
       if 'y' in ans or 'never' in ans or is_cancel_query(ans):
           self.voice.speak("Okay, returning to wander mode.")
           self.state = "WANDERING"
       else:
           self.target_index += 1
           self.start_guidance()

   def goal_resp_cb(self, future):
       try:
           self.goal_handle = future.result()
       except Exception as exc:
           self.is_navigating = False
           self.get_logger().error(f"NAV: Goal request failed: {type(exc).__name__}: {exc}")
           return

       if not self.goal_handle.accepted:
           self.is_navigating = False
           self.status_message = "Navigation goal rejected."
           self.get_logger().error("NAV: Goal was rejected by Nav2.")
           return
       self.status_message = "Navigation goal accepted."
       self.get_logger().info("NAV: Goal accepted by Nav2.")
       self.goal_handle.get_result_async().add_done_callback(self.nav_done_cb)

   def nav_done_cb(self, future):
       try:
           status = future.result().status
       except Exception as exc:
           self.is_navigating = False
           self.get_logger().error(f"NAV: Goal result failed: {type(exc).__name__}: {exc}")
           return

       self.get_logger().info(f"NAV: Goal finished with status {status}.")
       if status == 4:
           self.status_message = "Navigation goal completed."
           self.is_navigating = False
           if self.state == "GUIDING":
               try:
                   t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                   self.face_item_then_prompt((t.transform.translation.x, t.transform.translation.y),
                                              self.object_memory[self.target_label][self.target_index])
               except Exception: self.trigger_arrival_prompt()

   def rgb_cb(self, msg):
       try:
           img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
           depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, 'passthrough') if self.latest_depth_msg else None
       except Exception: return

       cv2.rectangle(img, (0, 0), (img.shape[1], 38), (30, 30, 30), -1)
       cv2.putText(
           img,
           f"State: {self.state} | Memory: {sum(len(v) for v in self.object_memory.values())} objects | Last: {self.last_detection_label}",
           (10, 25),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.58,
           (240, 240, 240),
           2
       )
      
       if self.state == "WANDERING":
           results = self.model(img, stream=True, conf=0.45, verbose=False)
           current_frame_detections = []
           
           for r in results:
               for box in r.boxes:
                   label = self.model.names[int(box.cls[0])]
                   if label in self.target_list:
                       x1, y1, x2, y2 = map(int, box.xyxy[0])
                       cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                       conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                       cv2.rectangle(img, (x1, y1), (x2, y2), (40, 220, 120), 2)
                       tag = f"{label} {conf:.2f}"
                       cv2.rectangle(img, (x1, max(40, y1 - 24)), (x1 + 165, max(64, y1)), (40, 220, 120), -1)
                       cv2.putText(img, tag, (x1 + 6, max(58, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 25, 20), 1)
                       
                       if depth is not None and self.camera_intrinsics:
                           z = float(depth[cy, cx]) / 1000.0 if float(depth[cy, cx]) > 20 else float(depth[cy, cx])
                           if 0.3 < z < 5.0:
                               loc = self.get_map_coords(cx, cy, z, msg.header)
                               if loc:
                                   current_frame_detections.append((label, loc))
                                   before_count = len(self.object_memory[label])
                                   self.save_to_map(label, loc)
                                   if len(self.object_memory[label]) > before_count:
                                       cv2.putText(img, "SAVED", (x1, min(img.shape[0] - 12, y2 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 120), 2)
           
           robot_pos = self.get_robot_pose()
           if robot_pos:
               try:
                   t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                   z_q, w_q = t.transform.rotation.z, t.transform.rotation.w
                   robot_yaw = math.atan2(2.0 * w_q * z_q, 1.0 - 2.0 * z_q * z_q)

                   for item_type, locations in self.object_memory.items():
                       for i in range(len(locations) - 1, -1, -1):
                           loc = locations[i]
                           dist = math.hypot(loc[0] - robot_pos[0], loc[1] - robot_pos[1])
                           if dist < 2.5: # Object is near the robot
                               angle_to_loc = math.atan2(loc[1] - robot_pos[1], loc[0] - robot_pos[0])
                               angle_diff = (angle_to_loc - robot_yaw + math.pi) % (2 * math.pi) - math.pi
                               if abs(angle_diff) < 0.4:
                                   match_found = any(d[0] == item_type and math.hypot(d[1][0]-loc[0], d[1][1]-loc[1]) < 1.2 
                                                     for d in current_frame_detections)
                                   strike_key = f"{item_type}_{loc[0]:.2f}_{loc[1]:.2f}"
                                   if not match_found:
                                       self.memory_strikes[strike_key] = self.memory_strikes.get(strike_key, 0) + 1
                                       if self.memory_strikes[strike_key] > 20:
                                           self.get_logger().info(f"Memory Cleanup: {item_type} at {loc} not seen for 20 frames. Deleting.")
                                           self.object_memory[item_type].pop(i)
                                           del self.memory_strikes[strike_key]
                                   else:
                                       self.memory_strikes[strike_key] = 0 # Reset strikes if seen
               except Exception: pass
       else:
           cv2.putText(img, "Vision paused while not wandering", (10, img.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2)
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
       except Exception: return None

   def save_to_map(self, label, loc):
       for inst in self.object_memory[label]:
           if math.hypot(inst[0]-loc[0], inst[1]-loc[1]) < 0.6: return
       self.object_memory[label].append([loc[0], loc[1]])
       self.last_detection_label = label
       self.last_detection_time = time.time()
       self.status_message = f"Saved {label} location."
       self.get_logger().info(f"!!! MEMORY: Found {label} !!!")

   def interaction_loop(self):
       while rclpy.ok():
           if self.state == "WANDERING":
               self.status_message = "Listening for 'Hey Dora'."
               self.current_request = ""
               wake_query = self.voice.listen("Say 'Hey Dora'.", typed_fallback=False)
               if not wake_query:
                   continue

               if not has_wake_word(wake_query):
                   self.status_message = "Ignored speech without Dora wake phrase."
                   continue

               self.stop_navigation()
               self.state = "LISTENING"
               self.status_message = "Dora is listening."
               self.voice.beep()
               self.voice.speak("Yes, I am listening. What are you looking for?")

               query = strip_wake_words(wake_query)
               if not query:
                   query = self.voice.listen("Ask your question now: ", typed_fallback=True)

               if query:
                   self.current_request = query
                   self.status_message = f"Heard request: {query[:40]}"
                   self.voice.speak(f"I heard you ask: {query}.")

               if not query:
                   self.status_message = "Dora did not hear a request."
                   self.voice.speak("I did not hear a question. I will keep looking around.")
                   self.state = "WANDERING"
                   continue
              
               if is_cancel_query(query):
                   self.voice.speak("Understood, resuming patrol.")
                   self.status_message = "Patrol resumed."
                   self.state = "WANDERING"
                   continue
              
               item = extract_item(query, self.target_list)
               if not item and len(query.split()) <= 2:
                   self.voice.speak("I only heard part of that. Please ask the full question again.")
                   retry_query = self.voice.listen("Please ask the full question again: ", typed_fallback=True, phrase_time_limit=9)
                   if retry_query:
                       query = strip_wake_words(retry_query)
                       self.current_request = query
                       self.status_message = f"Heard retry: {query[:40]}"
                       self.voice.speak(f"Now I heard: {query}.")
                       item = extract_item(query, self.target_list)

               if item:
                   locs = self.object_memory.get(item, [])
                   response = self.describe_item_locations(item, locs)
                   self.voice.speak(response)
                  
                   if locs and (wants_guidance(query) or self.confirm_guidance(item)):
                       if len(locs) > 1:
                           self.voice.speak(f"I will take you to the closest {item} first.")
                       self.voice.speak(f"Starting guidance to the {item} now.")
                       self.target_label = item
                       self.target_index = self.choose_closest_location_index(locs)
                       self.start_guidance()
                   else:
                       if locs:
                           self.voice.speak("Okay. I will keep patrolling.")
                       self.status_message = "Patrol resumed."
                       self.state = "WANDERING"
               else:
                   self.voice.speak("I heard you, but I did not recognize an item I can search for. Try asking for a backpack, cup, bottle, umbrella, handbag, laptop, or phone.")
                   self.status_message = "Request did not match a saved item."
                   self.state = "WANDERING"
           time.sleep(0.1)

   def confirm_guidance(self, item):
       self.voice.beep()
       self.voice.speak(f"Do you want me to guide you to the {item}? Please say yes or no.")
       ans = self.voice.listen(f"Do you want me to guide you to the {item}? (yes/no): ")
       if is_cancel_query(ans):
           self.voice.speak("Okay, I will not guide you right now.")
           return False
       if is_yes_query(ans) or wants_guidance(ans):
           return True
       self.voice.speak("Okay, I will stay here and keep searching.")
       return False

   def start_guidance(self):
       locs = self.object_memory.get(self.target_label, [])
       if self.target_index < len(locs):
           self.state = "GUIDING"
           target = locs[self.target_index]
           self.get_logger().info(
               f"GUIDE: Starting guidance to {self.target_label} #{self.target_index + 1} "
               f"at X {target[0]:.2f}, Y {target[1]:.2f}."
           )
           if not self.send_nav_goal(target[0], target[1]):
               self.voice.speak("I could not start navigation because Nav2 is not ready.")
               self.state = "WANDERING"
       else:
           self.voice.speak(f"I have no more locations saved for {self.target_label}.")
           self.state = "WANDERING"

   def map_cb(self, msg): self.latest_map = msg
   def info_cb(self, msg): self.camera_intrinsics = msg
   def depth_cb(self, msg): self.latest_depth_msg = msg

def main():
   try:
       val = input("Enter patrol radius in meters: ")
       user_radius = float(val) if val.strip() else 3.0
   except Exception: user_radius = 3.0
   rclpy.init()
   node = VoiceScout(user_radius)
   try: rclpy.spin(node)
   except Exception: pass
   finally:
       cv2.destroyAllWindows()
       rclpy.shutdown()

if __name__ == '__main__': main()