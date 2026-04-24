import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import sys

class InteractiveNavigator(Node):
    def __init__(self, x, y):
        super().__init__('interactive_navigator')
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_x = x
        self.goal_y = y
        
        self.get_logger().info(f"Navigating to Target: X={self.goal_x}, Y={self.goal_y}")
        self.send_goal()

    def send_goal(self):
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 server not found. Is the robot running?")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = self.goal_x
        goal_msg.pose.pose.position.y = self.goal_y
        goal_msg.pose.pose.orientation.w = 1.0  # Facing neutral

        self.get_logger().info("Communicating with Nav2...")
        self._send_goal_future = self.client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal REJECTED by Nav2 stack.")
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        if status == 4:
            print("\n✅ ARRIVAL SUCCESSFUL!")
        else:
            print(f"\n⚠️ Navigation failed with status code: {status}")
        
        rclpy.shutdown()

def main():
    print("--- Robot Mission Control ---")
    try:
        # Asking for user input
        user_x = float(input("Enter Target X coordinate: "))
        user_y = float(input("Enter Target Y coordinate: "))
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return

    rclpy.init()
    node = InteractiveNavigator(user_x, user_y)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nMission cancelled by user.")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()