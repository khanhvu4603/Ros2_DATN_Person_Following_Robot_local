#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import socket
import threading

class ManualBridge(Node):
    def __init__(self):
        super().__init__('manual_bridge')
        
        # Publisher
        self.pub = self.create_publisher(Twist, '/cmd_vel_manual', 10)
        
        # Mode Publisher
        self.mode_pub = self.create_publisher(String, '/control_mode', 10)
        self.current_mode = "AUTO" # Default
        
        # UDP Setup
        self.udp_ip = "127.0.0.1"
        self.udp_port = 9998
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        
        # Start listening thread
        self.thread = threading.Thread(target=self._udp_listener, daemon=True)
        self.thread.start()
        
        self.get_logger().info(f"ManualBridge started, listening on UDP {self.udp_ip}:{self.udp_port}")

    def _udp_listener(self):
        while rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(1024)
                msg_str = data.decode('utf-8').strip()
                
                # Check for MODE command
                if msg_str.startswith("MODE:"):
                    mode = msg_str.split(":")[1]
                    if mode in ["AUTO", "MANUAL"]:
                        self.current_mode = mode
                        msg = String()
                        msg.data = mode
                        self.mode_pub.publish(msg)
                        self.get_logger().info(f"Switched mode to: {mode}")
                    return

                # Parse "vx,vy,wz"
                parts = msg_str.split(',')
                if len(parts) == 3:
                    vx = float(parts[0])
                    vy = float(parts[1])
                    wz = float(parts[2])
                    
                    twist = Twist()
                    twist.linear.x = vx
                    twist.linear.y = vy
                    twist.angular.z = wz
                    
                    self.pub.publish(twist)
                    
            except Exception as e:
                self.get_logger().error(f"UDP receive error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ManualBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
