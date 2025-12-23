# mecanum.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    # ====== Common args ======    
    serial_port     = DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0')
    baud_rate       = DeclareLaunchArgument('baud_rate',   default_value='115200')

    lidar_serial    = DeclareLaunchArgument('lidar_serial_port', default_value='/dev/ttyUSB1')
    use_lidar       = DeclareLaunchArgument('use_lidar',   default_value='true')
    use_imu         = DeclareLaunchArgument('use_imu',     default_value='true')
    use_camera      = DeclareLaunchArgument('use_camera',  default_value='true')

    # Webcam OpenCV (mặc định TẮT khi dùng D455)
    use_opencv_cam  = DeclareLaunchArgument('use_opencv_cam', default_value='false')
    video_device    = DeclareLaunchArgument('video_device',   default_value='/dev/video4')
    show_window     = DeclareLaunchArgument('show_window',    default_value='false')
    opencv_enabled  = DeclareLaunchArgument('opencv_enabled', default_value='false')

    # Detector/Viewer
    yolo_model      = DeclareLaunchArgument('yolo_model',      default_value='yolov8n.onnx')  # hoặc yolov8n.pt
    use_viewer      = DeclareLaunchArgument('use_viewer',      default_value='true')
    no_use_viewer   = DeclareLaunchArgument('no_use_viewer',   default_value='false')
    depth_encoding  = DeclareLaunchArgument('depth_encoding',  default_value='16UC1')

    # ====== RealSense D455 ======
    d455_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d455',
        output='screen',
        parameters=[{
            # Quan trọng: bật align_depth để có topic depth đã căn theo color
            'align_depth': True,
            'enable_color': True, 'enable_depth': True,
            'color_width': 640, 'color_height': 480, 'color_fps': 30,
            'depth_fps': 30,
            'enable_infra1': False, 'enable_infra2': False,
            'enable_gyro': False,  'enable_accel': False,
        }],
        condition=IfCondition(LaunchConfiguration('use_camera')),
    )

    # ====== (tuỳ chọn) Webcam OpenCV ======
    opencv_cam = Node(
        package='mecanum_control',
        executable='opencv_camera_node',
        name='opencv_camera_node',
        output='screen',
        parameters=[
            {'video_device': LaunchConfiguration('video_device')},
            {'width': 640}, {'height': 480}, {'fps': 30},
            {'image_topic': '/webcam/image_raw'},
            {'show_window': LaunchConfiguration('show_window')},
            {'enabled': LaunchConfiguration('opencv_enabled')},
        ],
        condition=IfCondition(LaunchConfiguration('use_opencv_cam')),
    )

    # ====== IMU raw + Madgwick ======
    imu_reader = Node(
        package='mecanum_control',
        executable='imu_raw_publisher',
        name='imu_raw_publisher',
        output='screen',
        parameters=[
            {'serial_port': LaunchConfiguration('serial_port')},
            {'baud_rate':   LaunchConfiguration('baud_rate')},
            {'publish_rate': 100.0},
            {'frame_id': 'imu_link'},
        ],
        condition=IfCondition(LaunchConfiguration('use_imu')),
    )
    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{'use_mag': False}, {'publish_tf': True}, {'world_frame': 'enu'}],
        remappings=[
            ('imu/data_raw', '/imu/data_raw'),
            ('imu/data',     '/imu/filtered'),
        ],
        condition=IfCondition(LaunchConfiguration('use_imu')),
    )

    # ====== RPLIDAR + Processor ======
    rplidar = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        output='screen',
        parameters=[
            {'serial_port': LaunchConfiguration('lidar_serial_port')},
            {'serial_baudrate': 115200},
            {'frame_id': 'laser_frame'},
            {'angle_compensate': True},
        ],
        condition=IfCondition(LaunchConfiguration('use_lidar')),
    )
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_tf_publisher',
        arguments=['0.1','0','0.2','0','0','0','base_link','laser_frame'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_lidar')),
    )
    lidar_processor = Node(
        package='mecanum_control',
        executable='lidar_processor',
        name='lidar_processor',
        output='screen',
        parameters=[
            {'scan_topic': '/scan'},
            {'publish_topic': '/cmd_vel_emergency'},

            # Ngưỡng & dải góc
            {'min_safe_distance': 0.60},
            {'safety_zone_sides': 0.50},
            {'angle_range_front': 90.0},
            {'angle_range_sides': 70.0},
            {'yaw_offset_deg': 0.0},

            # Né & bypass
            {'emergency_vy': 0.22},
            {'bypass_min_time_s': 0.40},
            {'bypass_release_hysteresis_m': 0.05},
            {'bypass_timeout_s': 3.0},

            # NHẢ NGAY khi phía trước thoáng (strict release)
            {'release_on_clear_immediate': True},
            {'clear_debounce_s': 0.02},
            {'hold_centering_after_release_s': 0.10},
            {'instant_stop_on_clear': True},

            # Quan trọng: để trước trống vẫn cho đi thẳng
            {'gate_mode': 'front_only'},

            # Làm mượt & căn giữa
            {'vy_slew_rate': 1.8},
            {'enable_corridor_centering': True},
            {'center_k': 0.6},
            {'center_vy_cap': 0.20},

            # Không lách khi rất gần người
            {'suppress_when_target_close': True},
            {'suppress_threshold_m': 0.90},

            # Person-distance masking
            {'person_ignore_enable': True},
            {'person_ignore_mode': 'distance_only'},
            {'person_ignore_margin_m': 0.40},
            {'person_ignore_deg': 60.0},
            {'person_ignore_only_when_locked': True},

            # Né theo hông
            {'side_trigger_enable': True},
            {'side_soft_avoid': True},
            {'side_soft_gain': 0.18},

            # Gating theo trạng thái bám người
            {'target_distance_m': 1.60},
        ],
        condition=IfCondition(LaunchConfiguration('use_lidar')),
    )

    # ====== Dynamic Obstacle Tracker (node mới) ======
    dynamic_tracker = Node(
        package='mecanum_control',
        executable='dynamic_obstacle_tracker',
        name='dynamic_obstacle_tracker',
        output='screen',
        parameters=[
            {'scan_topic': '/scan'},
            {'cmd_vel_topic': '/cmd_vel_arbiter'},
            {'cluster_dist_thresh': 0.30},
            {'cluster_min_points': 3},
            {'assoc_dist_thresh': 0.50},
            {'track_max_age_s': 1.5},
            {'vel_smoothing_alpha': 0.6},
            {'dynamic_speed_thresh': 0.25},
            {'ttc_thresh_front': 1.5},
            {'lane_half_width': 0.40},
            {'min_vrel_x': 0.05},
            {'max_range_considered': 8.0},
        ],
        condition=IfCondition(LaunchConfiguration('use_lidar')),
    )

    # ====== Person Detector ======
    person_detector = Node(
        package='mecanum_control',
        executable='person_detector',
        name='person_detector',
        output='screen',
        parameters=[
            {'camera_topic': '/camera/d455/color/image_raw'},
            {'yolo_model': LaunchConfiguration('yolo_model')},
            {'publish_debug_image': True},
            {'use_depth': True},
            {'depth_topic': '/camera/d455/depth/image_rect_raw'},
            {'depth_encoding': LaunchConfiguration('depth_encoding')},
        ],
    )

    # ====== Controller + Arbiter + STM32 ======
    mecanum_controller = Node(
        package='mecanum_control',
        executable='mecanum_controller',
        name='mecanum_controller',
        output='screen',
        parameters=[
            {'wheel_separation_x': 0.3},
            {'wheel_separation_y': 0.3},
            {'max_speed': 0.5},
            {'use_raw_imu': LaunchConfiguration('use_imu')},
            {'wait_for_detection': True},
            {'allowed_states': ['LOCKED']},
        ],
        remappings=[
            ('/follow_state', '/person_detector/follow_state'),
        ],
    )

    velocity_arbiter = Node(
        package='mecanum_control',
        executable='velocity_arbiter',
        name='velocity_arbiter',
        output='screen',
        parameters=[
            {'merge_when_emergency': True},
            {'allow_person_when_unsafe': False},
            {'v_planar_cap': 0.55},
            {'stale_ms': 300},
            {'ema_alpha': 0.35},
            {'max_vx_when_unsafe': 0.2},
        ],
    )

    stm32 = Node(
        package='mecanum_control',
        executable='stm32_communicator',
        name='stm32_communicator',
        output='screen',
        parameters=[
            {'serial_port': LaunchConfiguration('serial_port')},
            {'baud_rate': LaunchConfiguration('baud_rate')},
            {'wheel_separation_x': 0.3},
            {'wheel_separation_y': 0.3},
            {'wheel_radius': 0.049},
            {'pwm_max': 150},  #130
            {'pwm_min': 100},
            {'pwm_deadzone': 10},
            {'ema_alpha': 0.4},
            {'pwm_filter_alpha': 0.8},
            {'control_period': 0.1},
            {'odom_publish_rate': 20.0},
            {'connection_timeout': 5.0},
            {'serial_write_timeout': 0.01},
        ],
    )

    manual_bridge = Node(
        package='mecanum_control',
        executable='manual_bridge',
        name='manual_bridge',
        output='screen'
    )

    # ====== Viewers ======
    color_view = Node(
        package='image_view',
        executable='image_view',
        name='color_view',
        remappings=[('image', '/camera/d455/color/image_raw')],
        condition=IfCondition(LaunchConfiguration('no_use_viewer')),
        output='screen',
    )
    depth_view = Node(
        package='image_view',
        executable='image_view',
        name='depth_view',
        remappings=[('image', '/camera/aligned_depth_to_color/image_raw')],
        condition=IfCondition(LaunchConfiguration('no_use_viewer')),
        output='screen',
    )
    det_debug_view = Node(
        package='image_view',
        executable='image_view',
        name='detector_debug_view',
        remappings=[('image', '/person_detector/debug_image')],
        condition=IfCondition(LaunchConfiguration('no_use_viewer')),
        output='screen',
    )
    recovery_view = Node(
        package='mecanum_control',
        executable='opencv_view_node',
        name='recovery_state_viewer',
        remappings=[('image', '/person_detector/debug_image')],
        parameters=[{'image_topic': '/person_detector/debug_image'}],
        condition=IfCondition(LaunchConfiguration('use_viewer')),
        output='screen',
    )

    return LaunchDescription([
        # Args
        serial_port, baud_rate,
        lidar_serial, use_lidar, use_imu, use_camera,
        use_opencv_cam, video_device, show_window, opencv_enabled,
        yolo_model, use_viewer, depth_encoding, no_use_viewer,

        # Nodes
        d455_node, opencv_cam,
        imu_reader, imu_filter,
        rplidar, lidar_tf, lidar_processor,
        dynamic_tracker,         
        person_detector,
        mecanum_controller, velocity_arbiter, stm32, manual_bridge,
        color_view, depth_view, det_debug_view, recovery_view,
    ])
