# test_person_depth.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # --- Args ---
    use_view = DeclareLaunchArgument(
        'use_view', default_value='true',
        description='Show color and depth images with image_view'
    )
    color_res = DeclareLaunchArgument(
        'color_res', default_value='640x480',
        description='Color resolution WxH'
    )
    color_fps = DeclareLaunchArgument(
        'color_fps', default_value='30',
        description='Color FPS'
    )
    depth_fps = DeclareLaunchArgument(
        'depth_fps', default_value='30',
        description='Depth FPS'
    )

    # Parse WxH
    res = LaunchConfiguration('color_res')
    # image_view không cần biết WxH; realsense2_camera cần
    # Ta tách W,H từ '640x480' trong tham số node (C++ driver chấp nhận string này)
    color_width = LaunchConfiguration('color_res')
    color_height = LaunchConfiguration('color_res')

    # --- RealSense D455 driver ---
    realsense = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d455',
        output='screen',
        parameters=[{
            'align_depth': True,
            'color_width': 640,
            'color_height': 480,
            'color_fps': LaunchConfiguration('color_fps'),
            'depth_fps': LaunchConfiguration('depth_fps'),
            'enable_color': True,
            'enable_depth': True,
            # Tắt các stream khác cho gọn
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
        }]
    )

    # --- Person Detector (bạn đổi your_pkg_name thành tên gói của bạn) ---
    person_detector = Node(
        package='mecanum_control',
        executable='person_detector',
        name='person_detector',
        output='screen',
        parameters=[{
            # topics
            'image_topic': '/camera/color/image_raw',
            'use_depth': True,
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'depth_encoding': '16UC1',        # hoặc '32FC1' nếu bạn xuất float (m)
            # detector & control params (điền giá trị mặc định bạn đang dùng)
            'target_bbox_width': 200.0,       # px: độ rộng bbox mục tiêu (khoảng cách “đẹp”)
            'stop_margin_px': 15.0,           # px: dải dừng (NO-REVERSE khi quá gần)
            'kx': 0.0015,                     # lướt ngang
            'kz': 0.0010,                     # xoay
            'kd': 0.0020,                     # tiến (theo bbox width) — nếu bạn đã đổi sang depth control thì chỉnh sau
            'max_vx': 0.4,
            'max_vy': 0.4,
            'max_wz': 0.8,
            # depth ROI & lọc
            'roi_shrink_ratio': 0.4,
            'depth_min_m': 0.15,
            'depth_max_m': 8.0,
        }],
        remappings=[('/person_estimated_distance', '/person_distance')],  
    )

    # --- Viewers (tùy chọn) ---
    color_view = Node(
        package='image_view',
        executable='image_view',
        name='color_view',
        output='screen',
        remappings=[('image', '/camera/color/image_raw')],
        condition=None
    )

    depth_view = Node(
        package='image_view',
        executable='image_view',
        name='depth_view',
        output='screen',
        remappings=[('image', '/camera/aligned_depth_to_color/image_raw')],
        # image_view sẽ hiển thị depth dạng gray; nếu muốn colormap, dùng rqt_image_view
        condition=None
    )

    return LaunchDescription([
        use_view, color_res, color_fps, depth_fps,
        realsense,
        person_detector,
        color_view,
        depth_view,
    ])
