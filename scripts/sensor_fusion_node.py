#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import numpy as np
import cv2
import yaml
import message_filters
from ultralytics import YOLO
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class KalmanFilter:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1, estimation_error=1.0):
        self.A = 1  # 状态转移模型
        self.H = 1  # 观测模型
        self.Q = process_noise  # 过程噪声协方差
        self.R = measurement_noise  # 测量噪声协方差
        self.P = estimation_error  # 初始估计误差协方差
        self.x = 0  # 初始状态

    def update(self, measurement):
        # 预测
        self.P = self.A * self.P * self.A + self.Q
        # 更新
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.x = self.x + K * (measurement - self.H * self.x)
        self.P = (1 - K * self.H) * self.P
        return self.x

class SensorFusion:
    def __init__(self):
        rospy.init_node("sensor_fusion")

        self.bridge = CvBridge()
        self.model = YOLO("/root/orb_ws/src/yolo_lidar_fusion/model/yolo11s3.pt")

        # 预加载YOLO模型来避免第一次推理卡顿
        self.warmup_yolo()

        # 创建消息过滤器以进行时间同步
        image_sub = message_filters.Subscriber("/usb_cam/image_raw", Image)
        pointcloud_sub = message_filters.Subscriber("/livox/lidar", PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, pointcloud_sub], queue_size=5, slop=0.05)
        self.ts.registerCallback(self.data_callback)

        # 发布检测后的图像和过滤后的点云
        self.detection_pub = rospy.Publisher("/yolo/detections", Image, queue_size=1)
        self.filtered_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=1)
        
        # 发布3D标记的Publisher
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=11)
        # 为场地元素和轨迹可视化创建独立的发布器
        self.backboard_marker_pub_ = rospy.Publisher("/backboard_marker", Marker, queue_size=1)
        self.basket_marker_pub_ = rospy.Publisher("/basket_marker", Marker, queue_size=1)
        self.three_point_line_marker_pub_ = rospy.Publisher("/three_point_line_marker", Marker, queue_size=1)
        self.trajectory_marker_pub = rospy.Publisher("/visualization_marker_trajectory", Marker, queue_size=1)


        # 加载标定文件
        self.load_calibration("/root/orb_ws/src/yolo_lidar_fusion/config/calibration.yaml")
        
        # 初始化点云存储和范围
        self.points = []
        self.x_range = (-float('inf'), float('inf'))
        self.y_range = (-float('inf'), float('inf'))
        self.z_range = (-float('inf'), float('inf'))

        # 初始化卡尔曼滤波器
        self.kf_x = KalmanFilter()
        self.kf_y = KalmanFilter()
        self.kf_z = KalmanFilter()
        self.kf_vx = KalmanFilter()
        self.kf_vy = KalmanFilter()
        self.kf_vz = KalmanFilter()

        # 初始化上一帧的位置和时间
        self.previous_position = None
        self.previous_time = None

        # 物理常数
        self.GRAVITY = 9.81  # 重力加速度，单位：m/s^2
        self.AIR_DENSITY = 1.2  # 空气密度，单位：kg/m^3
        self.DRAG_COEFFICIENT = 0.47  # 阻力系数（球形物体）
        self.BALL_RADIUS = 0.12  # 篮球半径，单位：m
        self.BALL_AREA = np.pi * self.BALL_RADIUS ** 2  # 篮球正面投影面积
        self.BALL_MASS = 0.62  # 篮球质量，单位：kg
        self.MAGNUS_COEFFICIENT = 1e-5  # Magnus效应系数，根据实际情况调整
        self.TIME_STEP = 0.01  # 仿真时间步长，单位：s

        # 篮筐和篮板的位置和尺寸
        self.basket_position = np.array([7.10, 0, 3.05])  # 篮筐位置
        self.basket_radius = 0.23  # 篮筐半径
        self.backboard_position = np.array([7.39, 0, 3.05])  # 篮板位置
        self.backboard_height = 2.7  # 篮板高度

    def warmup_yolo(self):
        blank_image = np.zeros((640, 480, 3), dtype=np.uint8)
        self.model(blank_image)

    def load_calibration(self, file_path):
        rospy.loginfo(f"Loading calibration from {file_path}")
        with open(file_path, "r") as file:
            calib_data = yaml.safe_load(file)

        # 提取相机内参
        camera_matrix = calib_data['camera_matrix']['data']
        self.fx = camera_matrix[0]
        self.fy = camera_matrix[4]
        self.cx = camera_matrix[2]
        self.cy = camera_matrix[5]

        # 构造相机内参矩阵
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        # 提取畸变系数
        self.distortion_coeffs = np.array(calib_data['distortion_coefficients']['data'])

        # 加载相机到雷达的外参（旋转和平移）
        rotation = calib_data['extrinsic']['rotation']
        translation = calib_data['extrinsic']['translation']
        self.rotation_matrix = np.array(rotation).reshape(3, 3)
        self.translation_vector = np.array(translation).reshape(3, 1)

    def data_callback(self, img_msg, pcl_msg):
        try:
            self.process_data(img_msg, pcl_msg)
        except Exception as e:
            rospy.logerr(f"Error in data_callback: {e}")

    def process_data(self, img_msg, pcl_msg):
        # 发布篮球场元素
        self.publish_basket_marker()
        self.publish_backboard_marker()
        self.publish_three_point_line_marker()
        self.image_callback(img_msg)
        self.pointcloud_callback(pcl_msg)

    def image_callback(self, msg):
        # 将 ROS 图像消息转换为 OpenCV 图像
        undistorted_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 畸变矫正
        # undistorted_image = cv2.undistort(undistorted_image, self.camera_matrix, self.distortion_coeffs)

        # 用矫正后的图像进行YOLO检测
        results = self.model(undistorted_image)

        known_diameter = 0.25  # 设定球的实际直径
        best_box = None
        best_confidence = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]  # 获取置信度
                cv2.rectangle(undistorted_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 在框内绘制置信度
                cv2.putText(undistorted_image, f"{confidence:.2f}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 检查置信度并更新最佳框
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2)

        # 处理最佳框
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            self.map_to_3d(cx, cy, w, h, known_diameter)

        # 发布检测后的图像
        self.detection_pub.publish(self.bridge.cv2_to_imgmsg(undistorted_image, "bgr8"))

    def pointcloud_callback(self, msg):
        self.points = []
        for point in pc2.read_points(msg, skip_nans=True):
            x, y, z = point[:3]
            if (self.x_range[0] <= x <= self.x_range[1] and
                self.y_range[0] <= y <= self.y_range[1] and
                self.z_range[0] <= z <= self.z_range[1]):
                self.points.append((x, y, z))

        self.publish_filtered_pointcloud()

    def publish_filtered_pointcloud(self):
        if not self.points:
            return
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "livox_frame"
        filtered_cloud = pc2.create_cloud_xyz32(header, self.points)
        self.filtered_pub.publish(filtered_cloud)

    def map_to_3d(self, cx, cy, w, h, known_diameter):
        # 使用相机内参估算深度
        focal_length = (self.fx + self.fy) / 2  # 使用 fx 和 fy 的平均值作为焦距
        estimated_depth = focal_length * known_diameter / ((w + h) / 2)
        rospy.loginfo(f"相机深度: {estimated_depth}")
        # 将像素坐标转换为相机坐标
        u = cx
        v = cy
        rospy.loginfo(f"像素坐标: {u}, {v}")
        # 计算相机坐标系下的 3D 坐标
        x_camera = (self.cx - u) * estimated_depth / self.fx
        y_camera = (self.cy - v) * estimated_depth / self.fy
        rospy.loginfo(f"相机坐标系下的3D坐标: {x_camera}, {y_camera}")
        # 将相机坐标系下的 3D 点转换为雷达坐标系
        x_lidar, y_lidar, z_lidar = self.transform_to_lidar_frame(estimated_depth, x_camera, y_camera)
        rospy.loginfo(f"雷达坐标系的相机3D点: {x_lidar}, {y_lidar}, {z_lidar}")
        # 在点云中寻找匹配的深度和对应的点
        matched_depth = self.get_depth_from_pointcloud(y_lidar, z_lidar, x_lidar)
        rospy.loginfo(f"匹配深度: {matched_depth}")
        # 使用平均深度来计算 3D 坐标
        if matched_depth is not None:
            # 计算x和y的平均值
            y_values = []
            z_values = []

            for point in self.points:
                x, y, z = point[:3]
                if abs(x - matched_depth) <= 0.5 and np.sqrt((y_lidar - y) ** 2 + (z_lidar - z) ** 2) <= 0.5:  # 容差范围
                    y_values.append(y)
                    z_values.append(z)

            # 计算平均值
            if y_values and z_values:
                avg_x = matched_depth
                avg_y = np.mean(y_values)
                avg_z = np.mean(z_values)

                # 使用卡尔曼滤波器平滑位置
                avg_x = self.kf_x.update(avg_x)
                avg_y = self.kf_y.update(avg_y)
                avg_z = self.kf_z.update(avg_z)

                rospy.loginfo(f"Matched 3D Position in Lidar Frame: {avg_x}, {avg_y}, {avg_z}")

                # 发布到 RViz
                self.publish_marker(avg_x, avg_y, avg_z)

                # 计算速度
                current_time = rospy.Time.now().to_sec()
                current_position = np.array([avg_x, avg_y, avg_z])

                if self.previous_position is not None and self.previous_time is not None:
                    dt = current_time - self.previous_time
                    if dt > 0:
                        # 计算速度
                        velocity = (current_position - self.previous_position) / dt
                        # 使用卡尔曼滤波器平滑速度
                        vx = self.kf_vx.update(velocity[0])
                        vy = self.kf_vy.update(velocity[1])
                        vz = self.kf_vz.update(velocity[2])
                        velocity = np.array([vx, vy, vz])
                        # 预测轨迹
                        angular_velocity = np.array([0, 0, 20])  # 假设的旋转速度
                        self.publish_predicted_trajectory(current_position, velocity, angular_velocity)
                        # 进行蒙特卡罗模拟
                        self.monte_carlo_simulation(100, current_position, velocity, angular_velocity)
                else:
                    velocity = np.array([0, 0, 0])

                self.previous_position = current_position
                self.previous_time = current_time

            else:
                rospy.logwarn("No matching x or y values found in point cloud for estimated depth.")
        else:
            rospy.logwarn("No matching depth found in point cloud for estimated depth.")


    def get_depth_from_pointcloud(self, cx, cy, estimated_depth, tolerance=0.5):
        matching_depths = []

        for point in self.points:
            x, y, z = point[:3]

            if abs(x - estimated_depth) <= tolerance and np.sqrt((cx - y) ** 2 + (cy - z) ** 2) <= tolerance:
                matching_depths.append(x)

        if matching_depths:
            return np.mean(matching_depths)
        
        return None

    def transform_to_lidar_frame(self, x, y, z):
        """
        将相机坐标系下的 3D 点 (x, y, z) 转换到雷达坐标系
        """
        camera_point = np.array([[x], [y], [z]])
        lidar_point = self.rotation_matrix @ camera_point + self.translation_vector
        return lidar_point.flatten()  # 转换为 1D 数组 [x, y, z]

    def publish_marker(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = "livox_frame"  # 雷达坐标系的 frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ball"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = marker.scale.y = marker.scale.z = 0.24  # 球的直径
        marker.color.a = 1.0
        marker.color.r = 1.0
        self.marker_pub.publish(marker)

    def set_xyz_range(self, x_range, y_range, z_range):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    # 以下是物理模型和轨迹预测的函数
    def apply_air_resistance(self, velocity):
        speed = np.linalg.norm(velocity)
        drag_force = -0.5 * self.AIR_DENSITY * self.DRAG_COEFFICIENT * self.BALL_AREA * speed * velocity / self.BALL_MASS
        return drag_force

    def apply_magnus_effect(self, velocity, angular_velocity):
        magnus_force = self.MAGNUS_COEFFICIENT * np.cross(angular_velocity, velocity)
        return magnus_force / self.BALL_MASS

    def runge_kutta_step(self, position, velocity, angular_velocity):
        # 定义加速度函数
        def acceleration(v):
            g = np.array([0, 0, -self.GRAVITY])  # 重力加速度
            air_resistance = self.apply_air_resistance(v)
            magnus_effect = self.apply_magnus_effect(v, angular_velocity)
            return g + air_resistance + magnus_effect

        # Runge-Kutta 4阶方法
        k1_v = acceleration(velocity) * self.TIME_STEP
        k1_p = velocity * self.TIME_STEP

        k2_v = acceleration(velocity + 0.5 * k1_v) * self.TIME_STEP
        k2_p = (velocity + 0.5 * k1_v) * self.TIME_STEP

        k3_v = acceleration(velocity + 0.5 * k2_v) * self.TIME_STEP
        k3_p = (velocity + 0.5 * k2_v) * self.TIME_STEP

        k4_v = acceleration(velocity + k3_v) * self.TIME_STEP
        k4_p = (velocity + k3_v) * self.TIME_STEP

        new_velocity = velocity + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        new_position = position + (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6

        # 检查碰撞并处理反弹
        if self.check_collision_with_backboard(new_position):
            normal = np.array([0, -1, 0])  # 篮板法线方向
            self.bounce_ball(new_velocity, normal, angular_velocity)
        elif self.check_collision_with_rim(new_position):
            normal = (new_position - self.basket_position) / np.linalg.norm(new_position - self.basket_position)
            self.bounce_ball(new_velocity, normal, angular_velocity)

        return new_position, new_velocity

    def check_if_ball_will_score(self, position):
        distance_to_basket = np.sqrt((position[0] - self.basket_position[0]) ** 2 +
                                     (position[1] - self.basket_position[1]) ** 2)
        return distance_to_basket <= self.basket_radius and position[2] >= self.basket_position[2]

    def check_collision_with_rim(self, position):
        distance_to_basket = np.sqrt((position[0] - self.basket_position[0]) ** 2 +
                                     (position[1] - self.basket_position[1]) ** 2)
        return (distance_to_basket <= self.basket_radius) and (abs(position[2] - self.basket_position[2]) < 0.1)

    def check_collision_with_backboard(self, position):
        return (position[1] >= self.backboard_position[1]) and (position[2] <= self.backboard_height)

    def bounce_ball(self, velocity, normal, angular_velocity):
        normal_velocity = np.dot(velocity, normal)
        velocity -= 2 * normal * normal_velocity
        # 考虑自旋效应，增加反弹后的横向速度分量
        magnus_effect = self.MAGNUS_COEFFICIENT * np.cross(angular_velocity, normal)
        velocity += magnus_effect
        # 模拟能量损失
        velocity *= 0.8

    def publish_predicted_trajectory(self, position, velocity, angular_velocity):
        # 初始化Marker
        predicted_marker = Marker()
        predicted_marker.header.frame_id = "livox_frame"
        predicted_marker.header.stamp = rospy.Time.now()
        predicted_marker.ns = "predicted_trajectory"
        predicted_marker.id = 0
        predicted_marker.type = Marker.LINE_STRIP
        predicted_marker.action = Marker.ADD
        predicted_marker.scale.x = 0.02  # 线条宽度
        predicted_marker.color.r = 1.0
        predicted_marker.color.g = 0.0
        predicted_marker.color.b = 0.0
        predicted_marker.color.a = 1.0

        # 模拟运动直到篮球落地或超出范围
        trajectory_points = []
        sim_position = position.copy()
        sim_velocity = velocity.copy()
        max_simulation_time = 5.0  # 最大模拟时间，单位：秒
        sim_time = 0.0

        while sim_position[2] > 0 and sim_time < max_simulation_time:
            sim_position, sim_velocity = self.runge_kutta_step(sim_position, sim_velocity, angular_velocity)
            point = Point()
            point.x = sim_position[0]
            point.y = sim_position[1]
            point.z = sim_position[2]
            trajectory_points.append(point)
            sim_time += self.TIME_STEP

        predicted_marker.points = trajectory_points
        self.trajectory_marker_pub.publish(predicted_marker)

    def monte_carlo_simulation(self, num_simulations, initial_position, initial_velocity, angular_velocity):
        success_count = 0
        left_count = 0
        right_count = 0

        for _ in range(num_simulations):
            # 添加随机扰动
            perturbed_position = initial_position + np.random.normal(0, 0.1, size=3)
            perturbed_velocity = initial_velocity + np.random.normal(0, 0.1, size=3)
            result = self.predict_ball_trajectory(perturbed_position, perturbed_velocity, angular_velocity)
            if result['scored']:
                success_count += 1
            elif result['landing_side'] == 'left':
                left_count += 1
            elif result['landing_side'] == 'right':
                right_count += 1

        success_probability = success_count / num_simulations
        no_score_probability = 1 - success_probability
        left_probability = left_count / (num_simulations - success_count) if success_count != num_simulations else 0
        right_probability = right_count / (num_simulations - success_count) if success_count != num_simulations else 0

        rospy.loginfo(f"Scored Probability: {success_probability * 100:.2f}%")
        rospy.loginfo(f"Missed Probability: {no_score_probability * 100:.2f}%")
        if no_score_probability > 0:
            rospy.loginfo(f"  - Landing Left Probability: {left_probability * 100:.2f}%")
            rospy.loginfo(f"  - Landing Right Probability: {right_probability * 100:.2f}%")

    def predict_ball_trajectory(self, position, velocity, angular_velocity):
        sim_position = position.copy()
        sim_velocity = velocity.copy()
        sim_time = 0.0
        max_simulation_time = 5.0

        while sim_position[2] > 0 and sim_time < max_simulation_time:
            sim_position, sim_velocity = self.runge_kutta_step(sim_position, sim_velocity, angular_velocity)
            if self.check_if_ball_will_score(sim_position):
                return {'scored': True, 'landing_side': ''}
            sim_time += self.TIME_STEP

        landing_side = 'left' if sim_position[0] < 0 else 'right'
        return {'scored': False, 'landing_side': landing_side}

    # 发布篮球场的元素
    def publish_basket_marker(self):
        marker = Marker()
        marker.header.frame_id = "livox_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "court"
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = self.basket_position[0]
        marker.pose.position.y = self.basket_position[1]
        marker.pose.position.z = self.basket_position[2]
        marker.scale.x = marker.scale.y = self.basket_radius * 2  # 直径
        marker.scale.z = 0.05  # 厚度
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.basket_marker_pub_.publish(marker)

    def publish_backboard_marker(self):
        marker = Marker()
        marker.header.frame_id = "livox_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "court"
        marker.id = 2
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = self.backboard_position[0]
        marker.pose.position.y = self.backboard_position[1]
        marker.pose.position.z = self.backboard_position[2]
        marker.scale.x = 0.05  # 厚度
        marker.scale.y = 1.83  # 宽度
        marker.scale.z = 1.22  # 高度
        marker.color.r = 0.6
        marker.color.g = 0.6
        marker.color.b = 0.6
        marker.color.a = 1.0
        self.backboard_marker_pub_.publish(marker)

    def publish_three_point_line_marker(self):
        marker = Marker()
        marker.header.frame_id = "livox_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "court"
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02  # 线条宽度
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        three_point_radius = 7.24  # 三分线半径

        # 设置旋转角度为 180 度
        angle_offset = np.pi  # 180度（即 π）
        cos_angle = np.cos(angle_offset)
        sin_angle = np.sin(angle_offset)

        points = []
        for angle in np.linspace(-np.pi / 2, np.pi / 2, num=100):
            x = three_point_radius * np.cos(angle)
            y = self.basket_position[1] + three_point_radius * np.sin(angle)
            
            # 应用 180 度旋转变换
            x_rot = x * cos_angle - y * sin_angle
            y_rot = x * sin_angle + y * cos_angle

            point = Point()
            point.x = x_rot+7.24
            point.y = y_rot
            point.z = 0
            points.append(point)

        marker.points = points
        self.three_point_line_marker_pub_.publish(marker)

if __name__ == "__main__":
    processor = SensorFusion()
    processor.set_xyz_range((-1, 1), (-1, 1), (0, 1))
    rospy.spin()
