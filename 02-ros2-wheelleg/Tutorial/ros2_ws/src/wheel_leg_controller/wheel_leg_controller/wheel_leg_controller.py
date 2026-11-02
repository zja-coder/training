import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from rm_msgs.msg import WheelLegCmd

from .vmc_cal import five_links_param
from .lqr_update import lqr_controller
from .pid import PIDController
from .climb_traj import VelocityRampFilter
from .ground_detection import detect_leg_ground_status  


class WheelLegController(Node):
    def __init__(self):
        super().__init__('wheel_leg_controller')

        # ----------  订阅 / 发布 ----------
        self.create_subscription(Imu, 'imu', self.imu_cb, 10)
        self.create_subscription(JointState, 'joint_states', self.joint_cb, 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        self.pub = self.create_publisher(WheelLegCmd, 'actuator_cmd', 10)

        # ----------  基础参数 ----------
        self.dt = 0.002
        self.wheel_radius = 0.06
        self.target_length = 0.1

        self.target_v = 0.0
        self.target_w = 0.0
        self.current_v = 0.0

        # ----------  IMU 状态 ----------
        self.pitch = 0.0
        self.roll = 0.0
        self.gyro_pitch = 0.0
        self.gyro_yaw = 0.0

        # ----------  关节缓存 ----------
        self.info = {'pos': {}, 'vel': {}}
        
        # ---------- [新增] 关节名就绪标志 ----------
        self.joints_ready = False
        self.print_count = 0  # 限制打印频率，防止刷屏

        # ----------  算法对象 ----------
        self.five_links_right = five_links_param(dt=self.dt)
        self.five_links_left = five_links_param(dt=self.dt)
        self.lqr_right = lqr_controller(dt=self.dt)
        self.lqr_left = lqr_controller(dt=self.dt)

        # ----------  PID ----------
        self.left_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0)
        self.right_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0)
        self.left_L0_pid = PIDController(700.0, 0.0, 5.0, 150.0, 0.0)
        self.right_L0_pid = PIDController(700.0, 0.0, 5.0, 150.0, 0.0)

        # ---------- 速度爬坡 ----------
        self.v_filter = VelocityRampFilter(dt=self.dt, accel_rate=0.8, decel_rate=1.4)

        # ----------  离地保护状态 ----------
        self.robot_is_airborne = False
        self.ground_detection_enabled = False
        self.get_logger().info('独立 WheelLegController 启动（LQR + VMC + 离地检测）')
    

    def imu_cb(self, msg):
        w, x, y, z = (msg.orientation.w, msg.orientation.x,
                      msg.orientation.y, msg.orientation.z)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(np.clip(sinp, -1.0, 1.0))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        _yaw = math.atan2(siny_cosp, cosy_cosp)

        self.pitch = -pitch
        self.roll = -roll
        self.gyro_pitch = -msg.angular_velocity.y
        self.gyro_yaw = msg.angular_velocity.z

    def joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            self.info['pos'][name] = msg.position[i]
            self.info['vel'][name] = msg.velocity[i]
            
        if not self.joints_ready:
            if all(k in self.info['pos'] for k in ['right1', 'right2', 'left1', 'left2', 'rightwheel', 'leftwheel']):
                self.joints_ready = True
                self.get_logger().info('✅ 数据就绪，开始平衡')
            else:
                return
                
        self.step()

    def cmd_cb(self, msg):
        self.target_v = msg.linear.x
        self.target_w = msg.angular.z

        # ==================== 封装的轮腿控制函数 ====================

    def control_left_leg(self, info, five_links, lqr, yaw_pid, L0_pid, pitch, gyro_pitch, gyro_yaw, 
                        target_w, target_length, common_motor_pos, common_motor_speed, wheel_radius,
                        is_airborne_last_step, ground_detection_enabled, leg_tp_comp=0.0):
        """
        左腿 LQR + VMC 核心控制计算与条件离地保护机制
        """
        motor1_angle = info['pos']['left1'] 
        motor2_angle = info['pos']['left2']

        # 1. 运动学与状态更新
        five_links.forward_kinematics_cal(1, motor1_angle, motor2_angle, pitch, gyro_pitch)

        # 强制规定腿部几何构型，防止双解跳变
        # L_0 是腿长，物理限制一定在 0 到 0.3 之间（不可能为负）
        five_links.L_0 = np.clip(five_links.L_0, 0.02, 0.3) 
        
        # theta 是腿部姿态角，根据你的模型，限制在合理的物理区间（比如 -1.5 到 1.5 弧度）
        five_links.theta = np.clip(five_links.theta, -1.5, 1.5)

        lqr.states_update(1, five_links, pitch, gyro_pitch, common_motor_pos, common_motor_speed, wheel_radius)
        lqr.calc_k_matrix_from_poly(five_links.L_0)

        # 2. VMC 虚拟力基本计算
        F_0_ctrl = L0_pid.compute(five_links.L_0, target_length)
        F_0 = (4 / math.cos(five_links.theta) if math.cos(five_links.theta) != 0 else 0) + F_0_ctrl
        F_0 = np.clip(F_0, -100.0, 100.0)

        # 预计算常规状况下的 Tp
        T_p = (lqr.K[1] * (lqr.theta - 0.0)
            + lqr.K[3] * (lqr.d_theta - 0.0)
            + lqr.K[5] * (lqr.x_b - lqr.x_b_set)
            + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
            + lqr.K[9] * (lqr.phi - 0.0)
            + lqr.K[11] * (lqr.d_phi - 0.0))

        # 3. 核心条件判定
        if ground_detection_enabled and (detect_leg_ground_status(five_links, F_0, T_p) == 1 or is_airborne_last_step):
            left_wheel_T = 0.0
            T_p = lqr.K[1] * (lqr.theta - 0.0) + lqr.K[3] * (lqr.d_theta - 0.0)
            T_p += leg_tp_comp
            flight_flag = 1
        else:
            yaw_ctrl = yaw_pid.compute(gyro_yaw, target_w)
            wheel_T = (lqr.K[0] * (lqr.theta - 0.0)
                    + lqr.K[2] * (lqr.d_theta - 0.0)
                    + lqr.K[4] * (lqr.x_b - lqr.x_b_set)
                    + lqr.K[6] * (lqr.v_b_whole - lqr.v_set)
                    + lqr.K[8] * (lqr.phi - 0.0)
                    + lqr.K[10] * (lqr.d_phi - 0.0))
            
            left_wheel_T = -wheel_T - yaw_ctrl
            left_wheel_T = np.clip(left_wheel_T, -1.5, 1.5)
            flight_flag = 0

        torque_set_0, torque_set_1 = five_links.VMC_torque_cal(F_0, T_p)

        return -torque_set_0, -torque_set_1, left_wheel_T, flight_flag

    def control_right_leg(self, info, five_links, lqr, yaw_pid, L0_pid, pitch, gyro_pitch, gyro_yaw, 
                        target_w, target_length, common_motor_pos, common_motor_speed, wheel_radius,
                        is_airborne_last_step, ground_detection_enabled, leg_tp_comp=0.0):
        """
        右腿 LQR + VMC 核心控制计算与条件离地保护机制
        """
        motor1_angle = info['pos']['right1']
        motor2_angle = info['pos']['right2']
        
        # 1. 运动学与状态更新
        five_links.forward_kinematics_cal(0, motor1_angle, motor2_angle, pitch, gyro_pitch)

        # 强制规定腿部几何构型，防止双解跳变
        # L_0 是腿长，物理限制一定在 0 到 0.3 之间（不可能为负）
        five_links.L_0 = np.clip(five_links.L_0, 0.02, 0.3) 
        
        # theta 是腿部姿态角，根据你的模型，限制在合理的物理区间（比如 -1.5 到 1.5 弧度）
        five_links.theta = np.clip(five_links.theta, -1.5, 1.5)
        
        lqr.states_update(0, five_links, pitch, gyro_pitch, -common_motor_pos, -common_motor_speed, wheel_radius)
        lqr.calc_k_matrix_from_poly(five_links.L_0)

        # 2. VMC 虚拟力基本计算
        F_0_ctrl = L0_pid.compute(five_links.L_0, target_length)
        F_0 = (4 / math.cos(five_links.theta) if math.cos(five_links.theta) != 0 else 0) + F_0_ctrl
        F_0 = np.clip(F_0, -100.0, 100.0)

        # 预计算常规状况下的 Tp
        T_p = (lqr.K[1] * (lqr.theta - 0.0)
            + lqr.K[3] * (lqr.d_theta - 0.0)
            + lqr.K[5] * (lqr.x_b - lqr.x_b_set)
            + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
            + lqr.K[9] * (lqr.phi - 0.0)
            + lqr.K[11] * (lqr.d_phi - 0.0))
        
        # 3. 核心条件判定
        if ground_detection_enabled and (detect_leg_ground_status(five_links, F_0, T_p) == 1 or is_airborne_last_step):
            right_wheel_T = 0.0
            T_p = lqr.K[1] * (lqr.theta - 0.0) + lqr.K[3] * (lqr.d_theta - 0.0)
            T_p += leg_tp_comp
            flight_flag = 1
        else:
            yaw_ctrl = yaw_pid.compute(gyro_yaw, target_w)
            wheel_T = (lqr.K[0] * (lqr.theta - 0.0)
                    + lqr.K[2] * (lqr.d_theta - 0.0)
                    + lqr.K[4] * (lqr.x_b - lqr.x_b_set)
                    + lqr.K[6] * (lqr.v_b_whole - lqr.v_set)
                    + lqr.K[8] * (lqr.phi - 0.0)
                    + lqr.K[10] * (lqr.d_phi - 0.0))
                    
            right_wheel_T = wheel_T - yaw_ctrl
            right_wheel_T = np.clip(right_wheel_T, -1.5, 1.5)
            flight_flag = 0

        torque_set_0, torque_set_1 = five_links.VMC_torque_cal(F_0, T_p)
        return torque_set_0, torque_set_1, right_wheel_T, flight_flag

    def step(self):
        # 如果没准备好，直接返回
        if not self.joints_ready:
            return

        #當前速度
        motor_speed_left = self.info['vel'].get('leftwheel', 0.0)
        motor_speed_right = self.info['vel'].get('rightwheel', 0.0)
        motor_pos_left = self.info['pos'].get('leftwheel', 0.0)
        motor_pos_right = self.info['pos'].get('rightwheel', 0.0)

        common_motor_speed = (motor_speed_left - motor_speed_right) / 2.0
        common_motor_pos = (motor_pos_left - motor_pos_right) / 2.0

        #判斷位置
        if self.ground_detection_enabled and self.robot_is_airborne:
            self.lqr_left.x_b_set = common_motor_pos * self.wheel_radius
            self.lqr_right.x_b_set = common_motor_pos * self.wheel_radius
            self.current_v = 0.0
            self.target_w = 0.0
        else:
            self.current_v = self.v_filter.update(
                current_v=self.current_v, target_v=self.target_v)
            if not (self.current_v == 0.0 and self.target_v == 0):
                self.lqr_left.x_b_set = common_motor_pos * self.wheel_radius
                self.lqr_right.x_b_set = common_motor_pos * self.wheel_radius

        self.lqr_left.v_set = self.current_v
        self.lqr_right.v_set = self.current_v
        
        # ========== 左腿调用 ==========
        ctrl_left_0, ctrl_left_1, left_wheel_T, left_flight = self.control_left_leg(
            info=self.info,
            five_links=self.five_links_left,
            lqr=self.lqr_left,
            yaw_pid=self.left_yaw_pid,
            L0_pid=self.left_L0_pid,
            pitch=self.pitch,
            gyro_pitch=self.gyro_pitch,
            gyro_yaw=self.gyro_yaw,
            target_w=self.target_w,
            target_length=self.target_length,
            common_motor_pos=common_motor_pos,
            common_motor_speed=common_motor_speed,
            wheel_radius=self.wheel_radius,
            is_airborne_last_step=self.robot_is_airborne,
            ground_detection_enabled=self.ground_detection_enabled, # 你的代码里没有 teleop，用这个
            leg_tp_comp=0.0
        )

        # ========== 右腿调用 ==========
        ctrl_right_0, ctrl_right_1, right_wheel_T, right_flight = self.control_right_leg(
            info=self.info,
            five_links=self.five_links_right,
            lqr=self.lqr_right,
            yaw_pid=self.right_yaw_pid,
            L0_pid=self.right_L0_pid,
            pitch=self.pitch,
            gyro_pitch=self.gyro_pitch,
            gyro_yaw=self.gyro_yaw,
            target_w=self.target_w,
            target_length=self.target_length,
            common_motor_pos=common_motor_pos,     # 原版这里没负号，直接传
            common_motor_speed=common_motor_speed, # 原版这里没负号
            wheel_radius=self.wheel_radius,
            is_airborne_last_step=self.robot_is_airborne,
            ground_detection_enabled=self.ground_detection_enabled,
            leg_tp_comp=0.0
        )

        if self.ground_detection_enabled:
            self.robot_is_airborne = (left_flight == 1) or (right_flight == 1)
        else:
            self.robot_is_airborne = False

        cmd = WheelLegCmd()
        cmd.right_phi1 = ctrl_right_0
        cmd.right_phi4 = ctrl_right_1
        cmd.left_phi1 = ctrl_left_0
        cmd.left_phi4 = ctrl_left_1
        cmd.right_wheel = right_wheel_T
        cmd.left_wheel = left_wheel_T
        self.pub.publish(cmd)
        # ---------- 打印状态信息，方便调试和录屏 ----------
        self.print_count += 1
        if self.print_count % 1000 == 0:  # 每 1000 帧打印一次，防止刷屏太快
            self.get_logger().info(
                f"Pitch: {math.degrees(self.pitch):.1f}° | "
                f"L_0: {self.five_links_left.L_0:.3f} | "
                f"Flight: {self.robot_is_airborne} | "
                f"T_out: {ctrl_left_0:.2f}, {left_wheel_T:.2f}"
            )
       

def main(args=None):
    rclpy.init(args=args)
    node = WheelLegController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()