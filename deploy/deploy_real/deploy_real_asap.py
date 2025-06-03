from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class PreciseTimer:
    def __init__(self, dt):
        self.dt = dt
        self.next_time = None
    
    def start(self):
        """开始计时"""
        self.next_time = time.time() + self.dt
    
    def wait(self):
        """精确等待，替代time.sleep"""
        if self.next_time is None:
            self.start()
            return
            
        current_time = time.time()
        if current_time < self.next_time:
            time.sleep(self.next_time - current_time)
        self.next_time += self.dt


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        
        # 添加精确定时器
        self.timer = PreciseTimer(config.control_dt)

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # 读取关节位置限制参数
        self.clip_positions = getattr(config, 'clip_positions', False)
        if hasattr(config, 'joint_limits_lower') and hasattr(config, 'joint_limits_upper'):
            self.joint_limits_lower = np.array(config.joint_limits_lower, dtype=np.float32)
            self.joint_limits_upper = np.array(config.joint_limits_upper, dtype=np.float32)
        else:
            # 如果配置文件中没有关节限制，使用默认值
            self.joint_limits_lower = np.array([-10.0] * config.num_actions, dtype=np.float32)
            self.joint_limits_upper = np.array([10.0] * config.num_actions, dtype=np.float32)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        
        self.timer.start()  # 只需要启动一次
        
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            # time.sleep(self.config.control_dt)
            self.timer.wait()  # 直接替代time.sleep

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        # Combine all joints: legs + arms/waist + wrists
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx + self.config.wrist_joint2motor_idx
        kps = (self.config.kps + self.config.arm_waist_kps + self.config.wrist_kps)
        kds = (self.config.kds + self.config.arm_waist_kds + self.config.wrist_kds) 
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target, self.config.wrist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            #time.sleep(self.config.control_dt)
            self.timer.wait()  # 直接替代time.sleep

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            # Control legs
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # Control arms/waist
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # Control wrists
            for i in range(len(self.config.wrist_joint2motor_idx)):
                motor_idx = self.config.wrist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            # time.sleep(self.config.control_dt)
            self.timer.wait()  # 直接替代time.sleep

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity for all policy-controlled joints (legs + arms/waist)
        policy_joint_indices = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        for i in range(len(policy_joint_indices)):
            motor_idx = policy_joint_indices[i]
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.cmd[0] = 0.025 #self.remote_controller.ly
        self.cmd[1] = 0 #self.remote_controller.lx * -1
        self.cmd[2] = 0 #self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        # self.obs[:3] = ang_vel
        # self.obs[3:6] = gravity_orientation
        # self.obs[6 : 6 + num_actions] = qj_obs
        # self.obs[6 + num_actions : 6 + num_actions * 2] = dqj_obs
        # self.obs[6 + num_actions * 2 : 6 + num_actions * 3] = self.action

        self.obs_buf = np.concatenate((self.action, ang_vel[0], qj_obs, dqj_obs, gravity_orientation), axis=-1, dtype=np.float32)

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs_buf).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # 根据策略输出顺序正确切片动作
        # 策略输出顺序: 12个腿部关节 + 3个腰部关节 + 8个手臂关节 = 23个
        leg_actions = self.action[:12]  # 前12个是腿部关节
        waist_actions = self.action[12:15]  # 13-15是腰部关节
        arm_actions = self.action[15:23]  # 16-23是手臂关节
        # waist_actions = np.zeros_like(self.action[12:15])  # 13-15是腰部关节
        # arm_actions = np.zeros_like(self.action[15:23])  # 16-23是手臂关节
        
        # 组合腰部和手臂动作（按照配置文件中arm_waist的顺序）
        arm_waist_actions = np.concatenate([waist_actions, arm_actions])
        
        # transform action to target_dof_pos for legs
        leg_default_angles = self.config.default_angles[:len(self.config.leg_joint2motor_idx)]
        target_leg_pos = leg_default_angles + leg_actions * self.config.action_scale
        
        # transform action to target_dof_pos for arms/waist
        arm_waist_default_angles = self.config.default_angles[len(self.config.leg_joint2motor_idx):len(self.config.leg_joint2motor_idx)+len(self.config.arm_waist_joint2motor_idx)]
        target_arm_waist_pos = arm_waist_default_angles + arm_waist_actions * self.config.action_scale

        # === 添加关节位置限制 ===
        if self.clip_positions:
            # 合并所有目标位置进行统一限制
            combined_target_pos = np.concatenate([target_leg_pos, target_arm_waist_pos])
            combined_target_pos_before = combined_target_pos.copy()
            
            # 应用关节位置限制
            combined_target_pos = np.clip(combined_target_pos, self.joint_limits_lower, self.joint_limits_upper)
            
            # 检查是否有关节被限制
            clipped_indices = np.where((combined_target_pos_before != combined_target_pos))[0]
            if len(clipped_indices) > 0:
                print(f"关节位置被限制: {clipped_indices}, 原值: {combined_target_pos_before[clipped_indices]}, 限制后: {combined_target_pos[clipped_indices]}")
            
            # 分离回腿部和手臂/腰部位置
            target_leg_pos = combined_target_pos[:len(self.config.leg_joint2motor_idx)]
            target_arm_waist_pos = combined_target_pos[len(self.config.leg_joint2motor_idx):]

        # target_leg_pos = leg_default_angles #np.zeros_like(target_leg_pos)
        # # target_leg_pos[0] = -0.25
        # # target_leg_pos[6] = 0.25
        # target_arm_waist_pos = np.zeros_like(target_arm_waist_pos)
        # target_arm_waist_pos[1] = .2
        # # target_arm_waist_pos[4] = .5
        # # target_arm_waist_pos[-3] = -.5
        # print(target_arm_waist_pos)

        # custom_init_angles = np.array([ 0.00128237,  0.28164408, -0.07875013,  0.13568419,  0.00686174,
        #     0.        , -0.14858443, -0.31706324,  0.10721936,  0.04902612,
        #     0.00255974,  0.        , -0.23306386,  0.05477175,  0.12390576,
        # -1.084431  , -0.05879271, -0.74253875,  0.15691191, -0.6602233 ,
        # -0.5101053 , -0.04324551,  0.        ], dtype=np.float32)
        # # 根据关节顺序分配角度到target_leg_pos和target_arm_waist_pos
        # # 腿部关节(0-11)
        # target_leg_pos = custom_init_angles[:12]  # 前12个是腿部关节
        
        # # 躯干和手臂关节(12-22)
        # target_arm_waist_pos = custom_init_angles[12:]  # 后11个是躯干和手臂关节
        
        # # 打印分配结果以供验证
        # print("腿部关节角度:", target_leg_pos)
        # print("躯干和手臂关节角度:", target_arm_waist_pos)

        kp_scale = 1.0
        kd_scale = 4.0
        # Build low cmd for legs
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_leg_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i] * kp_scale
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i] * kd_scale
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # Build low cmd for arms/waist (policy controlled)
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_arm_waist_pos[i] # 0
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i] * kp_scale
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i] * kd_scale  
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # Build low cmd for wrists (keep default pose)
        for i in range(len(self.config.wrist_joint2motor_idx)):
            motor_idx = self.config.wrist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i] * kp_scale
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i] * kd_scale
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        # time.sleep(self.config.control_dt)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            controller.timer.wait()  # 直接替代time.sleep
            
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
