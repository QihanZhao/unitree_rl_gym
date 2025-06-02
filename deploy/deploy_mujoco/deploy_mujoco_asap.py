import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

# def quat_rotate_inverse(q, v): # q 是四元数，v 是三维向量 
#     q_w = q[-1] # 四元数的实部 
#     q_vec = q[:3] # 四元数的虚部（向量部分）
#     # 计算旋转后的向量
#     a = v * (2.0 * q_w ** 2 - 1.0)
#     b = 2.0 * q_w * np.cross(q_vec, v)
#     c = 2.0 * q_vec * np.dot(q_vec, v)
#     return a - b + c


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    
    q_conj = np.array([w, -x, -y, -z])
    
    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32) # * 0.5
        kds = np.array(config["kds"], dtype=np.float32) #* .8

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    # custom_init_angles = np.array([ 0.00128237,  0.28164408, -0.07875013,  0.13568419,  0.00686174,
    #     0.        , -0.14858443, -0.31706324,  0.10721936,  0.04902612,
    #     0.00255974,  0.        , -0.23306386,  0.05477175,  0.12390576,
    #    -1.084431  , -0.05879271, -0.74253875,  0.15691191, -0.6602233 ,
    #    -0.5101053 , -0.04324551,  0.        ], dtype=np.float32)
    custom_init_angles = np.array([
        -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,  # 左腿 (6个DOF)
        -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,  # 右腿 (6个DOF)
         0.0,  0.0,  0.0,  # 腰部 (3个DOF) 
         0.0,  0.9,  0.0,  0.0,  #左臂 (4个DOF)
         0.0,  -0.9, 0.0,  0.0   # 右臂 (4个DOF)
    ], dtype=np.float32)
    target_dof_pos = custom_init_angles.copy()
    # target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # === 使用标准直立姿态（与ASAP default_dof_pos对齐）===
    # 标准直立四元数 [w,x,y,z] = [1,0,0,0]
    d.qpos[3:7] = [ 1,0,0,0]  # 标准直立姿态

    # === 使用ASAP训练时的实际姿态 ===
    # ASAP中观察到的四元数 [x,y,z,w] = [-0.1059, 0.1389, 0.9357, 0.3065]
    # MuJoCo需要 [w,x,y,z] 格式
    # asap_quat_xyzw = np.array([ 0.03094886, -0.0356843 ,  0.96169288,  0.27002888])
    # asap_quat_wxyz = np.array([asap_quat_xyzw[3], asap_quat_xyzw[0], asap_quat_xyzw[1], asap_quat_xyzw[2]])
    # d.qpos[3:7] = asap_quat_wxyz  # 使用ASAP训练时的实际姿态
    d.qpos[7:] = custom_init_angles
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    # 读取clip参数
    action_clip_value = config.get("action_clip_value", 0.0)
    clip_torques = config.get("clip_torques", False)
    torque_limits = np.array(config.get("torque_limits", [0]*num_actions), dtype=np.float32)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # print(target_dof_pos)
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            # === ASAP对齐：clip torque ===
            if clip_torques:
                tau = np.clip(tau, -torque_limits, torque_limits)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)


            counter += 1  # 修复：确保counter正确递增
            if counter % control_decimation == 0:
                # tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                # # === ASAP对齐：clip torque ===
                # if clip_torques:
                #     tau = np.clip(tau, -torque_limits, torque_limits)
                # d.ctrl[:] = tau
                # # mj_step can be replaced with code that also evaluates
                # # a policy and applies a control signal before stepping the physics.
                # mujoco.mj_step(m, d)

                # print("max|action|:", np.abs(action).max(),"max|tau|:", np.abs(tau).max())
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]  # MuJoCo: [w,x,y,z]
                omega = d.qvel[3:6]

                #Z === 添加传感器噪声 ===
                qj = qj + np.random.normal(0, 0.01, qj.shape)  # 位置传感器噪声
                dqj = dqj + np.random.normal(0, 0.01, dqj.shape)  # 速度传感器噪声
                omega = omega + np.random.normal(0, 0.01, omega.shape)  # IMU噪声
                
                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale

                # === 修复：与ASAP的projected_gravity计算对齐 ===
                # ASAP使用 quat_rotate_inverse(base_quat, gravity_vec)
                # 其中 base_quat 是 [x,y,z,w] 格式，gravity_vec = [0,0,-1]
                quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])  # 转换为 [x,y,z,w]
                gravity_orientation = quat_rotate_inverse(quat_xyzw, np.array([0,0,-1]))
                # print(quat_xyzw, gravity_orientation, get_gravity_orientation(quat))
                gravity_orientation = gravity_orientation + np.random.normal(0, 0.01, 3)  #Z 重力方向计算（添加IMU噪声）

                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                # obs[:3] = omega # np.zeros(3)
                # obs[3:6] = gravity_orientation # np.array([0, 0, -1])
                # # obs[6:9] = cmd * cmd_scale
                # obs[6 : 6 + num_actions] = qj #default_angles
                # obs[6 + num_actions : 6 + 2 * num_actions] = dqj #np.zeros_like(default_angles)
                # obs[6 + 2 * num_actions : 6 + 3 * num_actions] = np.zeros_like(default_angles)
                # # obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                # obs[:3] = omega
                # obs[3:6] = gravity_orientation
                # # obs[6:9] = cmd * cmd_scale
                # obs[6 : 6 + num_actions] = qj
                # obs[6 + num_actions : 6 + 2 * num_actions] = dqj
                # obs[6 + 2 * num_actions : 6 + 3 * num_actions] = action
                # # obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                obs_buf = np.concatenate((action, omega, qj, dqj, gravity_orientation), axis=-1, dtype=np.float32)
                obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # === ASAP对齐：clip action ===
                action = np.clip(action, -action_clip_value, action_clip_value)
                # transform action to target_dof_pos
                # action_cp = action.copy()
                # action_cp[13:] = 0.
                # action[13:] = 0.
                target_dof_pos = default_angles + action * action_scale #custom_init_angles #
            # else:
            #     tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            #     # === ASAP对齐：clip torque ===
            #     if clip_torques:
            #         tau = np.clip(tau, -torque_limits, torque_limits)
            #     d.ctrl[:] = tau
            #     # mj_step can be replaced with code that also evaluates
            #     # a policy and applies a control signal before stepping the physics.
            #     mujoco.mj_step(m, d)
            # counter += 1

            # #Z === 添加外部干扰 ===
            # if counter % 5 == 0:  # 每500步添加一次随机推力
            #     # 模拟外部推力干扰
            #     push_force = np.random.uniform(-80, 80, 3)  # 随机推力
            #     d.qfrc_applied[0:3] = push_force
            #     print("push_force:", push_force)

            # #Z === 添加地面摩擦力变化 ===
            # if counter % 1000 == 0:  # 每1000步改变地面参数
            #     # 随机改变地面摩擦系数
            #     friction_coeff = np.random.uniform(0.3, 1.2)
            #     m.geom_friction[0, 0] = friction_coeff  # 修改地面摩擦

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
