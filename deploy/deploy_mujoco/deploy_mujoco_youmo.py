import time
import mujoco.viewer
import mujoco
import numpy as np
import torch

def quat_rotate_inverse(q, v): # q 是四元数，v 是三维向量 
    q_w = q[-1] # 四元数的实部 
    q_vec = q[:3] # 四元数的虚部（向量部分）
    # 计算旋转后的向量
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = 2.0 * q_w * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.dot(q_vec, v)
    return a - b + c


def pd_control(kps, actions_scaled, default_angles, dof_pos, kds, dof_vel): 
    pos = actions_scaled + default_angles 
    #pos = np.clip(pos, -dof_pos_limit_up, dof_pos_limit_up) 
    torques = kps * KP_SCALE *(pos - dof_pos) - kds * KD_SCALE * dof_vel
    torque_limits = [88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 
                        88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 
                        88.0, 50.0, 50.0,
                        25.0, 25.0, 25.0, 25.0, 
                        25.0, 25.0, 25.0, 25.0]
    torques = torch.from_numpy(np.array(torques))
    torque_limits = torch.from_numpy(np.array(torque_limits))
    return torch.clip(torques, -torque_limits, torque_limits)

if __name__ == "__main__":

    KP_SCALE = 1
    KD_SCALE = 1
    #policy_path = "/home/zsq/ASAP/logs/MotionTracking/20250423_235448-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/exported/model_11500.onnx"
    #policy_path = "/home/zsq/ASAP/logs/MotionTracking/20250504_092036-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/exported/model_11500.onnx"
    policy_path = "/home/zsq/ASAP/logs/MotionTracking/20250504_231249-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/exported/model_26000.onnx"
    xml_path = "/home/zsq/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_raw.xml"

    print("policy_path: ", policy_path)
    print("xml_path: ", xml_path)

    control_interval = 0.005#gym中每次施加力矩的周期
    control_decimation =4  #每3个控制周期4个时间步
    fps = 200
    dt=0.02
    counter = 0
    num_actions = 23
    motion_length=3.933
    clip_action_limit=100
    action = np.zeros(num_actions, dtype=np.float32)
    dof_pos_limit_low = np.array([-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, 
                            -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618, 
                            -2.618, -0.52, -0.52,
                            -3.0892, -1.5882, -2.618, -1.0472, 
                            -3.0892, -2.2515, -2.618, -1.0472], dtype=np.float32)

    dof_pos_limit_up = np.array([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
                            2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 
                            2.618, 0.52, 0.52,
                            2.6704, 2.2515, 2.618, 2.0944,
                            2.6704, 1.5882, 2.618, 2.0944], dtype=np.float32)
    default_angles =  np.array([ -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                0.0, 0.0, 0.0, 
                                0.0, 0.0, 0.0, 0.0, 
                                0.0, 0.0, 0.0, 0.0 ], dtype=np.float32)
    default_angles_before_env=np.array([0.0535,  0.0308,  0.1582, -0.1208,  0.0138, -0.0276,
        -0.0255, -0.0912,  0.0475, -0.0562, -0.0423, -0.0141,  0.0772, -0.0046,
        -0.0338,  0.3369,  0.2924, -0.0845,  0.6023,  0.2206, -0.2428,  0.0613,
        0.5796],dtype=np.float32)
    default_angles_motion_res=np.array([ 
    -0.1040,  0.0146,  0.2722,  0.0424,  0.0014,  0.0000, -0.2361, -0.1315,
        -0.0481,  0.0927,  0.0043,  0.0000,  0.2081, -0.0671, -0.2738,  0.2576,
        0.3911, -0.4247,  0.9946,  0.2217, -0.3084,  0.2673,  0.9876],dtype=np.float32)

    default_vel_before_env = np.array([0.3334,  0.2569,  0.8968, -0.2103,  0.1112, -0.3184,  0.0372,
        0.3178,  1.1418,  0.6026, -0.4792, -0.2446,  0.1068, -0.0358,  0.5959,
        0.1269, -0.1776,  0.5440, -1.3886,  0.1853,  0.2840, -0.6984, -1.5377],dtype=np.float32)
    default_vel_motion_res = np.array([-0.0275, -0.0459,  0.0235, -0.0290, -0.0019,  0.0000, -0.0083,  0.0066,
        0.0007,  0.0030, -0.0002,  0.0000, -0.0481, -0.0216, -0.0235, -0.0966,
        -0.1212, -0.0390, -0.0361, -0.0507,  0.1070,  0.0176, -0.0074],dtype=np.float32)
    default_base_ang_vel_before_env=np.array([0.1505,
        -1.4219, -0.9871], dtype=np.float32)

    kps = np.array([ 100, 100, 100, 200, 20, 20, 
                    100, 100, 100, 200, 20, 20, 
                    400, 400, 400,
                    90,   60,  20, 60, 
                    90,   60,  20, 60 ], dtype=np.float32)
    kds = np.array([ 2.5, 2.5, 2.5, 5.0, 0.2, 0.1, 
                    2.5, 2.5, 2.5, 5.0, 0.2, 0.1, 
                    5.0, 5.0, 5.0, 
                    2.0, 1.0, 0.4, 1.0, 
                    2.0, 1.0, 0.4, 1.0 ], dtype=np.float32)

    dof_vel_limit_list=np.array([32.0, 32.0, 32.0, 20.0, 37.0, 37.0, 
                            32.0, 32.0, 32.0, 20.0, 37.0, 37.0, 
                            32.0, 37.0, 37.0, 
                            37.0, 37.0, 37.0, 37.0, 
                            37.0, 37.0, 37.0, 37.0], dtype=np.float32)
    default_projected_gravity_before_env = np.array([0.1919,  0.0487, -1.0586], dtype=np.float32)
    target_dof_pos = default_angles.copy()
    ref_motion_phase = 0

    history_length = 4  
    lin_vel_buf = np.zeros(history_length * 3, dtype=np.float32)
    ang_vel_buf = np.zeros(history_length * 3, dtype=np.float32)
    proj_g_buf = np.zeros(history_length * 3, dtype=np.float32)
    dof_pos_buf = np.zeros(history_length * 23, dtype=np.float32)
    dof_vel_buf = np.zeros(history_length * 23, dtype=np.float32)
    action_buf = np.zeros(history_length * 23, dtype=np.float32)
    ref_motion_phase_buf = np.zeros(history_length * 1, dtype=np.float32)

    # load onnx model
    onnx_model = onnx.load(policy_path)

    ort_session = onnxruntime.InferenceSession(policy_path)
    input_name = ort_session.get_inputs()[0].name
    episode_length_buf=0
    motion_start_times = 0
    motion_times = 0
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)


    m.opt.timestep = control_interval
    # 设置初始关节位置和速度
    d.qpos[7:7 + len(target_dof_pos)] = default_angles  # 设置关节位置
    d.qvel[6:6 + len(target_dof_pos)] = [0] # 设置关节速度为 0
    # d.qpos[:3] = [0.0, 0.0, 0.8]  # 根部位置 (x, y, z)
    # d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 根部方向（四元数）w,x,y,z！！！
    # d.qvel[:3] = [0.0, 0.0, 0.0]  # 根部线速度
    # d.qvel[3:6] = [0.0, 0.0, 0.0]  # 根部角速度
    test_flag=False


    with mujoco.viewer.launch_passive(m, d) as viewer:
        wait_length = 0.05    
        start = time.time()
        #input("Press Enter to continue...")
        while viewer.is_running():              
            episode_length_buf += 1
            motion_times=((episode_length_buf) * dt+motion_start_times)% motion_length
            #episode_length_buf += 1
            ref_motion_phase = motion_times / motion_length
            print("ref_motion_phase: ", ref_motion_phase)
            qj = d.qpos[7:]   
            dqj = d.qvel[6:]    
            quat = d.qpos[3:7]
            quat = np.concatenate((quat[1:], quat[:1])) 


            # action = np.clip(action, -clip_action_limit, clip_action_limit)
            # actions_scaled = action * 0.25
            # for i in range(control_decimation):
            #     step_start = time.time()                    
                
            #     torque = pd_control(kps, actions_scaled, target_dof_pos, qj, kds, dqj)
            #     d.ctrl[:] = torque
            #     mujoco.mj_step(m, d)

            #     if TIMESTEP:
            #         time_until_next_step = m.opt.timestep - (time.time() - step_start)
            #         if time_until_next_step 0:
            #             time.sleep(time_until_next_step)
            # viewer.sync()


            lin_vel = d.qvel[:3]
            ang_vel = d.qvel[3:6]
            # 限制线速度和角速度
            max_linear_velocity = 1000.0
            max_angular_velocity = 1000.0
            

            projected_gravity = quat_rotate_inverse(quat, np.array([0,0,-1]))######重要
            dof_pos = (qj-default_angles) * 1.0
            dof_vel = dqj * 0.05
            base_ang_vel = ang_vel * 0.25
            base_lin_vel = lin_vel * 2.0  

            history_obs_buf = np.concatenate((action_buf, ang_vel_buf, dof_pos_buf, dof_vel_buf, proj_g_buf, ref_motion_phase_buf), axis=-1, dtype=np.float32)
            obs_buf = np.concatenate((action, base_ang_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, [ref_motion_phase]), axis=-1, dtype=np.float32)
            #第一次交互，action全0，base_ang_vel有数值（3），dof_pos有(23),dof_vel(23)都接近0，projected_gravity(有),ref_motion_phase(有,0.0102，对应2*dt)
            # update history
            ang_vel_buf = np.concatenate((base_ang_vel, ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
            lin_vel_buf = np.concatenate((base_lin_vel, lin_vel_buf[:-3]), axis=-1, dtype=np.float32)
            proj_g_buf = np.concatenate((projected_gravity, proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
            dof_pos_buf = np.concatenate((dof_pos, dof_pos_buf[:-23] ), axis=-1, dtype=np.float32)
            dof_vel_buf = np.concatenate((dof_vel, dof_vel_buf[:-23] ), axis=-1, dtype=np.float32)
            action_buf = np.concatenate((action, action_buf[:-23] ), axis=-1, dtype=np.float32)
            ref_motion_phase_buf = np.concatenate((np.array([ref_motion_phase]), ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)                

            obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
            action = np.squeeze(ort_session.run(None, {input_name: obs_tensor})[0])
            action = np.clip(action, -clip_action_limit, clip_action_limit)
            actions_scaled = action * 0.25
            for i in range(control_decimation):
                step_start = time.time()                    
                
                torque = pd_control(kps, actions_scaled, target_dof_pos, qj, kds, dqj)
                d.ctrl[:] = torque
                mujoco.mj_step(m, d)


                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                viewer.sync()
