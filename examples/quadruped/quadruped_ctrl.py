# Description: This script is used to simulate the full model of the robot in mujoco

# Ported from [Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC)

import time
import numpy as np
from tqdm import tqdm
import argparse

import os
import sys

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(current_file_path))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)


from envs.quadruped.quadruped_env import QuadrupedEnv
from envs.quadruped import config as cfg
from envs.orca_gym_env import ActionSpaceType
from gymnasium.envs.registration import register
import gymnasium as gym
from envs.quadruped.utils.mujoco.visual import render_vector
from envs.quadruped.utils.quadruped_utils import LegsAttr
from envs.quadruped.helpers.foothold_reference_generator import FootholdReferenceGenerator
from envs.quadruped.helpers.periodic_gait_generator import PeriodicGaitGenerator
from envs.quadruped.helpers.swing_trajectory_controller import SwingTrajectoryController
from envs.quadruped.helpers.terrain_estimator import TerrainEstimator
from envs.quadruped.helpers.quadruped_utils import plot_swing_mujoco

if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
    from envs.quadruped.sensors.heightmap import HeightMap
    from envs.quadruped.helpers.visual_foothold_adaptation import VisualFootholdAdaptation
from envs.quadruped.utils.mujoco.visual import render_sphere




def register_env(grpc_address, agent_name,
                 robot_name, hip_height, robot_leg_joints, 
                 robot_feet_geom_names, robot_hip_body_names, scene_name, simulation_dt, state_observables_names):
    print("register_env: ", grpc_address)
    gym.register(
        id=f"Quadruped-v0-OrcaGym-{grpc_address[-2:]}",
        entry_point="envs.quadruped.quadruped_env:QuadrupedEnv",
        kwargs={'frame_skip': 1,   # 1 action per frame
                'reward_type': "dense",
                'action_space_type': ActionSpaceType.CONTINUOUS,
                'action_step_count': 0,
                'grpc_address': grpc_address, 
                'agent_names': [agent_name], 
                'time_step': simulation_dt,
                'robot': robot_name,
                'hip_height': hip_height,
                'legs_joint_names': robot_leg_joints,  # Joint names of the legs DoF
                'feet_geom_name': robot_feet_geom_names,  # Geom/Frame id of feet
                'hip_body_name' : robot_hip_body_names, # Body names of the hip
                'scene': scene_name,
                'sim_dt': simulation_dt,
                'ref_base_lin_vel': 0.0,  # pass a float for a fixed value
                'ground_friction_coeff': 1.5,  # pass a float for a fixed value
                'base_vel_command_type': "human",  # "forward", "random", "forward+rotate", "human"
                'state_obs_names': state_observables_names,  # Desired quantities in the 'state' vec
                },
        max_episode_steps=sys.maxsize,  # never stop
        reward_threshold=0.0,
    )



if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Simulation Configuration')
    parser.add_argument('--grpc_address', type=str, required=True, help='The gRPC address for the simulation')

    args = parser.parse_args()

    grpc_address = f"{args.grpc_address}:50051"


    robot_name = cfg.robot
    hip_height = cfg.hip_height
    robot_leg_joints = {key: [f"{robot_name}_{joint_name}" for joint_name in value] for key, value in cfg.robot_leg_joints.items()}
    robot_feet_geom_names = {key: f"{robot_name}_{geom_name}" for key, geom_name in cfg.robot_feet_geom_names.items()}
    robot_hip_body_names = {key: f"{robot_name}_{body_name}" for key, body_name in cfg.robot_hip_body_names.items()}
    scene_name = cfg.simulation_params['scene']
    simulation_dt = cfg.simulation_params['dt']

    state_observables_names = ('base_pos', 'base_lin_vel', 'base_ori_euler_xyz', 'base_ori_quat_wxyz', 'base_ang_vel',
                               'qpos_js', 'qvel_js', 'tau_ctrl_setpoint',
                               'feet_pos_base', 'feet_vel_base', 'contact_state', 'contact_forces_base',)

    print("simulation running... , grpc_address: ", grpc_address)
    print("robot_feet_geom_names:", robot_feet_geom_names)
    env_id = f"Quadruped-v0-OrcaGym-{grpc_address[-2:]}"

    register_env(grpc_address, robot_name,
                 robot_name, hip_height, robot_leg_joints, robot_feet_geom_names, robot_hip_body_names,
                 scene_name, simulation_dt, state_observables_names)

    env = gym.make(env_id)        
    print("启动仿真环境")
    
    # Create the quadruped robot environment -----------------------------------------------------------
    # env = QuadrupedEnv(robot=robot_name,
    #                    hip_height=hip_height,
    #                    legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
    #                    feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
    #                    scene=scene_name,
    #                    sim_dt=simulation_dt,
    #                    ref_base_lin_vel=0.0,  # pass a float for a fixed value
    #                    ground_friction_coeff=1.5,  # pass a float for a fixed value
    #                    base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
    #                    state_obs_names=state_observables_names,  # Desired quantities in the 'state' vec
    #                    )
    # env = QuadrupedEnv(robot=robot_name,
    #                    hip_height=hip_height,
    #                    legs_joint_names=LegsAttr(**robot_leg_joints),  # Joint names of the legs DoF
    #                    feet_geom_name=LegsAttr(**robot_feet_geom_names),  # Geom/Frame id of feet
    #                    scene=scene_name,
    #                    sim_dt=simulation_dt,
    #                    ref_base_lin_vel=(-4.0 * hip_height, 4.0 * hip_height),  # pass a float for a fixed value
    #                    ref_base_ang_vel=(-np.pi * 3 / 4, np.pi * 3 / 4),  # pass a float for a fixed value
    #                    ground_friction_coeff=(0.3, 1.5),  # pass a float for a fixed value
    #                    base_vel_command_type="random",  # "forward", "random", "forward+rotate", "human"
    #                    state_obs_names=state_observables_names,  # Desired quantities in the 'state' vec
    #                    )


    # Some robots require a change in the zero joint-space configuration. If provided apply it
    # if cfg.qpos0_js is not None:
    #     env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], cfg.qpos0_js))

    env.reset(random=False)
    # env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType


    # Controller initialization -------------------------------------------------------------
    mpc_frequency = cfg.simulation_params['mpc_frequency']
    mpc_dt = cfg.mpc_params['dt']
    horizon = cfg.mpc_params['horizon']

    # input_rates optimize the delta_GRF (smoooth!)
    # nominal optimize directly the GRF (not smooth)
    # sampling use GPU
    if cfg.mpc_params['type'] == 'nominal':
        from envs.quadruped.controllers.gradient.nominal.centroidal_nmpc_nominal import Acados_NMPC_Nominal

        controller = Acados_NMPC_Nominal()

        if cfg.mpc_params['optimize_step_freq']:
            from envs.quadruped.controllers.gradient.nominal.centroidal_nmpc_gait_adaptive import \
                Acados_NMPC_GaitAdaptive

            batched_controller = Acados_NMPC_GaitAdaptive()

    elif cfg.mpc_params['type'] == 'input_rates':
        from envs.quadruped.controllers.gradient.input_rates.centroidal_nmpc_input_rates import Acados_NMPC_InputRates

        controller = Acados_NMPC_InputRates()

        if cfg.mpc_params['optimize_step_freq']:
            from envs.quadruped.controllers.gradient.nominal.centroidal_nmpc_gait_adaptive import \
                Acados_NMPC_GaitAdaptive

            batched_controller = Acados_NMPC_GaitAdaptive()

    elif cfg.mpc_params['type'] == 'sampling':
        if cfg.mpc_params['optimize_step_freq']:
            from envs.quadruped.controllers.sampling.centroidal_nmpc_jax_gait_adaptive import Sampling_MPC
        else:
            from envs.quadruped.controllers.sampling.centroidal_nmpc_jax import Sampling_MPC

        controller = Sampling_MPC(horizon=horizon,
                                  dt=mpc_dt,
                                  num_parallel_computations=cfg.mpc_params['num_parallel_computations'],
                                  sampling_method=cfg.mpc_params['sampling_method'],
                                  control_parametrization=cfg.mpc_params['control_parametrization'],
                                  device="gpu")


    # Periodic gait generator --------------------------------------------------------------
    gait_name = cfg.simulation_params['gait']
    gait_params = cfg.simulation_params['gait_params'][gait_name]
    gait_type, duty_factor, step_frequency = gait_params['type'], gait_params['duty_factor'], gait_params['step_freq']
    # Given the possibility to use nonuniform discretization,
    # we generate a contact sequence two times longer and with a dt half of the one of the mpc
    pgg = PeriodicGaitGenerator(duty_factor=duty_factor,
                                step_freq=step_frequency,
                                gait_type=gait_type,
                                horizon=horizon)
    # in the case of nonuniform discretization, we create a list of dts and horizons for each nonuniform discretization
    if (cfg.mpc_params['use_nonuniform_discretization']):
        contact_sequence_dts = [cfg.mpc_params['dt_fine_grained'], mpc_dt]
        contact_sequence_lenghts = [cfg.mpc_params['horizon_fine_grained'], horizon]
    else:
        contact_sequence_dts = [mpc_dt]
        contact_sequence_lenghts = [horizon]
    contact_sequence = pgg.compute_contact_sequence(contact_sequence_dts=contact_sequence_dts, 
                                                    contact_sequence_lenghts=contact_sequence_lenghts)
    nominal_sample_freq = pgg.step_freq
    

    # Create the foothold reference generator ------------------------------------------------
    stance_time = (1 / pgg.step_freq) * pgg.duty_factor
    frg = FootholdReferenceGenerator(stance_time=stance_time, hip_height=cfg.hip_height, lift_off_positions=env.feet_pos(frame='world'))


    # Create swing trajectory generator ------------------------------------------------------
    step_height = cfg.simulation_params['step_height']
    swing_period = (1 - pgg.duty_factor) * (1 / pgg.step_freq) 
    position_gain_fb = cfg.simulation_params['swing_position_gain_fb']
    velocity_gain_fb = cfg.simulation_params['swing_velocity_gain_fb']
    swing_generator = cfg.simulation_params['swing_generator']
    stc = SwingTrajectoryController(step_height=step_height, swing_period=swing_period,
                                    position_gain_fb=position_gain_fb, velocity_gain_fb=velocity_gain_fb,
                                    generator=swing_generator)


    # Terrain estimator -----------------------------------------------------------------------
    terrain_computation = TerrainEstimator()


    # Initialization of variables used in the main control loop --------------------------------
    # Set the reference for the state
    ref_pose = np.array([0, 0, cfg.hip_height])
    ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
    ref_orientation = np.array([0.0, 0.0, 0.0])
    
    # TODO: I would suggest to create a DataClass for "BaseConfig" used in the PotatoModel controllers.
    ref_state = {}

    # Starting contact sequence
    previous_contact = np.array([1, 1, 1, 1])
    previous_contact_mpc = np.array([1, 1, 1, 1])
    current_contact = np.array([1, 1, 1, 1])

    nmpc_GRFs = np.zeros((12,))
    nmpc_wrenches = np.zeros((6,))
    nmpc_footholds = np.zeros((12,))

    # Jacobian matrices
    jac_feet_prev = LegsAttr(*[np.zeros((3, env.model.nv)) for _ in range(4)])
    jac_feet_dot = LegsAttr(*[np.zeros((3, env.model.nv)) for _ in range(4)])
    # Torque vector
    tau = LegsAttr(*[np.zeros((env.model.nv, 1)) for _ in range(4)])
    # State
    state_current, state_prev = {}, {}
    feet_pos = None
    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]


    # Create HeightMap -----------------------------------------------------------------------
    if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
        resolution_vfa = 0.04
        dimension_vfa = 7
        heightmaps = LegsAttr(FL=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        FR=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        RL=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        RR=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData))
        vfa = VisualFootholdAdaptation(legs_order=legs_order, adaptation_strategy=cfg.simulation_params['visual_foothold_adaptation'])



    # --------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    N_EPISODES = 500
    N_STEPS_PER_EPISODE = 2000 if env.base_vel_command_type != "human" else 20000
    last_render_time = time.time()

    state_obs_history, ctrl_state_history = [], []
    for episode_num in tqdm(range(N_EPISODES), desc="Episodes"):

        ep_state_obs_history, ep_ctrl_state_history = [], []
        for _ in range(N_STEPS_PER_EPISODE):
            try:
                step_timestep_0 = time.time()

                # Update the robot state --------------------------------
                feet_pos = env.feet_pos(frame='world')
                hip_pos = env.hip_positions(frame='world')
                base_lin_vel = env.base_lin_vel(frame='world')
                base_ang_vel = env.base_ang_vel(frame='world')

                state_current = dict(
                    position=env.base_pos,
                    linear_velocity=base_lin_vel,
                    orientation=env.base_ori_euler_xyz,
                    angular_velocity=base_ang_vel,
                    foot_FL=feet_pos.FL,
                    foot_FR=feet_pos.FR,
                    foot_RL=feet_pos.RL,
                    foot_RR=feet_pos.RR
                    )


                # Update the desired contact sequence ---------------------------
                pgg.run(simulation_dt, pgg.step_freq)
                contact_sequence = pgg.compute_contact_sequence(contact_sequence_dts=contact_sequence_dts, 
                                                        contact_sequence_lenghts=contact_sequence_lenghts)


                previous_contact = current_contact
                current_contact = np.array([contact_sequence[0][0],
                                            contact_sequence[1][0],
                                            contact_sequence[2][0],
                                            contact_sequence[3][0]])


                # Compute the reference for the footholds ---------------------------------------------------
                frg.update_lift_off_positions(previous_contact, current_contact, feet_pos, legs_order)
                ref_feet_pos = frg.compute_footholds_reference(
                    com_position=env.base_pos,
                    base_ori_euler_xyz=env.base_ori_euler_xyz,
                    base_xy_lin_vel=base_lin_vel[0:2],
                    ref_base_xy_lin_vel=ref_base_lin_vel[0:2],
                    hips_position=hip_pos,
                    com_height_nominal=cfg.simulation_params['ref_z'])
                
                step_timestep_1 = time.time()

                # Adjust the footholds given the terrain -----------------------------------------------------
                if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
                    
                    time_adaptation = time.time()
                    if(stc.check_apex_condition(current_contact, interval=0.01) and vfa.initialized == False):
                        for leg_id, leg_name in enumerate(legs_order):
                            heightmaps[leg_name].update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
                        vfa.compute_adaptation(legs_order, ref_feet_pos, hip_pos, heightmaps, base_lin_vel, env.base_ori_euler_xyz, base_ang_vel)
                        #print("Adaptation time: ", time.time() - time_adaptation)
                    
                    if(stc.check_full_stance_condition(current_contact)):
                        vfa.reset()
                    
                    ref_feet_pos = vfa.get_footholds_adapted(ref_feet_pos)
                    
                    """step_height_modification = False
                    for leg_id, leg_name in enumerate(legs_order):
                        if(ref_feet_pos[leg_name][2] - frg.lift_off_positions[leg_name][2] > (cfg.simulation_params['step_height'] - 0.05) 
                        and ref_feet_pos[leg_name][2] - frg.lift_off_positions[leg_name][2] > 0.0):
                            step_height = ref_feet_pos[leg_name][2] - frg.lift_off_positions[leg_name][2] + 0.05
                            step_height_modification = True
                            stc.regenerate_swing_trajectory_generator(step_height=step_height, swing_period=stc.swing_period)
                    if(step_height_modification == False):
                        step_height = cfg.simulation_params['step_height']
                        stc.regenerate_swing_trajectory_generator(step_height=step_height, swing_period=stc.swing_period)"""
                


                # Estimate the terrain slope and elevation -------------------------------------------------------
                terrain_roll, \
                    terrain_pitch, \
                    terrain_height = terrain_computation.compute_terrain_estimation(
                    base_position=env.base_pos,
                    yaw=env.base_ori_euler_xyz[2],
                    feet_pos=frg.lift_off_positions,
                    current_contact=current_contact)

                ref_pos = np.array([0, 0, cfg.hip_height])
                ref_pos[2] = cfg.simulation_params['ref_z'] + terrain_height


                # Update state reference ------------------------------------------------------------------------
                ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

                
                ref_state |= dict(ref_foot_FL=ref_feet_pos.FL.reshape((1, 3)),
                                ref_foot_FR=ref_feet_pos.FR.reshape((1, 3)),
                                ref_foot_RL=ref_feet_pos.RL.reshape((1, 3)),
                                ref_foot_RR=ref_feet_pos.RR.reshape((1, 3)),
                                # Also update the reference base linear velocity and
                                ref_linear_velocity=ref_base_lin_vel,
                                ref_angular_velocity=ref_base_ang_vel,
                                ref_orientation=np.array([terrain_roll, terrain_pitch, 0.0]),
                                ref_position=ref_pos
                                )
                # -------------------------------------------------------------------------------------------------
                step_timestep_2 = time.time()

                # TODO: this should be hidden inside the controller forward/get_action method
                # Solve OCP ---------------------------------------------------------------------------------------
                if env.step_num % round(1 / (mpc_frequency * simulation_dt)) == 0:

                    time_start = time.time()

                    # We can recompute the inertia of the single rigid body model
                    # or use the fixed one in cfg.py
                    if(cfg.simulation_params['use_inertia_recomputation']):
                        inertia = env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
                        # np.set_printoptions(precision=10, suppress=False)
                        # print("Inertia: ", env.get_base_inertia())
                        # raise NotImplementedError
                    else:
                        inertia = cfg.inertia.flatten()

                    if ((cfg.mpc_params['optimize_step_freq'])):
                        # we can always optimize the step freq, or just at the apex of the swing
                        # to avoid possible jittering in the solution
                        #optimize_swing = 1  
                        optimize_swing = stc.check_apex_condition(current_contact)
                        if(optimize_swing == 1):
                            nominal_sample_freq = pgg.step_freq
                    else:
                        optimize_swing = 0


                    # If we use sampling
                    if (cfg.mpc_params['type'] == 'sampling'):

                        # Convert data to jax and shift previous solution
                        state_current_jax, \
                        reference_state_jax, = controller.prepare_state_and_reference(state_current,
                                                                                    ref_state,
                                                                                    current_contact,
                                                                                    previous_contact_mpc)
                        previous_contact_mpc = current_contact

                        for iter_sampling in range(cfg.mpc_params['num_sampling_iterations']):
                            controller = controller.with_newkey()
                            if (cfg.mpc_params['sampling_method'] == 'cem_mppi'):
                                if (iter_sampling == 0):
                                    controller = controller.with_newsigma(cfg.mpc_params['sigma_cem_mppi'])

                                nmpc_GRFs, \
                                nmpc_footholds, \
                                controller.best_control_parameters, \
                                best_cost, \
                                best_sample_freq, \
                                costs, \
                                sigma_cem_mppi = controller.jitted_compute_control(state_current_jax, reference_state_jax,
                                                                            contact_sequence, controller.best_control_parameters,
                                                                            controller.master_key, controller.sigma_cem_mppi)
                                controller = controller.with_newsigma(sigma_cem_mppi)
                            else:
                                nmpc_GRFs, \
                                nmpc_footholds, \
                                controller.best_control_parameters, \
                                best_cost, \
                                best_sample_freq, \
                                costs = controller.jitted_compute_control(state_current_jax, reference_state_jax,
                                                                contact_sequence, controller.best_control_parameters,
                                                                controller.master_key, pgg.phase_signal,
                                                                nominal_sample_freq, optimize_swing)


                        nmpc_footholds = ref_feet_pos
                        nmpc_GRFs = np.array(nmpc_GRFs)


                    # If we use Gradient-Based MPC
                    else:

                        nmpc_GRFs, \
                        nmpc_footholds, _, \
                        status = controller.compute_control(state_current,
                                                            ref_state,
                                                            contact_sequence,
                                                            inertia=inertia)


                        nmpc_footholds = LegsAttr(FL=nmpc_footholds[0],
                                                FR=nmpc_footholds[1],
                                                RL=nmpc_footholds[2],
                                                RR=nmpc_footholds[3])


                        if cfg.mpc_params['optimize_step_freq'] and optimize_swing == 1:
                            contact_sequence_temp = np.zeros((len(cfg.mpc_params['step_freq_available']), 4, horizon))
                            for j in range(len(cfg.mpc_params['step_freq_available'])):
                                pgg_temp = PeriodicGaitGenerator(duty_factor=pgg.duty_factor,
                                                                step_freq=cfg.mpc_params['step_freq_available'][j],
                                                                gait_type=pgg.gait_type,
                                                                horizon=horizon)
                                pgg_temp.set_phase_signal(pgg.phase_signal)
                                contact_sequence_temp[j] = pgg_temp.compute_contact_sequence(contact_sequence_dts=contact_sequence_dts, 
                                                                                                contact_sequence_lenghts=contact_sequence_lenghts)


                            costs, \
                            best_sample_freq = batched_controller.compute_batch_control(state_current,
                                                                                        ref_state,
                                                                                        contact_sequence_temp)

                        # If the controller is using RTI, we need to linearize the mpc after its computation
                        # this helps to minize the delay between new state->control, but only in a real case.
                        # Here we are in simulation and does not make any difference for now
                        if (controller.use_RTI):
                            # preparation phase
                            controller.acados_ocp_solver.options_set('rti_phase', 1)
                            status = controller.acados_ocp_solver.solve()
                            # print("preparation phase time: ", controller.acados_ocp_solver.get_stats('time_tot'))
                    
                    
                    # If we have optimized the gait, we set all the timing parameters
                    if ((cfg.mpc_params['optimize_step_freq']) and (optimize_swing == 1)):
                        pgg.step_freq = np.array([best_sample_freq])[0]
                        nominal_sample_freq = pgg.step_freq
                        frg.stance_time = (1 / pgg.step_freq) * pgg.duty_factor
                        swing_period = (1 - pgg.duty_factor) * (1 / pgg.step_freq)
                        stc.regenerate_swing_trajectory_generator(step_height=step_height, swing_period=swing_period)


                    # TODO: Indexing should not be hardcoded. Env should provide indexing of leg actuator dimensions.
                    nmpc_GRFs = LegsAttr(FL=nmpc_GRFs[0:3] * current_contact[0],
                                        FR=nmpc_GRFs[3:6] * current_contact[1],
                                        RL=nmpc_GRFs[6:9] * current_contact[2],
                                        RR=nmpc_GRFs[9:12] * current_contact[3])

                step_timestep_3 = time.time()

                # Compute Stance Torque ---------------------------------------------------------------------------
                feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)

                step_timestep_4 = time.time()

                # Compute feet velocities
                feet_vel = LegsAttr(**{leg_name: feet_jac[leg_name] @ env.data.qvel for leg_name in legs_order})
                # Compute jacobian derivatives of the contact points
                jac_feet_dot = (feet_jac - jac_feet_prev) / simulation_dt  # Finite difference approximation
                jac_feet_prev = feet_jac  # Update previous Jacobians
                # Compute the torque with the contact jacobian (-J.T @ f)   J: R^nv -> R^3,   f: R^3
                tau.FL = -np.matmul(feet_jac.FL[:, env.legs_qvel_idx.FL].T, nmpc_GRFs.FL)
                tau.FR = -np.matmul(feet_jac.FR[:, env.legs_qvel_idx.FR].T, nmpc_GRFs.FR)
                tau.RL = -np.matmul(feet_jac.RL[:, env.legs_qvel_idx.RL].T, nmpc_GRFs.RL)
                tau.RR = -np.matmul(feet_jac.RR[:, env.legs_qvel_idx.RR].T, nmpc_GRFs.RR)


                # Compute Swing Torque ------------------------------------------------------------------------------
                # TODO: Move contact sequence to labels FL, FR, RL, RR instead of a fixed indexing.
                # The swing controller is in the end-effector space. For its computation,
                # we save for simplicity joints position and velocities
                qpos, qvel = env.data.qpos, env.data.qvel
                # centrifugal, coriolis, gravity
                legs_mass_matrix = env.legs_mass_matrix
                legs_qfrc_bias = env.legs_qfrc_bias

                stc.update_swing_time(current_contact, legs_order, simulation_dt)

                for leg_id, leg_name in enumerate(legs_order):
                    if current_contact[leg_id] == 0:  # If in swing phase, compute the swing trajectory tracking control.
                        tau[leg_name], _, _ = stc.compute_swing_control(
                            leg_id=leg_id,
                            q_dot=qvel[env.legs_qvel_idx[leg_name]],
                            J=feet_jac[leg_name][:, env.legs_qvel_idx[leg_name]],
                            J_dot=jac_feet_dot[leg_name][:, env.legs_qvel_idx[leg_name]],
                            lift_off=frg.lift_off_positions[leg_name],
                            touch_down=nmpc_footholds[leg_name],
                            foot_pos=feet_pos[leg_name],
                            foot_vel=feet_vel[leg_name],
                            h=legs_qfrc_bias[leg_name],
                            mass_matrix=legs_mass_matrix[leg_name]
                            )
                        
                
                # Set control and mujoco step ----------------------------------------------------------------------
                action = np.zeros(env.model.nu)
                action[env.legs_tau_idx.FL] = tau.FL
                action[env.legs_tau_idx.FR] = tau.FR
                action[env.legs_tau_idx.RL] = tau.RL
                action[env.legs_tau_idx.RR] = tau.RR

                action_noise = np.random.normal(0, 2, size=env.model.nu)
                action += action_noise

                step_timestep_5 = time.time()

                state, reward, is_terminated, is_truncated, info = env.step(action=action)
            
                step_timestep_6 = time.time()


                # Store the history of observations and control -------------------------------------------------------
                ep_state_obs_history.append(state)
                base_lin_vel_err = ref_base_lin_vel - base_lin_vel
                base_ang_vel_err = ref_base_ang_vel - base_ang_vel
                base_poz_z_err = ref_pos[2] - env.base_pos[2]
                ctrl_state = np.concatenate((base_lin_vel_err, base_ang_vel_err, [base_poz_z_err], pgg._phase_signal))
                ep_ctrl_state_history.append(ctrl_state)


                # Render only at a certain frequency -----------------------------------------------------------------
                if False: # time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1:
                    _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

                    # Plot the swing trajectory
                    feet_traj_geom_ids = plot_swing_mujoco(viewer=env.viewer,
                                                        swing_traj_controller=stc,
                                                        swing_period=stc.swing_period,
                                                        swing_time=LegsAttr(FL=stc.swing_time[0],
                                                                            FR=stc.swing_time[1],
                                                                            RL=stc.swing_time[2],
                                                                            RR=stc.swing_time[3]),
                                                        lift_off_positions=frg.lift_off_positions,
                                                        nmpc_footholds=nmpc_footholds,
                                                        ref_feet_pos=ref_feet_pos,
                                                        geom_ids=feet_traj_geom_ids)
                    
                    
                    # Update and Plot the heightmap
                    if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
                        #if(stc.check_apex_condition(current_contact, interval=0.01)):
                        for leg_id, leg_name in enumerate(legs_order):
                            data = heightmaps[leg_name].data#.update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
                            if(data is not None):
                                for i in range(data.shape[0]):
                                    for j in range(data.shape[1]):
                                            heightmaps[leg_name].geom_ids[i, j] = render_sphere(viewer=env.viewer,
                                                                                                position=([data[i][j][0][0],data[i][j][0][1],data[i][j][0][2]]),
                                                                                                diameter=0.01,
                                                                                                color=[0, 1, 0, .5],
                                                                                                geom_id=heightmaps[leg_name].geom_ids[i, j]
                                                                                                )
                                
                    # Plot the GRF
                    for leg_id, leg_name in enumerate(legs_order):
                        feet_GRF_geom_ids[leg_name] = render_vector(env.viewer,
                                                                    vector=feet_GRF[leg_name],
                                                                    pos=feet_pos[leg_name],
                                                                    scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                                                                    color=np.array([0, 1, 0, .5]),
                                                                    geom_id=feet_GRF_geom_ids[leg_name])

                    env.render()
                    last_render_time = time.time()


                # Reset the environment if the episode is terminated ------------------------------------------------
                if env.step_num > N_STEPS_PER_EPISODE or is_terminated or is_truncated:
                    if is_terminated:
                        print("Environment terminated")
                    else:
                        state_obs_history.append(ep_state_obs_history)
                        ctrl_state_history.append(ep_ctrl_state_history)
                    env.reset(random=False)
                    pgg.reset()
                    if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'): vfa.reset()
                    frg.lift_off_positions = env.feet_pos(frame='world')
                    current_contact = np.array([0, 0, 0, 0])
                    previous_contact = np.asarray(current_contact)

                step_timestep_7 = time.time()

                print("Step time: ", (step_timestep_7 - step_timestep_0) * 1000)

                # print("Step times: ", 
                #         "0-1:", (step_timestep_1 - step_timestep_0) * 1000, 
                #         "1-2:", (step_timestep_2 - step_timestep_1) * 1000,
                #         "2-3:", (step_timestep_3 - step_timestep_2) * 1000,
                #         "3-4:", (step_timestep_4 - step_timestep_3) * 1000,
                #         "4-5:", (step_timestep_5 - step_timestep_4) * 1000,
                #         "5-6:", (step_timestep_6 - step_timestep_5) * 1000,
                #         "6-7:", (step_timestep_7 - step_timestep_6) * 1000
                #         )

                frame_time = time.time() - step_timestep_0
                delta_time = simulation_dt - frame_time
                if (delta_time > 0):
                    time.sleep(delta_time)

            except KeyboardInterrupt:
                print("Simulation stopped")        
                env.set_time_step(0.01666666666666666)
                env.close()
                

    env.close()


