# API Flat List: `orca_gym/environment`

本页为自动生成的符号清单（class/function/method 名称），用于快速检索。

- 仅列出源码里的符号名，不依赖 `examples/` 或 `envs/`。

- class 方法只列出 **public method**（不以下划线开头）。

## `orca_gym/environment/async_env/orca_gym_async_agent.py`

### Classes

- `OrcaGymAsyncAgent`
  - `OrcaGymAsyncAgent.dt()`
  - `OrcaGymAsyncAgent.name()`
  - `OrcaGymAsyncAgent.name_space()`
  - `OrcaGymAsyncAgent.name_space_list()`
  - `OrcaGymAsyncAgent.joint_names()`
  - `OrcaGymAsyncAgent.actuator_names()`
  - `OrcaGymAsyncAgent.site_names()`
  - `OrcaGymAsyncAgent.sensor_names()`
  - `OrcaGymAsyncAgent.nu()`
  - `OrcaGymAsyncAgent.nq()`
  - `OrcaGymAsyncAgent.nv()`
  - `OrcaGymAsyncAgent.truncated()`
  - `OrcaGymAsyncAgent.ctrl_start()`
  - `OrcaGymAsyncAgent.action_range()`
  - `OrcaGymAsyncAgent.kps()`
  - `OrcaGymAsyncAgent.kds()`
  - `OrcaGymAsyncAgent.get_obs()`
  - `OrcaGymAsyncAgent.init_ctrl_info()`
  - `OrcaGymAsyncAgent.get_ctrl_info()`
  - `OrcaGymAsyncAgent.init_joint_index()`
  - `OrcaGymAsyncAgent.set_action_space()`
  - `OrcaGymAsyncAgent.on_step()`
  - `OrcaGymAsyncAgent.step()`
  - `OrcaGymAsyncAgent.on_reset()`
  - `OrcaGymAsyncAgent.reset()`
  - `OrcaGymAsyncAgent.is_success()`
  - `OrcaGymAsyncAgent.is_terminated()`
  - `OrcaGymAsyncAgent.compute_reward()`
  - `OrcaGymAsyncAgent.set_init_state()`
  - `OrcaGymAsyncAgent.get_action_size()`
  - `OrcaGymAsyncAgent.compute_torques()`
  - `OrcaGymAsyncAgent.setup_curriculum()`

## `orca_gym/environment/async_env/orca_gym_async_env.py`

### Classes

- `OrcaGymAsyncEnv`
  - `OrcaGymAsyncEnv.set_obs_space()`
  - `OrcaGymAsyncEnv.set_action_space()`
  - `OrcaGymAsyncEnv.initialize_agents()`
  - `OrcaGymAsyncEnv.get_obs()`
  - `OrcaGymAsyncEnv.reset_agents()`
  - `OrcaGymAsyncEnv.step_agents()`
  - `OrcaGymAsyncEnv.step()`
  - `OrcaGymAsyncEnv.reset_model()`
  - `OrcaGymAsyncEnv.get_observation()`
  - `OrcaGymAsyncEnv.init_agent_joint_index()`
  - `OrcaGymAsyncEnv.reorder_agents()`
  - `OrcaGymAsyncEnv.generate_action_scale_array()`
  - `OrcaGymAsyncEnv.setup_curriculum()`

## `orca_gym/environment/async_env/orca_gym_vector_env.py`

### Classes

- `OrcaGymVectorEnv`
  - `OrcaGymVectorEnv.reset()`
  - `OrcaGymVectorEnv.step()`
  - `OrcaGymVectorEnv.render()`
  - `OrcaGymVectorEnv.close()`

## `orca_gym/environment/async_env/single_agent_env_runner.py`

### Classes

- `OrcaGymAsyncSingleAgentEnvRunner`
  - `OrcaGymAsyncSingleAgentEnvRunner.make_env()`

## `orca_gym/environment/async_env/subproc_vec_env.py`

### Classes

- `OrcaGymAsyncSubprocVecEnv`
  - `OrcaGymAsyncSubprocVecEnv.step_async()`
  - `OrcaGymAsyncSubprocVecEnv.step_wait()`
  - `OrcaGymAsyncSubprocVecEnv.reset()`
  - `OrcaGymAsyncSubprocVecEnv.close()`
  - `OrcaGymAsyncSubprocVecEnv.get_images()`
  - `OrcaGymAsyncSubprocVecEnv.get_attr()`
  - `OrcaGymAsyncSubprocVecEnv.set_attr()`
  - `OrcaGymAsyncSubprocVecEnv.env_method()`
  - `OrcaGymAsyncSubprocVecEnv.env_is_wrapped()`
  - `OrcaGymAsyncSubprocVecEnv.setup_curriculum()`

## `orca_gym/environment/orca_gym_env.py`

### Classes

- `RewardType`
- `OrcaGymBaseEnv`
  - `OrcaGymBaseEnv.step()`
  - `OrcaGymBaseEnv.reset_model()`
  - `OrcaGymBaseEnv.initialize_simulation()`
  - `OrcaGymBaseEnv.render()`
  - `OrcaGymBaseEnv.generate_action_space()`
  - `OrcaGymBaseEnv.generate_observation_space()`
  - `OrcaGymBaseEnv.reset()`
  - `OrcaGymBaseEnv.set_seed_value()`
  - `OrcaGymBaseEnv.body()`
  - `OrcaGymBaseEnv.joint()`
  - `OrcaGymBaseEnv.actuator()`
  - `OrcaGymBaseEnv.site()`
  - `OrcaGymBaseEnv.mocap()`
  - `OrcaGymBaseEnv.sensor()`
  - `OrcaGymBaseEnv.dt()`
  - `OrcaGymBaseEnv.agent_num()`
  - `OrcaGymBaseEnv.do_simulation()`
  - `OrcaGymBaseEnv.close()`
  - `OrcaGymBaseEnv.initialize_grpc()`
  - `OrcaGymBaseEnv.pause_simulation()`
  - `OrcaGymBaseEnv.init_qpos_qvel()`
  - `OrcaGymBaseEnv.reset_simulation()`
  - `OrcaGymBaseEnv.set_time_step()`

## `orca_gym/environment/orca_gym_local_env.py`

### Classes

- `OrcaGymLocalEnv`
  - `OrcaGymLocalEnv.initialize_simulation()`
  - `OrcaGymLocalEnv.initialize_grpc()`
  - `OrcaGymLocalEnv.pause_simulation()`
  - `OrcaGymLocalEnv.close()`
  - `OrcaGymLocalEnv.get_body_manipulation_anchored()`
  - `OrcaGymLocalEnv.begin_save_video()`
  - `OrcaGymLocalEnv.stop_save_video()`
  - `OrcaGymLocalEnv.get_next_frame()`
  - `OrcaGymLocalEnv.get_current_frame()`
  - `OrcaGymLocalEnv.get_camera_time_stamp()`
  - `OrcaGymLocalEnv.get_frame_png()`
  - `OrcaGymLocalEnv.get_body_manipulation_movement()`
  - `OrcaGymLocalEnv.do_simulation()`
  - `OrcaGymLocalEnv.render_mode()`
  - `OrcaGymLocalEnv.is_subenv()`
  - `OrcaGymLocalEnv.sync_render()`
  - `OrcaGymLocalEnv.render()`
  - `OrcaGymLocalEnv.do_body_manipulation()`
  - `OrcaGymLocalEnv.release_body_anchored()`
  - `OrcaGymLocalEnv.anchor_actor()`
  - `OrcaGymLocalEnv.update_anchor_equality_constraints()`
  - `OrcaGymLocalEnv.set_ctrl()`
  - `OrcaGymLocalEnv.mj_step()`
  - `OrcaGymLocalEnv.mj_forward()`
  - `OrcaGymLocalEnv.mj_jacBody()`
  - `OrcaGymLocalEnv.mj_jacSite()`
  - `OrcaGymLocalEnv.set_time_step()`
  - `OrcaGymLocalEnv.update_data()`
  - `OrcaGymLocalEnv.reset_simulation()`
  - `OrcaGymLocalEnv.init_qpos_qvel()`
  - `OrcaGymLocalEnv.query_joint_offsets()`
  - `OrcaGymLocalEnv.query_joint_lengths()`
  - `OrcaGymLocalEnv.get_body_xpos_xmat_xquat()`
  - `OrcaGymLocalEnv.query_sensor_data()`
  - `OrcaGymLocalEnv.query_joint_qpos()`
  - `OrcaGymLocalEnv.query_joint_qvel()`
  - `OrcaGymLocalEnv.query_joint_qacc()`
  - `OrcaGymLocalEnv.jnt_qposadr()`
  - `OrcaGymLocalEnv.jnt_dofadr()`
  - `OrcaGymLocalEnv.query_site_pos_and_mat()`
  - `OrcaGymLocalEnv.query_site_pos_and_quat()`
  - `OrcaGymLocalEnv.query_site_size()`
  - `OrcaGymLocalEnv.query_site_pos_and_quat_B()`
  - `OrcaGymLocalEnv.set_joint_qpos()`
  - `OrcaGymLocalEnv.set_joint_qvel()`
  - `OrcaGymLocalEnv.query_site_xvalp_xvalr()`
  - `OrcaGymLocalEnv.query_site_xvalp_xvalr_B()`
  - `OrcaGymLocalEnv.update_equality_constraints()`
  - `OrcaGymLocalEnv.set_mocap_pos_and_quat()`
  - `OrcaGymLocalEnv.query_contact_simple()`
  - `OrcaGymLocalEnv.set_geom_friction()`
  - `OrcaGymLocalEnv.add_extra_weight()`
  - `OrcaGymLocalEnv.query_contact_force()`
  - `OrcaGymLocalEnv.get_cfrc_ext()`
  - `OrcaGymLocalEnv.query_actuator_torques()`
  - `OrcaGymLocalEnv.query_joint_dofadrs()`
  - `OrcaGymLocalEnv.get_goal_bounding_box()`
  - `OrcaGymLocalEnv.query_velocity_body_B()`
  - `OrcaGymLocalEnv.query_position_body_B()`
  - `OrcaGymLocalEnv.query_orientation_body_B()`
  - `OrcaGymLocalEnv.query_joint_axes_B()`
  - `OrcaGymLocalEnv.query_robot_velocity_odom()`
  - `OrcaGymLocalEnv.query_robot_position_odom()`
  - `OrcaGymLocalEnv.query_robot_orientation_odom()`
  - `OrcaGymLocalEnv.set_actuator_trnid()`
  - `OrcaGymLocalEnv.disable_actuator()`
  - `OrcaGymLocalEnv.load_content_file()`

## `orca_gym/environment/orca_gym_remote_env.py`

### Classes

- `OrcaGymRemoteEnv`
  - `OrcaGymRemoteEnv.initialize_simulation()`
  - `OrcaGymRemoteEnv.do_simulation()`
  - `OrcaGymRemoteEnv.set_qpos_qvel()`
  - `OrcaGymRemoteEnv.render()`
  - `OrcaGymRemoteEnv.get_observation()`
  - `OrcaGymRemoteEnv.close()`
  - `OrcaGymRemoteEnv.get_body_com_dict()`
  - `OrcaGymRemoteEnv.get_body_com_xpos_xmat()`
  - `OrcaGymRemoteEnv.get_body_com_xpos_xmat_list()`
  - `OrcaGymRemoteEnv.get_body_xpos_xmat_xquat()`
  - `OrcaGymRemoteEnv.get_geom_xpos_xmat()`
  - `OrcaGymRemoteEnv.initialize_grpc()`
  - `OrcaGymRemoteEnv.pause_simulation()`
  - `OrcaGymRemoteEnv.init_qpos_qvel()`
  - `OrcaGymRemoteEnv.reset_simulation()`
  - `OrcaGymRemoteEnv.set_ctrl()`
  - `OrcaGymRemoteEnv.mj_forward()`
  - `OrcaGymRemoteEnv.set_time_step()`
  - `OrcaGymRemoteEnv.query_joint_qpos()`
  - `OrcaGymRemoteEnv.query_joint_qvel()`
  - `OrcaGymRemoteEnv.set_joint_qpos()`
  - `OrcaGymRemoteEnv.query_cfrc_ext()`
  - `OrcaGymRemoteEnv.query_actuator_force()`
  - `OrcaGymRemoteEnv.load_keyframe()`
  - `OrcaGymRemoteEnv.query_joint_limits()`
  - `OrcaGymRemoteEnv.query_body_velocities()`
  - `OrcaGymRemoteEnv.query_actuator_gain_prm()`
  - `OrcaGymRemoteEnv.set_actuator_gain_prm()`
  - `OrcaGymRemoteEnv.query_actuator_bias_prm()`
  - `OrcaGymRemoteEnv.set_actuator_bias_prm()`
  - `OrcaGymRemoteEnv.query_mocap_pos_and_quat()`
  - `OrcaGymRemoteEnv.set_mocap_pos_and_quat()`
  - `OrcaGymRemoteEnv.query_site_pos_and_mat()`
  - `OrcaGymRemoteEnv.query_site_pos_and_quat()`
  - `OrcaGymRemoteEnv.query_site_xvalp_xvalr()`
  - `OrcaGymRemoteEnv.update_equality_constraints()`
  - `OrcaGymRemoteEnv.query_all_geoms()`
  - `OrcaGymRemoteEnv.query_opt_config()`
  - `OrcaGymRemoteEnv.set_opt_config()`
  - `OrcaGymRemoteEnv.query_contact_simple()`
  - `OrcaGymRemoteEnv.query_contact()`
  - `OrcaGymRemoteEnv.query_contact_force()`
  - `OrcaGymRemoteEnv.mj_jac()`
  - `OrcaGymRemoteEnv.calc_full_mass_matrix()`
  - `OrcaGymRemoteEnv.query_qfrc_bias()`
  - `OrcaGymRemoteEnv.query_subtree_com()`
  - `OrcaGymRemoteEnv.set_geom_friction()`
  - `OrcaGymRemoteEnv.query_sensor_data()`
  - `OrcaGymRemoteEnv.query_joint_offsets()`
