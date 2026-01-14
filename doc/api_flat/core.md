# API Flat List: `orca_gym/core`

本页为自动生成的符号清单（class/function/method 名称），用于快速检索。

- 仅列出源码里的符号名，不依赖 `examples/` 或 `envs/`。

- class 方法只列出 **public method**（不以下划线开头）。

## `orca_gym/core/orca_gym.py`

### Classes

- `OrcaGymBase`
  - `OrcaGymBase.pause_simulation()`
  - `OrcaGymBase.print_opt_config()`
  - `OrcaGymBase.print_model_info()`
  - `OrcaGymBase.set_qpos()`
  - `OrcaGymBase.mj_forward()`
  - `OrcaGymBase.mj_inverse()`
  - `OrcaGymBase.mj_step()`
  - `OrcaGymBase.set_qvel()`

## `orca_gym/core/orca_gym_data.py`

### Classes

- `OrcaGymData`
  - `OrcaGymData.update_qpos_qvel_qacc()`
  - `OrcaGymData.update_qfrc_bias()`

## `orca_gym/core/orca_gym_local.py`

### Classes

- `AnchorType`
- `CaptureMode`
- `OrcaGymLocal`
  - `OrcaGymLocal.load_model_xml()`
  - `OrcaGymLocal.init_simulation()`
  - `OrcaGymLocal.render()`
  - `OrcaGymLocal.update_local_env()`
  - `OrcaGymLocal.load_content_file()`
  - `OrcaGymLocal.process_xml_node()`
  - `OrcaGymLocal.begin_save_video()`
  - `OrcaGymLocal.stop_save_video()`
  - `OrcaGymLocal.get_current_frame()`
  - `OrcaGymLocal.get_camera_time_stamp()`
  - `OrcaGymLocal.get_frame_png()`
  - `OrcaGymLocal.xml_file_dir()`
  - `OrcaGymLocal.process_xml_file()`
  - `OrcaGymLocal.load_local_env()`
  - `OrcaGymLocal.get_body_manipulation_anchored()`
  - `OrcaGymLocal.get_body_manipulation_movement()`
  - `OrcaGymLocal.set_time_step()`
  - `OrcaGymLocal.set_opt_timestep()`
  - `OrcaGymLocal.set_timestep_remote()`
  - `OrcaGymLocal.set_opt_config()`
  - `OrcaGymLocal.query_opt_config()`
  - `OrcaGymLocal.query_model_info()`
  - `OrcaGymLocal.query_all_equality_constraints()`
  - `OrcaGymLocal.query_all_mocap_bodies()`
  - `OrcaGymLocal.query_all_actuators()`
  - `OrcaGymLocal.get_goal_bounding_box()`
  - `OrcaGymLocal.set_actuator_trnid()`
  - `OrcaGymLocal.disable_actuator()`
  - `OrcaGymLocal.query_all_bodies()`
  - `OrcaGymLocal.query_all_joints()`
  - `OrcaGymLocal.query_all_geoms()`
  - `OrcaGymLocal.query_all_sites()`
  - `OrcaGymLocal.query_all_sensors()`
  - `OrcaGymLocal.update_data()`
  - `OrcaGymLocal.update_data_external()`
  - `OrcaGymLocal.query_qfrc_bias()`
  - `OrcaGymLocal.load_initial_frame()`
  - `OrcaGymLocal.query_joint_offsets()`
  - `OrcaGymLocal.query_joint_lengths()`
  - `OrcaGymLocal.query_body_xpos_xmat_xquat()`
  - `OrcaGymLocal.query_sensor_data()`
  - `OrcaGymLocal.set_ctrl()`
  - `OrcaGymLocal.mj_step()`
  - `OrcaGymLocal.mj_forward()`
  - `OrcaGymLocal.mj_inverse()`
  - `OrcaGymLocal.mj_fullM()`
  - `OrcaGymLocal.mj_jacBody()`
  - `OrcaGymLocal.mj_jacSite()`
  - `OrcaGymLocal.query_joint_qpos()`
  - `OrcaGymLocal.query_joint_qvel()`
  - `OrcaGymLocal.query_joint_qacc()`
  - `OrcaGymLocal.jnt_qposadr()`
  - `OrcaGymLocal.jnt_dofadr()`
  - `OrcaGymLocal.query_site_pos_and_mat()`
  - `OrcaGymLocal.query_site_size()`
  - `OrcaGymLocal.set_joint_qpos()`
  - `OrcaGymLocal.set_joint_qvel()`
  - `OrcaGymLocal.mj_jac_site()`
  - `OrcaGymLocal.modify_equality_objects()`
  - `OrcaGymLocal.update_equality_constraints()`
  - `OrcaGymLocal.set_mocap_pos_and_quat()`
  - `OrcaGymLocal.query_contact_simple()`
  - `OrcaGymLocal.set_geom_friction()`
  - `OrcaGymLocal.add_extra_weight()`
  - `OrcaGymLocal.query_contact_force()`
  - `OrcaGymLocal.get_cfrc_ext()`
  - `OrcaGymLocal.query_actuator_torques()`
  - `OrcaGymLocal.query_joint_dofadrs()`
  - `OrcaGymLocal.query_velocity_body_B()`
  - `OrcaGymLocal.query_position_body_B()`
  - `OrcaGymLocal.query_orientation_body_B()`
  - `OrcaGymLocal.query_joint_axes_B()`
  - `OrcaGymLocal.query_robot_velocity_odom()`
  - `OrcaGymLocal.query_robot_position_odom()`
  - `OrcaGymLocal.query_robot_orientation_odom()`

### Functions

- `get_qpos_size()`
- `get_dof_size()`
- `get_eq_type()`

## `orca_gym/core/orca_gym_model.py`

### Classes

- `OrcaGymModel`
  - `OrcaGymModel.init_model_info()`
  - `OrcaGymModel.init_eq_list()`
  - `OrcaGymModel.get_eq_list()`
  - `OrcaGymModel.init_mocap_dict()`
  - `OrcaGymModel.init_actuator_dict()`
  - `OrcaGymModel.get_actuator_dict()`
  - `OrcaGymModel.get_actuator_byid()`
  - `OrcaGymModel.get_actuator_byname()`
  - `OrcaGymModel.actuator_name2id()`
  - `OrcaGymModel.actuator_id2name()`
  - `OrcaGymModel.init_body_dict()`
  - `OrcaGymModel.get_body_dict()`
  - `OrcaGymModel.get_body_byid()`
  - `OrcaGymModel.get_body_byname()`
  - `OrcaGymModel.body_name2id()`
  - `OrcaGymModel.body_id2name()`
  - `OrcaGymModel.init_joint_dict()`
  - `OrcaGymModel.get_joint_dict()`
  - `OrcaGymModel.get_joint_byid()`
  - `OrcaGymModel.get_joint_byname()`
  - `OrcaGymModel.joint_name2id()`
  - `OrcaGymModel.joint_id2name()`
  - `OrcaGymModel.init_geom_dict()`
  - `OrcaGymModel.get_geom_dict()`
  - `OrcaGymModel.get_geom_byid()`
  - `OrcaGymModel.get_geom_byname()`
  - `OrcaGymModel.geom_name2id()`
  - `OrcaGymModel.geom_id2name()`
  - `OrcaGymModel.get_body_names()`
  - `OrcaGymModel.get_geom_body_name()`
  - `OrcaGymModel.get_geom_body_id()`
  - `OrcaGymModel.get_actuator_ctrlrange()`
  - `OrcaGymModel.get_joint_qposrange()`
  - `OrcaGymModel.init_site_dict()`
  - `OrcaGymModel.get_site_dict()`
  - `OrcaGymModel.get_site()`
  - `OrcaGymModel.site_name2id()`
  - `OrcaGymModel.site_id2name()`
  - `OrcaGymModel.init_sensor_dict()`
  - `OrcaGymModel.gen_sensor_dict()`
  - `OrcaGymModel.get_sensor()`
  - `OrcaGymModel.sensor_name2id()`
  - `OrcaGymModel.sensor_id2name()`

## `orca_gym/core/orca_gym_opt_config.py`

### Classes

- `OrcaGymOptConfig`

## `orca_gym/core/orca_gym_remote.py`

### Classes

- `OrcaGymRemote`
  - `OrcaGymRemote.init_simulation()`
  - `OrcaGymRemote.update_data()`
  - `OrcaGymRemote.query_all_actuators()`
  - `OrcaGymRemote.query_joint_qpos()`
  - `OrcaGymRemote.query_joint_qvel()`
  - `OrcaGymRemote.get_agent_state()`
  - `OrcaGymRemote.set_control_input()`
  - `OrcaGymRemote.load_initial_frame()`
  - `OrcaGymRemote.query_model_info()`
  - `OrcaGymRemote.query_opt_config()`
  - `OrcaGymRemote.set_opt_config()`
  - `OrcaGymRemote.mj_differentiate_pos()`
  - `OrcaGymRemote.mjd_transition_fd()`
  - `OrcaGymRemote.mj_jac_subtree_com()`
  - `OrcaGymRemote.mj_jac_body_com()`
  - `OrcaGymRemote.query_joint_names()`
  - `OrcaGymRemote.query_joint_dofadr()`
  - `OrcaGymRemote.query_all_qpos_qvel_qacc()`
  - `OrcaGymRemote.load_keyframe()`
  - `OrcaGymRemote.resume_simulation()`
  - `OrcaGymRemote.query_actuator_moment()`
  - `OrcaGymRemote.query_qfrc_inverse()`
  - `OrcaGymRemote.query_qfrc_actuator()`
  - `OrcaGymRemote.query_body_subtreemass_by_name()`
  - `OrcaGymRemote.set_qacc()`
  - `OrcaGymRemote.set_opt_timestep()`
  - `OrcaGymRemote.set_ctrl()`
  - `OrcaGymRemote.query_joint_type_by_id()`
  - `OrcaGymRemote.query_all_joints()`
  - `OrcaGymRemote.query_all_bodies()`
  - `OrcaGymRemote.query_cfrc_ext()`
  - `OrcaGymRemote.set_joint_qpos()`
  - `OrcaGymRemote.query_actuator_force()`
  - `OrcaGymRemote.query_joint_limits()`
  - `OrcaGymRemote.query_body_velocities()`
  - `OrcaGymRemote.query_actuator_gain_prm()`
  - `OrcaGymRemote.set_actuator_gain_prm()`
  - `OrcaGymRemote.query_actuator_bias_prm()`
  - `OrcaGymRemote.set_actuator_bias_prm()`
  - `OrcaGymRemote.query_all_mocap_bodies()`
  - `OrcaGymRemote.query_mocap_pos_and_quat()`
  - `OrcaGymRemote.set_mocap_pos_and_quat()`
  - `OrcaGymRemote.query_all_equality_constraints()`
  - `OrcaGymRemote.query_site_pos_and_mat()`
  - `OrcaGymRemote.mj_jac_site()`
  - `OrcaGymRemote.update_equality_constraints()`
  - `OrcaGymRemote.query_all_geoms()`
  - `OrcaGymRemote.query_contact()`
  - `OrcaGymRemote.query_contact_simple()`
  - `OrcaGymRemote.query_body_com_xpos_xmat()`
  - `OrcaGymRemote.query_body_xpos_xmat_xquat()`
  - `OrcaGymRemote.query_geom_xpos_xmat()`
  - `OrcaGymRemote.query_contact_force()`
  - `OrcaGymRemote.mj_jac()`
  - `OrcaGymRemote.calc_full_mass_matrix()`
  - `OrcaGymRemote.query_qfrc_bias()`
  - `OrcaGymRemote.query_subtree_com()`
  - `OrcaGymRemote.set_geom_friction()`
  - `OrcaGymRemote.query_sensor_data()`
  - `OrcaGymRemote.query_joint_offsets()`
  - `OrcaGymRemote.query_all_sites()`
  - `OrcaGymRemote.begin_save_video()`
  - `OrcaGymRemote.stop_save_video()`
  - `OrcaGymRemote.get_current_frame()`
