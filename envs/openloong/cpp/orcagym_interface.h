
#pragma once

#include "data_bus.h"
#include <string>
#include <vector>
#include <array>
#include <map>

class OrcaGym_Interface
{
private:
    std::vector<double> motor_pos;
    std::vector<double> motor_vel;
    std::vector<double> motor_ctrl;

    double rpy[3]{0}; // roll,pitch and yaw of baselink
    double baseQuat[4]{0}; // in quat, mujoco order is [w,x,y,z], here we rearrange to [x,y,z,w]
    double f3d[3][2]{0}; // 3D foot-end contact force, L for 1st col, R for 2nd col
    double basePos[3]{0}; // position of baselink, in world frame
    double baseAcc[3]{0};  // acceleration of baselink, in body frame
    double baseAngVel[3]{0}; // angular velocity of baselink, in body frame
    double baseLinVel[3]{0}; // linear velocity of baselink, in body frame

    

    const std::vector<std::string> JointName={ "J_arm_l_01","J_arm_l_02","J_arm_l_03", "J_arm_l_04", "J_arm_l_05",
                                               "J_arm_l_06","J_arm_l_07","J_arm_r_01", "J_arm_r_02", "J_arm_r_03",
                                               "J_arm_r_04","J_arm_r_05","J_arm_r_06", "J_arm_r_07",
                                               "J_head_yaw","J_head_pitch","J_waist_pitch","J_waist_roll", "J_waist_yaw",
                                               "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch", "J_knee_l_pitch",
                                               "J_ankle_l_pitch", "J_ankle_l_roll", "J_hip_r_roll", "J_hip_r_yaw",
                                               "J_hip_r_pitch", "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll"}; // joint name in XML file, the corresponds motors name should be M_*, ref to line 29 of MJ_Interface.cpp
    const std::string baseName="base_link";
    const std::string orientationSensorName="baselink-quat"; // in quat, mujoco order is [w,x,y,z], here we rearrange to [x,y,z,w]
    const std::string velSensorName="baselink-velocity";
    const std::string gyroSensorName="baselink-gyro";
    const std::string accSensorName="baselink-baseAcc";

public:
    // Getter functions
    const std::vector<std::string> &getJointName() const
    {
        return JointName;
    }

    const std::string &getBaseName() const
    {
        return baseName;
    }

    const std::string &getOrientationSensorName() const
    {
        return orientationSensorName;
    }

    const std::string &getVelSensorName() const
    {
        return velSensorName;
    }

    const std::string &getGyroSensorName() const
    {
        return gyroSensorName;
    }

    const std::string &getAccSensorName() const
    {
        return accSensorName;
    }

    // Setters for vectors
    void setMotorPos(const std::vector<double>& pos) {
        motor_pos = pos;
    }

    void setMotorVel(const std::vector<double>& vel) {
        motor_vel = vel;
    }

    void setRPY(const std::array<double, 3>& new_rpy) {
        for (int i = 0; i < 3; ++i) {
            rpy[i] = new_rpy[i];
        }
    }

    void setBaseQuat(const std::array<double, 4>& new_baseQuat) {
        for (int i = 0; i < 4; ++i) {
            baseQuat[i] = new_baseQuat[i];
        }
    }

    void setF3d(const std::array<std::array<double, 2>, 3>& new_f3d) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                f3d[i][j] = new_f3d[i][j];
            }
        }
    }

    void setBasePos(const std::array<double, 3>& new_basePos) {
        for (int i = 0; i < 3; ++i) {
            basePos[i] = new_basePos[i];
        }
    }

    void setBaseAcc(const std::array<double, 3>& new_baseAcc) {
        for (int i = 0; i < 3; ++i) {
            baseAcc[i] = new_baseAcc[i];
        }
    }

    void setBaseAngVel(const std::array<double, 3>& new_baseAngVel) {
        for (int i = 0; i < 3; ++i) {
            baseAngVel[i] = new_baseAngVel[i];
        }
    }

    void setBaseLinVel(const std::array<double, 3>& new_baseLinVel) {
        for (int i = 0; i < 3; ++i) {
            baseLinVel[i] = new_baseLinVel[i];
        }
    }

    void setMotorCtrl(const std::vector<double>& ctrl) {
        if (ctrl.size() == motor_ctrl.size()) {
            motor_ctrl = ctrl;
        }
    }

    const std::vector<double>& getMotorCtrl() const {
        return motor_ctrl;
    }

    OrcaGym_Interface();
    // void updateSensorValues();
    // void setMotorsTorque(std::vector<double> &tauIn);
    void dataBusWrite(DataBus &busIn);

private:
    // OrcaGymModel *mj_model;
    
    // std::vector<int> jntId_qpos, jntId_qvel, jntId_dctl;

    // int orientataionSensorId;
    // int velSensorId;
    // int gyroSensorId;
    // int accSensorId;
    // int baseBodyId;

    // double timeStep{0.001}; // second
    // bool isIni{false};
};
