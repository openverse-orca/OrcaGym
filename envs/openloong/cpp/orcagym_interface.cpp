#include "orcagym_interface.h"

OrcaGym_Interface::OrcaGym_Interface(double timestep)
{
    this->timestep = timestep;
    jointNum = JointName.size();
    motor_pos.assign(jointNum, 0);
    motor_vel.assign(jointNum, 0);
    motor_ctrl.assign(jointNum, 0);
}

void OrcaGym_Interface::updateSensorValues(std::vector<double> &qpos,
                                      std::vector<double> &qvel,
                                      std::vector<double> &sensordata_quat,     // baselink-quat
                                      std::vector<double> &sensordata_vel,
                                      std::vector<double> &sensordata_gyro,     // gyro
                                      std::vector<double> &sensordata_acc,      // baseAcc
                                      std::vector<double> &xpos)
{
    std::cout << "jointOffsetQpos, Qvel";
    for (int i = 0; i < jointNum; i++)
    {
        std::cout << "[" << jointOffsetQpos[i] << ", " << jointOffsetQvel[i] << "] ";
        motor_pos[i] = qpos[jointOffsetQpos[i]];
        motor_vel[i] = qvel[jointOffsetQvel[i]];
    }
    std::cout << std::endl;
    for (int i = 0; i < 4; i++)
        baseQuat[i] = sensordata_quat[i];
    double tmp = baseQuat[0];
    baseQuat[0] = baseQuat[1];
    baseQuat[1] = baseQuat[2];
    baseQuat[2] = baseQuat[3];
    baseQuat[3] = tmp;

    rpy[0] = atan2(2 * (baseQuat[3] * baseQuat[0] + baseQuat[1] * baseQuat[2]), 1 - 2 * (baseQuat[0] * baseQuat[0] + baseQuat[1] * baseQuat[1]));
    rpy[1] = asin(2 * (baseQuat[3] * baseQuat[1] - baseQuat[0] * baseQuat[2]));
    rpy[2] = atan2(2 * (baseQuat[3] * baseQuat[2] + baseQuat[0] * baseQuat[1]), 1 - 2 * (baseQuat[1] * baseQuat[1] + baseQuat[2] * baseQuat[2]));

    for (int i = 0; i < 3; i++)
    {
        double posOld = basePos[i];
        basePos[i] = xpos[i];
        baseAcc[i] = sensordata_acc[i];
        baseAngVel[i] = sensordata_gyro[i];
        baseLinVel[i] = (basePos[i] - posOld) / (timestep);
    }
}


void OrcaGym_Interface::dataBusWrite(DataBus &busIn)
{
    busIn.motors_pos_cur = motor_pos;
    busIn.motors_vel_cur = motor_vel;
    busIn.rpy[0] = rpy[0];
    busIn.rpy[1] = rpy[1];
    busIn.rpy[2] = rpy[2];
    busIn.fL[0] = f3d[0][0];
    busIn.fL[1] = f3d[1][0];
    busIn.fL[2] = f3d[2][0];
    busIn.fR[0] = f3d[0][1];
    busIn.fR[1] = f3d[1][1];
    busIn.fR[2] = f3d[2][1];
    busIn.basePos[0] = basePos[0];
    busIn.basePos[1] = basePos[1];
    busIn.basePos[2] = basePos[2];
    busIn.baseLinVel[0] = baseLinVel[0];
    busIn.baseLinVel[1] = baseLinVel[1];
    busIn.baseLinVel[2] = baseLinVel[2];
    busIn.baseAcc[0] = baseAcc[0];
    busIn.baseAcc[1] = baseAcc[1];
    busIn.baseAcc[2] = baseAcc[2];
    busIn.baseAngVel[0] = baseAngVel[0];
    busIn.baseAngVel[1] = baseAngVel[1];
    busIn.baseAngVel[2] = baseAngVel[2];
    busIn.updateQ();
}
