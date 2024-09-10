#include "openloong_wbc.h"
#include <map>

OpenLoongWBC::OpenLoongWBC(const std::string &urdf_path, double timestep, const std::string &json_path, const std::string &log_path, int model_nq, int model_nv) : kinDynSolver(urdf_path),
                                                                                                                                                                   gaitScheduler(0.4, timestep),
                                                                                                                                                                   pvtCtr(timestep, json_path.c_str()),
                                                                                                                                                                   jsInterp(timestep),
                                                                                                                                                                   logger(log_path),
                                                                                                                                                                   RobotState(model_nv),
                                                                                                                                                                   WBC_solv(model_nv, 18, 22, 0.7, timestep)
{
    printf("kinDynSolver.model_nv=%d model_nq=%d model_nv=%d\n", kinDynSolver.model_nv, model_nq, model_nv);
    
    this->model_nv = model_nv;
    this->timestep = timestep;

    RobotState.width_hips = 0.229;
    footPlacement.kp_vx = 0.03;
    footPlacement.kp_vy = 0.03;
    footPlacement.kp_wz = 0.03;
    footPlacement.stepHeight = 0.20;
    footPlacement.legLength = stand_legLength;

    motors_pos_des.resize(model_nv - 6, 0);
    motors_pos_cur.resize(model_nv - 6, 0);
    motors_vel_des.resize(model_nv - 6, 0);
    motors_vel_cur.resize(model_nv - 6, 0);
    motors_tau_des.resize(model_nv - 6, 0);
    motors_tau_cur.resize(model_nv - 6, 0);


    // ini position and posture for foot-end and hand
    Eigen::Vector3d fe_l_pos_L_des = {-0.018, 0.113, 0 - stand_legLength};
    Eigen::Vector3d fe_r_pos_L_des = {-0.018, -0.116, 0 - stand_legLength};
    Eigen::Vector3d fe_l_eul_L_des = {-0.000, -0.008, -0.000};
    Eigen::Vector3d fe_r_eul_L_des = {0.000, -0.008, 0.000};
    Eigen::Matrix3d fe_l_rot_des = eul2Rot(fe_l_eul_L_des(0), fe_l_eul_L_des(1), fe_l_eul_L_des(2));
    Eigen::Matrix3d fe_r_rot_des = eul2Rot(fe_r_eul_L_des(0), fe_r_eul_L_des(1), fe_r_eul_L_des(2));

    Eigen::Vector3d hd_l_pos_L_des = {-0.02, 0.32, -0.159};
    Eigen::Vector3d hd_r_pos_L_des = {-0.02, -0.32, -0.159};
    Eigen::Vector3d hd_l_eul_L_des = {-1.7581, 0.2129, 2.9581};
    Eigen::Vector3d hd_r_eul_L_des = {1.7581, 0.2129, -2.9581};
    Eigen::Matrix3d hd_l_rot_des = eul2Rot(hd_l_eul_L_des(0), hd_l_eul_L_des(1), hd_l_eul_L_des(2));
    Eigen::Matrix3d hd_r_rot_des = eul2Rot(hd_r_eul_L_des(0), hd_r_eul_L_des(1), hd_r_eul_L_des(2));

    resLeg = kinDynSolver.computeInK_Leg(fe_l_rot_des, fe_l_pos_L_des, fe_r_rot_des, fe_r_pos_L_des);
    resHand = kinDynSolver.computeInK_Hand(hd_l_rot_des, hd_l_pos_L_des, hd_r_rot_des, hd_r_pos_L_des);

    // std::cout << "resLeg.jointPosRes.size(): " << resLeg.jointPosRes.size() << std::endl;
    // std::cout << "resHand.jointPosRes.size(): " << resHand.jointPosRes.size() << std::endl;
    // std::cout << "model_nq - 7: " << (model_nq - 7) << std::endl;

    qIniDes = Eigen::VectorXd::Zero(model_nq, 1);
    qIniDes.block(7, 0, model_nq - 7, 1) = resLeg.jointPosRes + resHand.jointPosRes;
}

void OpenLoongWBC::InitLogger()
{
    // register variable name for data logger
    logger.addIterm("simTime", 1);
    logger.addIterm("motors_pos_cur", model_nv - 6);
    logger.addIterm("motors_vel_cur", model_nv - 6);
    logger.addIterm("rpy", 3);
    logger.addIterm("fL", 3);
    logger.addIterm("fR", 3);
    logger.addIterm("basePos", 3);
    logger.addIterm("baseLinVel", 3);
    logger.addIterm("baseAcc", 3);
    logger.finishItermAdding();
}

/// ----------------- sim Loop ---------------
const double openLoopCtrTime = 3;

void OpenLoongWBC::Runsimulation(const ButtonState &buttonState, OrcaGym_Interface &orcagym_interface, double simTime)
{
    // orcagym_interface.updateSensorValues();     // 在进入runsim之前，由gym完成数据更新
    orcagym_interface.dataBusWrite(RobotState);

    // input from joystick
    // space: start and stop stepping (after 3s)
    // w: forward walking
    // s: stop forward walking
    // a: turning left
    // d: turning right
    if (simTime > openLoopCtrTime)
    {
        if (buttonState.key_space && RobotState.motionState == DataBus::Stand)
        {
            jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
            RobotState.motionState = DataBus::Walk;
        }
        else if (buttonState.key_space && RobotState.motionState == DataBus::Walk && fabs(jsInterp.vxLGen.y) < 0.01)
        {
            RobotState.motionState = DataBus::Walk2Stand;
            jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
        }

        if (buttonState.key_a && RobotState.motionState != DataBus::Stand)
        {
            if (jsInterp.wzLGen.yDes < 0)
                jsInterp.setWzDesLPara(0, 0.5);
            else
                jsInterp.setWzDesLPara(0.35, 1.0);
        }
        if (buttonState.key_d && RobotState.motionState != DataBus::Stand)
        {
            if (jsInterp.wzLGen.yDes > 0)
                jsInterp.setWzDesLPara(0, 0.5);
            else
                jsInterp.setWzDesLPara(-0.35, 1.0);
        }

        if (buttonState.key_w && RobotState.motionState != DataBus::Stand)
            jsInterp.setVxDesLPara(xv_des, 2.0);

        if (buttonState.key_s && RobotState.motionState != DataBus::Stand)
            jsInterp.setVxDesLPara(0, 0.5);

        if (buttonState.key_h)
            jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));
    }

    // update kinematics and dynamics info
    kinDynSolver.dataBusRead(RobotState);
    kinDynSolver.computeJ_dJ();
    kinDynSolver.computeDyn();
    kinDynSolver.dataBusWrite(RobotState);

    if (simTime >= openLoopCtrTime && simTime < openLoopCtrTime + 0.002)
    {
        RobotState.motionState = DataBus::Stand;
    }

    if (RobotState.motionState == DataBus::Walk2Stand || simTime <= openLoopCtrTime)
        jsInterp.setIniPos(RobotState.q(0), RobotState.q(1), RobotState.base_rpy(2));

    // switch between walk and stand
    if (RobotState.motionState == DataBus::Walk || RobotState.motionState == DataBus::Walk2Stand)
    {
        jsInterp.step();
        RobotState.js_pos_des(2) = stand_legLength + foot_height; // pos z is not assigned in jyInterp
        jsInterp.dataBusWrite(RobotState);                        // only pos x, pos y, theta z, vel x, vel y , omega z are rewrote.

        //                if (simTime <startSteppingTime+0.002)
        //                    RobotState.motionState=DataBus::Walk;
        // gait scheduler
        gaitScheduler.dataBusRead(RobotState);
        gaitScheduler.step();
        gaitScheduler.dataBusWrite(RobotState);

        footPlacement.dataBusRead(RobotState);
        footPlacement.getSwingPos();
        footPlacement.dataBusWrite(RobotState);
    }

    if (simTime <= openLoopCtrTime || RobotState.motionState == DataBus::Walk2Stand)
    {
        WBC_solv.setQini(qIniDes, RobotState.q);
        WBC_solv.fe_l_pos_des_W = RobotState.fe_l_pos_W;
        WBC_solv.fe_r_pos_des_W = RobotState.fe_r_pos_W;
        WBC_solv.fe_l_rot_des_W = RobotState.fe_l_rot_W;
        WBC_solv.fe_r_rot_des_W = RobotState.fe_r_rot_W;
        WBC_solv.pCoMDes = RobotState.pCoM_W;
        WBC_solv.pCoMDes(0) = (RobotState.fe_l_pos_W(0) + RobotState.fe_r_pos_W(0)) * 0.5;
        WBC_solv.pCoMDes(1) = (RobotState.fe_l_pos_W(1) + RobotState.fe_r_pos_W(1)) * 0.5;
    }

    if (RobotState.motionState == DataBus::Stand)
    {
        WBC_solv.pCoMDes(0) = (RobotState.fe_l_pos_W(0) + RobotState.fe_r_pos_W(0)) * 0.5;
        WBC_solv.pCoMDes(1) = (RobotState.fe_l_pos_W(1) + RobotState.fe_r_pos_W(1)) * 0.5;
    }

    //            std::cout<<"pCoM_W"<<std::endl<<RobotState.pCoM_W.transpose()<<std::endl<<"pCoM_Des"<<std::endl<<WBC_solv.pCoMDes.transpose()<<std::endl;

    // ------------- WBC ------------
    // WBC input
    RobotState.Fr_ff = Eigen::VectorXd::Zero(12);
    RobotState.des_ddq = Eigen::VectorXd::Zero(model_nv);
    RobotState.des_dq = Eigen::VectorXd::Zero(model_nv);
    RobotState.des_delta_q = Eigen::VectorXd::Zero(model_nv);
    RobotState.base_rpy_des << 0, 0, jsInterp.thetaZ;
    RobotState.base_pos_des = RobotState.js_pos_des;
    RobotState.base_pos_des(2) = stand_legLength + foot_height;

    RobotState.Fr_ff << 0, 0, 370, 0, 0, 0,
        0, 0, 370, 0, 0, 0;

    // adjust des_delata_q, des_dq and des_ddq to achieve forward walking
    if (RobotState.motionState == DataBus::Walk)
    {
        RobotState.des_delta_q.block<2, 1>(0, 0) << jsInterp.vx_W * timestep, jsInterp.vy_W * timestep;
        RobotState.des_delta_q(5) = jsInterp.wz_L * timestep;
        RobotState.des_dq.block<2, 1>(0, 0) << jsInterp.vx_W, jsInterp.vy_W;
        RobotState.des_dq(5) = jsInterp.wz_L;

        double k = 5; // 5
        RobotState.des_ddq.block<2, 1>(0, 0) << k * (jsInterp.vx_W - RobotState.dq(0)), k * (jsInterp.vy_W -
                                                                                             RobotState.dq(1));
        RobotState.des_ddq(5) = k * (jsInterp.wz_L - RobotState.dq(5));
    }
    printf("js_vx=%.3f js_vy=%.3f wz_L=%.3f px_w=%.3f py_w=%.3f thetaZ=%.3f\n", jsInterp.vx_W, jsInterp.vy_W, jsInterp.wz_L, jsInterp.px_W, jsInterp.py_W, jsInterp.thetaZ);

    // WBC Calculation
    WBC_solv.dataBusRead(RobotState);
    WBC_solv.computeDdq(kinDynSolver);
    WBC_solv.computeTau();
    WBC_solv.dataBusWrite(RobotState);

    // get the final joint command
    if (simTime <= openLoopCtrTime)
    {
        RobotState.motors_pos_des = eigen2std(resLeg.jointPosRes + resHand.jointPosRes);
        RobotState.motors_vel_des = motors_vel_des;
        RobotState.motors_tor_des = motors_tau_des;
    }
    else
    {
        Eigen::VectorXd pos_des = kinDynSolver.integrateDIY(RobotState.q, RobotState.wbc_delta_q_final);
        RobotState.motors_pos_des = eigen2std(pos_des.block(7, 0, model_nv - 6, 1));
        RobotState.motors_vel_des = eigen2std(RobotState.wbc_dq_final);
        RobotState.motors_tor_des = eigen2std(RobotState.wbc_tauJointRes);
    }

    pvtCtr.dataBusRead(RobotState);
    if (simTime <= openLoopCtrTime)
    {
        pvtCtr.calMotorsPVT(100.0 / 1000.0 / 180.0 * 3.1415);
    }
    else
    {
        if (RobotState.motionState == DataBus::Walk2Stand || RobotState.motionState == DataBus::Walk)
        {
            pvtCtr.setJointPD(100, 10, "J_ankle_l_pitch");
            pvtCtr.setJointPD(100, 10, "J_ankle_l_roll");
            pvtCtr.setJointPD(100, 10, "J_ankle_r_pitch");
            pvtCtr.setJointPD(100, 10, "J_ankle_r_roll");
            pvtCtr.setJointPD(1000, 100, "J_knee_l_pitch");
            pvtCtr.setJointPD(1000, 100, "J_knee_r_pitch");
        }
        else
        {
            pvtCtr.setJointPD(1000, 160, "J_ankle_l_pitch");
            pvtCtr.setJointPD(1000, 160, "J_ankle_l_roll");
            pvtCtr.setJointPD(1000, 160, "J_ankle_r_pitch");
            pvtCtr.setJointPD(1000, 160, "J_ankle_r_roll");
            pvtCtr.setJointPD(2000, 200, "J_knee_l_pitch");
            pvtCtr.setJointPD(2000, 200, "J_knee_r_pitch");
            pvtCtr.setJointPD(2000, 80, "J_waist_pitch");
        }
        pvtCtr.calMotorsPVT();
    }

    pvtCtr.dataBusWrite(RobotState);
    // orcagym_interface.setMotorsTorque(RobotState.motors_tor_out);
    // 这里调用setMotorCtrl，然后在Gym调用step之前通过getMotorCtrl获得ctrl值
    orcagym_interface.setMotorCtrl(RobotState.motors_tor_out);

    // data record

    logger.startNewLine();
    logger.recItermData("simTime", simTime);
    logger.recItermData("motors_pos_cur", RobotState.motors_pos_cur);
    logger.recItermData("motors_vel_cur", RobotState.motors_vel_cur);
    logger.recItermData("rpy", RobotState.rpy);
    logger.recItermData("fL", RobotState.fL);
    logger.recItermData("fR", RobotState.fR);
    logger.recItermData("basePos", RobotState.basePos);
    logger.recItermData("baseLinVel", RobotState.baseLinVel);
    logger.recItermData("baseAcc", RobotState.baseAcc);
    logger.finishLine();

    printf("rpyVal=[%.5f, %.5f, %.5f]\n", RobotState.rpy[0], RobotState.rpy[1], RobotState.rpy[2]);
    printf("gps=[%.5f, %.5f, %.5f]\n", RobotState.basePos[0], RobotState.basePos[1], RobotState.basePos[2]);
    printf("vel=[%.5f, %.5f, %.5f]\n", RobotState.baseLinVel[0], RobotState.baseLinVel[1], RobotState.baseLinVel[2]);
}