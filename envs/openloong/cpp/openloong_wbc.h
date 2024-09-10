#include <cstdio>
#include <iostream>
#include <memory>
#include "useful_math.h"
#include "GLFW_callbacks.h"
#include "MJ_interface.h"
#include "PVT_ctrl.h"
#include "pino_kin_dyn.h"
#include "data_logger.h"
#include "wbc_priority.h"
#include "gait_scheduler.h"
#include "foot_placement.h"
#include "joystick_interpreter.h"
#include "orcagym_interface.h"

struct ButtonState
{
    bool key_space;
    bool key_a;
    bool key_d;
    bool key_w;
    bool key_s;
    bool key_h;
};

class OpenLoongWBC
{

public:
    OpenLoongWBC(const std::string &urdf_path, double timestep, const std::string &json_path, const std::string &log_path, int model_nq, int model_nv);

    void InitLogger();

    void Runsimulation(const ButtonState &buttonState, OrcaGym_Interface &orcagym_interface, double simTime);

private:
    Pin_KinDyn kinDynSolver;
    DataBus RobotState;
    WBC_priority WBC_solv;
    GaitScheduler gaitScheduler;
    PVT_Ctr pvtCtr;
    FootPlacement footPlacement;
    JoyStickInterpreter jsInterp;
    DataLogger logger;

    // variables ini
    const double stand_legLength = 1.01; // desired baselink height
    const double foot_height = 0.07;     // distance between the foot ankel joint and the bottom
    const double xv_des = 0.7;           // desired velocity in x direction

    Eigen::VectorXd qIniDes;
    Pin_KinDyn::IkRes resLeg;
    Pin_KinDyn::IkRes resHand;

    int model_nv;
    double timestep;

    std::vector<double> motors_pos_des;
    std::vector<double> motors_pos_cur;
    std::vector<double> motors_vel_des;
    std::vector<double> motors_vel_cur;
    std::vector<double> motors_tau_des;
    std::vector<double> motors_tau_cur;    
};
