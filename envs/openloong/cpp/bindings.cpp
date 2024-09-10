#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For binding STL containers like std::vector
#include "openloong_wbc.h"
#include "orcagym_interface.h"

namespace py = pybind11;

// Binding the ButtonState struct
void bindButtonState(py::module &m) {
    py::class_<ButtonState>(m, "ButtonState")
        .def(py::init<>())  // Default constructor
        .def_readwrite("key_space", &ButtonState::key_space)
        .def_readwrite("key_a", &ButtonState::key_a)
        .def_readwrite("key_d", &ButtonState::key_d)
        .def_readwrite("key_w", &ButtonState::key_w)
        .def_readwrite("key_s", &ButtonState::key_s)
        .def_readwrite("key_h", &ButtonState::key_h);
}

// Binding the OrcaGym_Interface class
void bindOrcaGymInterface(py::module &m) {
    py::class_<OrcaGym_Interface>(m, "OrcaGym_Interface")
        .def(py::init<double>())  // Constructor
        .def("getJointName", &OrcaGym_Interface::getJointName)
        .def("getBaseName", &OrcaGym_Interface::getBaseName)
        .def("getOrientationSensorName", &OrcaGym_Interface::getOrientationSensorName)
        .def("getVelSensorName", &OrcaGym_Interface::getVelSensorName)
        .def("getGyroSensorName", &OrcaGym_Interface::getGyroSensorName)
        .def("getAccSensorName", &OrcaGym_Interface::getAccSensorName)
        .def("setMotorPos", &OrcaGym_Interface::setMotorPos)
        .def("setMotorVel", &OrcaGym_Interface::setMotorVel)
        .def("setRPY", &OrcaGym_Interface::setRPY)
        .def("setBaseQuat", &OrcaGym_Interface::setBaseQuat)
        .def("setF3d", &OrcaGym_Interface::setF3d)
        .def("setBasePos", &OrcaGym_Interface::setBasePos)
        .def("setBaseAcc", &OrcaGym_Interface::setBaseAcc)
        .def("setBaseAngVel", &OrcaGym_Interface::setBaseAngVel)
        .def("setBaseLinVel", &OrcaGym_Interface::setBaseLinVel)
        .def("setMotorCtrl", &OrcaGym_Interface::setMotorCtrl)
        .def("getMotorCtrl", &OrcaGym_Interface::getMotorCtrl)
        .def("setJointOffsetQpos", &OrcaGym_Interface::setJointOffsetQpos)
        .def("setJointOffsetQvel", &OrcaGym_Interface::setJointOffsetQvel)
        .def("updateSensorValues", &OrcaGym_Interface::updateSensorValues);
}

// Binding the OpenLoongWBC class
void bindOpenLoongEnv(py::module &m) {
    py::class_<OpenLoongWBC>(m, "OpenLoongWBC")
        .def(py::init<const std::string&, double, const std::string&, const std::string&, int, int>())  // Constructor
        .def("InitLogger", &OpenLoongWBC::InitLogger)
        .def("Runsimulation", &OpenLoongWBC::Runsimulation)
        .def("SetBaseUp", &OpenLoongWBC::SetBaseUp)
        .def("SetBaseDown", &OpenLoongWBC::SetBaseDown);
}

PYBIND11_MODULE(openloong_dyn_ctrl, m) {
    bindButtonState(m);
    bindOrcaGymInterface(m);
    bindOpenLoongEnv(m);
}
