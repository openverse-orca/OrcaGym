from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Observation(_message.Message):
    __slots__ = ("tick", "imu", "joint_data", "contact_force")
    class ImuData(_message.Message):
        __slots__ = ("angle_roll", "angle_pitch", "angle_yaw", "angular_velocity_roll", "angular_velocity_pitch", "angular_velocity_yaw", "acc_x", "acc_y", "acc_z")
        ANGLE_ROLL_FIELD_NUMBER: _ClassVar[int]
        ANGLE_PITCH_FIELD_NUMBER: _ClassVar[int]
        ANGLE_YAW_FIELD_NUMBER: _ClassVar[int]
        ANGULAR_VELOCITY_ROLL_FIELD_NUMBER: _ClassVar[int]
        ANGULAR_VELOCITY_PITCH_FIELD_NUMBER: _ClassVar[int]
        ANGULAR_VELOCITY_YAW_FIELD_NUMBER: _ClassVar[int]
        ACC_X_FIELD_NUMBER: _ClassVar[int]
        ACC_Y_FIELD_NUMBER: _ClassVar[int]
        ACC_Z_FIELD_NUMBER: _ClassVar[int]
        angle_roll: float
        angle_pitch: float
        angle_yaw: float
        angular_velocity_roll: float
        angular_velocity_pitch: float
        angular_velocity_yaw: float
        acc_x: float
        acc_y: float
        acc_z: float
        def __init__(self, angle_roll: _Optional[float] = ..., angle_pitch: _Optional[float] = ..., angle_yaw: _Optional[float] = ..., angular_velocity_roll: _Optional[float] = ..., angular_velocity_pitch: _Optional[float] = ..., angular_velocity_yaw: _Optional[float] = ..., acc_x: _Optional[float] = ..., acc_y: _Optional[float] = ..., acc_z: _Optional[float] = ...) -> None: ...
    class JointData(_message.Message):
        __slots__ = ("position", "velocity", "torque", "temperature")
        POSITION_FIELD_NUMBER: _ClassVar[int]
        VELOCITY_FIELD_NUMBER: _ClassVar[int]
        TORQUE_FIELD_NUMBER: _ClassVar[int]
        TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
        position: float
        velocity: float
        torque: float
        temperature: float
        def __init__(self, position: _Optional[float] = ..., velocity: _Optional[float] = ..., torque: _Optional[float] = ..., temperature: _Optional[float] = ...) -> None: ...
    class LegData(_message.Message):
        __slots__ = ("fl_leg", "fr_leg", "hl_leg", "hr_leg")
        FL_LEG_FIELD_NUMBER: _ClassVar[int]
        FR_LEG_FIELD_NUMBER: _ClassVar[int]
        HL_LEG_FIELD_NUMBER: _ClassVar[int]
        HR_LEG_FIELD_NUMBER: _ClassVar[int]
        fl_leg: _containers.RepeatedCompositeFieldContainer[Observation.JointData]
        fr_leg: _containers.RepeatedCompositeFieldContainer[Observation.JointData]
        hl_leg: _containers.RepeatedCompositeFieldContainer[Observation.JointData]
        hr_leg: _containers.RepeatedCompositeFieldContainer[Observation.JointData]
        def __init__(self, fl_leg: _Optional[_Iterable[_Union[Observation.JointData, _Mapping]]] = ..., fr_leg: _Optional[_Iterable[_Union[Observation.JointData, _Mapping]]] = ..., hl_leg: _Optional[_Iterable[_Union[Observation.JointData, _Mapping]]] = ..., hr_leg: _Optional[_Iterable[_Union[Observation.JointData, _Mapping]]] = ...) -> None: ...
    class ContactForce(_message.Message):
        __slots__ = ("fl_leg", "fr_leg", "hl_leg", "hr_leg")
        FL_LEG_FIELD_NUMBER: _ClassVar[int]
        FR_LEG_FIELD_NUMBER: _ClassVar[int]
        HL_LEG_FIELD_NUMBER: _ClassVar[int]
        HR_LEG_FIELD_NUMBER: _ClassVar[int]
        fl_leg: _containers.RepeatedScalarFieldContainer[float]
        fr_leg: _containers.RepeatedScalarFieldContainer[float]
        hl_leg: _containers.RepeatedScalarFieldContainer[float]
        hr_leg: _containers.RepeatedScalarFieldContainer[float]
        def __init__(self, fl_leg: _Optional[_Iterable[float]] = ..., fr_leg: _Optional[_Iterable[float]] = ..., hl_leg: _Optional[_Iterable[float]] = ..., hr_leg: _Optional[_Iterable[float]] = ...) -> None: ...
    TICK_FIELD_NUMBER: _ClassVar[int]
    IMU_FIELD_NUMBER: _ClassVar[int]
    JOINT_DATA_FIELD_NUMBER: _ClassVar[int]
    CONTACT_FORCE_FIELD_NUMBER: _ClassVar[int]
    tick: int
    imu: Observation.ImuData
    joint_data: Observation.LegData
    contact_force: Observation.ContactForce
    def __init__(self, tick: _Optional[int] = ..., imu: _Optional[_Union[Observation.ImuData, _Mapping]] = ..., joint_data: _Optional[_Union[Observation.LegData, _Mapping]] = ..., contact_force: _Optional[_Union[Observation.ContactForce, _Mapping]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("robot_cmd",)
    class JointCmd(_message.Message):
        __slots__ = ("position", "velocity", "torque", "kp", "kd")
        POSITION_FIELD_NUMBER: _ClassVar[int]
        VELOCITY_FIELD_NUMBER: _ClassVar[int]
        TORQUE_FIELD_NUMBER: _ClassVar[int]
        KP_FIELD_NUMBER: _ClassVar[int]
        KD_FIELD_NUMBER: _ClassVar[int]
        position: float
        velocity: float
        torque: float
        kp: float
        kd: float
        def __init__(self, position: _Optional[float] = ..., velocity: _Optional[float] = ..., torque: _Optional[float] = ..., kp: _Optional[float] = ..., kd: _Optional[float] = ...) -> None: ...
    class RobotCmd(_message.Message):
        __slots__ = ("fl_leg", "fr_leg", "hl_leg", "hr_leg")
        FL_LEG_FIELD_NUMBER: _ClassVar[int]
        FR_LEG_FIELD_NUMBER: _ClassVar[int]
        HL_LEG_FIELD_NUMBER: _ClassVar[int]
        HR_LEG_FIELD_NUMBER: _ClassVar[int]
        fl_leg: _containers.RepeatedCompositeFieldContainer[Action.JointCmd]
        fr_leg: _containers.RepeatedCompositeFieldContainer[Action.JointCmd]
        hl_leg: _containers.RepeatedCompositeFieldContainer[Action.JointCmd]
        hr_leg: _containers.RepeatedCompositeFieldContainer[Action.JointCmd]
        def __init__(self, fl_leg: _Optional[_Iterable[_Union[Action.JointCmd, _Mapping]]] = ..., fr_leg: _Optional[_Iterable[_Union[Action.JointCmd, _Mapping]]] = ..., hl_leg: _Optional[_Iterable[_Union[Action.JointCmd, _Mapping]]] = ..., hr_leg: _Optional[_Iterable[_Union[Action.JointCmd, _Mapping]]] = ...) -> None: ...
    ROBOT_CMD_FIELD_NUMBER: _ClassVar[int]
    robot_cmd: Action.RobotCmd
    def __init__(self, robot_cmd: _Optional[_Union[Action.RobotCmd, _Mapping]] = ...) -> None: ...
