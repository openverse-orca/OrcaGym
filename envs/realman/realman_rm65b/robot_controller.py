import datetime
from .robotic_arm_package.robotic_arm import *
from .roh_registers import *
import numpy as np
import threading

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


COM_PORT = 1
ROH_ADDR = 2


def L_BYTE(v):
    return v & 0xFF


def H_BYTE(v):
    return (v >> 8) & 0xFF




MAX_SINGLE_FINGER_POS = [H_BYTE(65535), L_BYTE(65535)]

MIN_SINGLE_FINGER_POS = [H_BYTE(0), L_BYTE(0)]

HAND_FIST_CLOSE_POS = [
    H_BYTE(0), L_BYTE(0),
    H_BYTE(65535), L_BYTE(65535),
    H_BYTE(65535), L_BYTE(65535),
    H_BYTE(65535), L_BYTE(65535),
    H_BYTE(65535), L_BYTE(65535),
    H_BYTE(65535), L_BYTE(65535),
]

HAND_FIST_OPEN_POS = [
    H_BYTE(0), L_BYTE(0),
    H_BYTE(0), L_BYTE(0),
    H_BYTE(0), L_BYTE(0),
    H_BYTE(0), L_BYTE(0),
    H_BYTE(0), L_BYTE(0),
    H_BYTE(65535), L_BYTE(65535),
]

HAND_VICTORY_POS = [
    H_BYTE(65535), L_BYTE(65535),
    H_BYTE(0), L_BYTE(0),
    H_BYTE(0), L_BYTE(0),
    H_BYTE(65535), L_BYTE(65535),
    H_BYTE(65535), L_BYTE(65535),
    H_BYTE(65535), L_BYTE(65535),
]

target_fingers = [
    ROH_FINGER_POS_TARGET0, #拇指 弯曲
    ROH_FINGER_POS_TARGET1, #食指
    ROH_FINGER_POS_TARGET2, #中指
    ROH_FINGER_POS_TARGET3, #无名指
    ROH_FINGER_POS_TARGET4, #小指
    ROH_FINGER_POS_TARGET5, #拇指 转动
]


class DeviceType:
    RM65 = 1
    RM75 = 2

class RobotController:
    def __init__(self,ip,devtype) -> None:
        self.enabled = True
        if not self.enabled:
            return
        
        # self.rm65_ip= "192.168.33.80"
       # self.rm65_ip= "192.168.1.18"
        self.ip= ip
       
       # self.robot = Arm(RM65, self.rm65_ip)
        self.robot = Arm(RM75, self.ip)
        self.ctrlnum = devtype==DeviceType.RM65 and 6 or 7
       
       
       
        # callback = RealtimePush_Callback(RobotController.MCallback)
        # self.robot.Realtime_Arm_Joint_State(callback)

        self.mutex= threading.Lock()
        self.ctrl = None

        self.gripper_open = True
        self.gripper_changed = False

        self.running = True
        _logger.info("RobotController init 。。。。")
        self.thread =threading.Thread(target=self.loop)
        self.thread.start()

    def MCallback(data):
        _logger.info(f"推送数据的机械臂IP:  {data.arm_ip}")
        _logger.info(f"机械臂错误码:  {data.arm_err}")
        _logger.info(f"系统错误码:  {data.sys_err}")
        _logger.info(f"机械臂当前角度: {list(data.joint_status.joint_position)}")
        print("机械臂当前位姿： ", data.waypoint.position.x, data.waypoint.position.y, data.waypoint.position.z,
            data.waypoint.euler.rx, data.waypoint.euler.ry, data.waypoint.euler.rz)

    
    def __del__(self):
        if not self.enabled:
            return
    
        self.running = False
        self.thread.join()
        self.robot.RM_API_UnInit()
        self.robot.Arm_Socket_Close()

    def init_joint_state(self, ctrl: np.array):
        if not self.enabled:
            return
        
        #self.robot.Movej_Cmd((ctrl * 57.29577951308232).tolist()[0:6], 30, 0, 0, True)
        self.robot.Movej_Cmd((ctrl * 57.29577951308232).tolist()[0:self.ctrlnum], 30, 0, 0, True)
        _logger.info(f"init_joint_state:  {ctrl}")

    def sync_joint(self, ctrl: np.array):
        if not self.enabled:
            return
        
        self.mutex.acquire()
        self.ctrl = ctrl
        _logger.info(f"sync_joint.............................:  {ctrl}")
        self.mutex.release()

    def loop(self):
        while self.running:
            do_nothing = True
           # print("loop1111111.............................: ")
            if self.ctrl is not None:
                #print("loop2222.............................: ")
                self.mutex.acquire()
                ctrl = self.ctrl.copy()
                self.ctrl = None
                self.mutex.release()
                do_nothing = False
               # print("loop333333.............................: ", ctrl)
                mm = ctrl * 57.29577951308232
                #self.robot.Movej_CANFD((ctrl * 57.29577951308232).tolist()[0:6], True)
                self.robot.Movej_CANFD((ctrl * 57.29577951308232).tolist()[0:self.ctrlnum], True)
                _logger.info(f"loop.............................:  {mm}")
                # self.robot.Movej_Cmd((ctrl * 57.29577951308232).tolist()[0:6], 30, 0, 0, True)

            if self.gripper_changed:
                self.mutex.acquire()
                self.gripper_changed = False
                gripper_open = self.gripper_open
                self.mutex.release()
                do_nothing = False
                if gripper_open:
                    self.robot.Set_Gripper_Pick_On(speed=500, force=500)
                    _logger.info("gripper pick")
                else:
                    self.robot.Set_Gripper_Release(speed=500)
                    _logger.info("gripper release")

            if do_nothing:
                time.sleep(0.002)

    def gripper_pick_on(self):
        if not self.enabled:
            return
        
        self.mutex.acquire()
        if not self.gripper_open:
            self.gripper_changed = True
            self.gripper_open = True
        self.mutex.release()

    def gripper_release(self):
        if not self.enabled:
            return
        
        self.mutex.acquire()
        if self.gripper_open:
            self.gripper_changed = True
            self.gripper_open = False
        self.mutex.release()
    
    
    def finger_move(self,target, num_finger, HAND_POS, delay):
        self.robot.Write_Registers(
            COM_PORT,
            target,
            num_finger,
            HAND_POS,
            ROH_ADDR,
            True
        )
        time.sleep(delay)
    
    def roh_test(self):
        _logger.info("roh_test..............................")
        self.robot.Close_Modbustcp_Mode()

        self.robot.Set_Modbus_Mode(1, 115200, 1, True)
        tt = [0, 0, 100, 62, 150, 62, 220, 180, 50, 79, 127, 0]
       # tt = [0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0]
       # self.finger_move(target_fingers[0], 6, tt, 0.005)
        tt = [0, 0, 100, 62, 150, 62, 220, 180, 0, 79, 127, 0]
        self.finger_move(target_fingers[0], 6, tt, 2)
        for index in range(20):
            tt = [0, 0, 100, 62, 150, 62, 220, 180, index*12, 79, 127, 0]
            self.finger_move(target_fingers[0], 6, tt, 0.005)
      #  self.finger_move(target_fingers[4], 1, MAX_SINGLE_FINGER_POS, 1.0)
       # self.finger_move(target_fingers[5], 1, MIN_SINGLE_FINGER_POS, 1.0)
       # self.finger_move(target_fingers[0], 6, HAND_FIST_CLOSE_POS, 1.0)
      #  for target in reversed(target_fingers):
         #   self.finger_move(target, 1, MAX_SINGLE_FINGER_POS, 1.0)
        '''

        self.robot.Write_Registers(
            COM_PORT,
            ROH_FINGER_POS_TARGET0,
            6,
            [
                H_BYTE(0),
                L_BYTE(0),
                H_BYTE(0),
                L_BYTE(0),
                H_BYTE(0),
                L_BYTE(0),
                H_BYTE(0),
                L_BYTE(0),
                H_BYTE(0),
                L_BYTE(0),
                H_BYTE(65535),
                L_BYTE(65535),
            ],
            ROH_ADDR,
            True,
        )  # 后两位是大拇指旋转  #前两位大拇指弯曲
    '''    
        return True