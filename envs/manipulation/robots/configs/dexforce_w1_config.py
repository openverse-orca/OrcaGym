"""Dexforce W1 dual-arm configuration.

该文件定义 `dexforce_w1_gripper` 机器人在 OrcaGym 中使用的关节、执行器映射。
默认姿态采用全零，便于在空场景下逐关节调试。
如需保存特定的 Reset 姿态，可在同目录的
`dexforce_w1_config_variants.py` 中维护多套参数，然后在此导入。
"""

from __future__ import annotations
from typing import Dict, List


JointConfig = Dict[str, List[float]]


def _load_default_neutral() -> JointConfig:
    """加载默认中性位。

    优先从 `dexforce_w1_config_variants.py` 中获取 `DEFAULT_NEUTRAL`，
    若不存在则回退为全零姿态。
    """

    try:  # pragma: no cover - 调试环境中可能不存在变体文件
        from .dexforce_w1_config_variants import DEFAULT_NEUTRAL  # type: ignore

        if "right" in DEFAULT_NEUTRAL and "left" in DEFAULT_NEUTRAL:
            return DEFAULT_NEUTRAL  # type: ignore[return-value]
    except Exception:
        pass

    return {
        "right": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


DEFAULT_NEUTRAL = _load_default_neutral()


dexforce_w1_config = {
    "robot_type": "dual_arm",
    "base": {
        "base_body_name": "base_link",
        # Dexforce W1 模型通过 dummy_joint 固定在场景中
        "base_joint_name": "dummy_joint",
        "dummy_joint_name": "dummy_joint",
    },
    "right_arm": {
        "joint_names": [
            "RIGHT_J1",
            "RIGHT_J2",
            "RIGHT_J3",
            "RIGHT_J4",
            "RIGHT_J5",
            "RIGHT_J6",
            "RIGHT_J7",
        ],
        "neutral_joint_values": DEFAULT_NEUTRAL["right"],
        "motor_names": [
            "M_arm_r_01",
            "M_arm_r_02",
            "M_arm_r_03",
            "M_arm_r_04",
            "M_arm_r_05",
            "M_arm_r_06",
            "M_arm_r_07",
        ],
        "position_names": [
            "P_arm_r_01",
            "P_arm_r_02",
            "P_arm_r_03",
            "P_arm_r_04",
            "P_arm_r_05",
            "P_arm_r_06",
            "P_arm_r_07",
        ],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": [
            "LEFT_J1",
            "LEFT_J2",
            "LEFT_J3",
            "LEFT_J4",
            "LEFT_J5",
            "LEFT_J6",
            "LEFT_J7",
        ],
        "neutral_joint_values": DEFAULT_NEUTRAL["left"],
        "motor_names": [
            "M_arm_l_01",
            "M_arm_l_02",
            "M_arm_l_03",
            "M_arm_l_04",
            "M_arm_l_05",
            "M_arm_l_06",
            "M_arm_l_07",
        ],
        "position_names": [
            "P_arm_l_01",
            "P_arm_l_02",
            "P_arm_l_03",
            "P_arm_l_04",
            "P_arm_l_05",
            "P_arm_l_06",
            "P_arm_l_07",
        ],
        "ee_center_site_name": "ee_center_site",
    },
}


