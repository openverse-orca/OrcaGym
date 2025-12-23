"""
无手类型配置

该配置用于没有手部执行器的双臂机器人。
只包含手臂配置，不包含手部配置。
适用于只需要手臂操作，不需要抓取功能的场景。
"""

no_hand_config = {
    "hand_type": "none",  # 手部类型：'none' 或 'no_hand' 表示无手
    # 注意：无手配置不包含 left_hand 和 right_hand 字段
}

