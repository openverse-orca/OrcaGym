from envs.gymloong import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


from envs.gymloong.utils.task_registry import task_registry
from .Azure_Loong.Azure_Loong_env import AzureLoongEnv
from .Azure_Loong.Azure_Loong_config import Azure_Loong_config, Azure_Loong_configPPO

task_registry.register("AzureLoong", AzureLoongEnv, Azure_Loong_config(), Azure_Loong_configPPO())
