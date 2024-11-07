from envs.gymloong.utils.task_registry import task_registry
from envs.gymloong.environ.Azure_Loong.Azure_Loong_env import AzureLoongEnv
from envs.gymloong.environ.Azure_Loong.Azure_Loong_config import Azure_Loong_config, Azure_Loong_configPPO

task_registry.register("AzureLoong", AzureLoongEnv, Azure_Loong_config(), Azure_Loong_configPPO())