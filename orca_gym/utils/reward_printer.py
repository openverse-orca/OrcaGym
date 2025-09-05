from datetime import datetime
from typing import Optional, Any, SupportsFloat
import numpy as np
import os


class RewardPrinter:
    PRINT_DETAIL = True

    def __init__(self, buffer_size : int = 100):
        self._reward_data : dict[str, np.ndarray] = {}
        self._reward_coeff : dict[str, float] = {}
        self._buffer_index : dict[str, int] = {}
        self._buffer_size = buffer_size

        self.reward_history = {}

        file_dir_name = "reward_history"
        if not os.path.exists(file_dir_name):
            os.makedirs(file_dir_name)

        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_name = os.path.join(file_dir_name, f"reward_history_{datetime_str}.txt")

    def print_reward(self, message : str, reward : Optional[float] = 0, coeff : Optional[float] = 1.0):
        if self._reward_data.get(message) is None:
            self._reward_data[message] = np.zeros(self._buffer_size)
            self._reward_coeff[message] = coeff if coeff is not None else 0.0
            self._buffer_index[message] = 0
            self._reward_data[message][self._buffer_index[message]] = reward
            self._buffer_index[message] += 1
        else:
            if self._buffer_index[message] < self._buffer_size:
                self._reward_data[message][self._buffer_index[message]] = reward
                self._buffer_index[message] += 1
            elif self._all_buffer_full(): 
                print("============= print reward =============")
                for key, value in self._reward_data.items():
                    if self.PRINT_DETAIL:
                        print(key, f"{value.mean():.10f}\t\t\t|{value.max():.4e}|{value.min():.4e}|{value.std():.4e}|{self._reward_coeff[key]:.4e}")

                        # log all rewards
                        if key not in self.reward_history:
                            self.reward_history[key] = []
                        self.reward_history[key].append(value.mean()) 

                        with open(self.file_name, "a") as f:
                            f.write(f"{key.split(':')[0]}_{self._reward_coeff[key]:.2e}:{value.mean():.10f}\n")
                    else:
                        mean_value = value.mean()
                        print(key, f"{mean_value:.10f} | {(mean_value / self._reward_coeff[key]):.10f}")
                print("========================================")
                self._buffer_index = {key: 0 for key in self._buffer_index.keys()}
                
    def _all_buffer_full(self):
        for key, value in self._buffer_index.items():
            if value < self._buffer_size:
                return False
        return True

# def print_reward(message : str, agent_id : str, reward : Optional[float] = None):
#     if PRINT_REWARD and print_reward.agent_id == agent_id:
#         if print_reward.print:
#             if print_reward.agent_id == agent_id:
#                 if reward is None:
#                     print(message)
#                 else:
#                     print(message, reward)
#         else:
#             if reward is None:
#                 print(message)
#             else:
#                 print(message, reward)

# def print_reward_begin(agent_id : str):
#     if PRINT_REWARD:
#         if not hasattr(print_reward, "last_time"):
#             print_reward.last_time = datetime.now()
#             print("Current time: ", print_reward.last_time)
#             print_reward.print = True

#         if not hasattr(print_reward, "agent_id"):
#             print_reward.agent_id = agent_id
#         elif print_reward.agent_id != agent_id:
#             return

#         if not hasattr(print_reward, "data"):
#             print_reward.data = {}

#         if (datetime.now() - print_reward.last_time).seconds > PRINT_REWARD_INTERVAL:
#             print_reward.print = True
#             print_reward.last_time = datetime.now()
        
# def print_reward_end(agent_id : str):
#     if PRINT_REWARD and print_reward.agent_id == agent_id:
#         print_reward.print = False