import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

if project_root not in sys.path:
    sys.path.append(project_root)

from orca_gym.devices.keyboard import KeyboardClient
import asyncio

async def run_client():
    try:
        client_1 = KeyboardClient()
        client_2 = KeyboardClient()
        while True:
            state_1 = client_1.get_state()
            state_2 = client_2.get_state()
            print("Client 1:", state_1)
            print("Client 2:", state_2)
            await asyncio.sleep(1)
    finally:
        client_1.close()
        client_2.close()
        print("Client closed")

if __name__ == '__main__':
    asyncio.run(run_client())