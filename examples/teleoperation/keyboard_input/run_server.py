import os
import sys
import time

current_file_path = os.path.abspath('')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

if project_root not in sys.path:
    sys.path.append(project_root)

from orca_gym.devices.keyboard import KeyboardServer
import asyncio

async def run_server():
    try:
        server = KeyboardServer()
        await server.run()
    finally:    
        server.close()
        print("Server closed")

if __name__ == '__main__':
    asyncio.run(run_server())
