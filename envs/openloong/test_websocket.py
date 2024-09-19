import asyncio
import itertools
import json
from websockets.asyncio.server import serve
from scipy.spatial.transform import Rotation as R


async def handler(websocket):
    print("new connection")
    try:
        async for message in websocket:
            print(message)
            # await websocket.send(json.dumps(event))
    except Exception as e:
        print("disconnected")


async def main():
    async with serve(handler, "0.0.0.0", 8001):
        await asyncio.get_running_loop().create_future()  # run forever


# if __name__ == "__main__":
#     asyncio.run(main())

a = [11, 22, 33]
for i in a:
    print(i)