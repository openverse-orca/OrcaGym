import grpc
import orca_gym.orca_lab.protos.asset_service_pb2_grpc as asset_service_pb2_grpc
import orca_gym.orca_lab.protos.asset_service_pb2 as asset_service_pb2

import argparse
import asyncio
import sys


address = "localhost:50651"
scheme_name = "orca"


class AssetDownloadServer(asset_service_pb2_grpc.GrpcServiceServicer):
    def __init__(self):
        self.server = grpc.aio.server()
        asset_service_pb2_grpc.add_GrpcServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(address)

    async def start(self):
        await self.server.start()

    async def stop(self):
        await self.server.stop(0)

    def DownloadAsset(self, request, context):
        response = asset_service_pb2.DownloadAssetResponse()
        print(f"Received DownloadAsset request: {request.url}")
        return response


class AssetDownloadClient:
    def __init__(self):
        self.channel = grpc.aio.insecure_channel(address)
        self.stub = asset_service_pb2_grpc.GrpcServiceStub(self.channel)

    def _check_response(self, response):
        if response.status_code != asset_service_pb2.StatusCode.Success:
            raise Exception(f"Request failed. {response.error_message}")

    async def download_asset(self, url):
        request = asset_service_pb2.DownloadAssetRequest(url=url)
        response = await self.stub.DownloadAsset(request)
        return response


async def serve():
    server = AssetDownloadServer()
    await server.start()
    await server.server.wait_for_termination()


async def send_url(url):
    client = AssetDownloadClient()
    response = await client.download_asset(url)
    print(f"DownloadAsset response: {response.status_code}")


def is_protocol_registered():
    import winreg

    try:
        key_path = rf"SOFTWARE\Classes\{scheme_name}"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
        winreg.CloseKey(key)
        return True
    except FileNotFoundError:
        return False


def register_protocol():
    import winreg

    executable = sys.executable

    # We do not need a console window for handling the protocol
    if executable.endswith("python.exe"):
        executable = executable.replace("python.exe", "pythonw.exe")

    this_file = __file__

    try:
        # Create the main key for the custom scheme
        key_path = rf"SOFTWARE\Classes\{scheme_name}"
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, f"URL:{scheme_name} Protocol")
        winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        winreg.CloseKey(key)

        # Create the 'shell\open\command' subkeys
        command_key_path = rf"{key_path}\shell\open\command"
        command_key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, command_key_path)
        winreg.SetValueEx(
            command_key,
            "",
            0,
            winreg.REG_SZ,
            f'"{executable}" "{this_file}" --url "%1"',
        )
        winreg.CloseKey(command_key)

        print(f"URI scheme '{scheme_name}' registered successfully.")
    except Exception as e:
        print(f"Error registering URI scheme: {e}")


def unregister_protocol():
    import winreg

    try:
        key_path = rf"SOFTWARE\Classes\{scheme_name}"
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, rf"{key_path}\shell\open\command")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, rf"{key_path}\shell\open")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, rf"{key_path}\shell")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
        print(f"URI scheme '{scheme_name}' unregistered successfully.")
    except Exception as e:
        print(f"Error unregistering URI scheme: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", required=False, type=str, help="The URL of the asset to download."
    )
    parser.add_argument(
        "--serve", action="store_true", help="Run as server. For testing purpose."
    )
    parser.add_argument(
        "--register", action="store_true", help="Register custom protocol."
    )
    parser.add_argument(
        "--unregister", action="store_true", help="Unregister custom protocol."
    )
    parser.add_argument(
        "--query", action="store_true", help="Query if custom protocol is registered."
    )
    args = parser.parse_args()

    if args.serve:
        asyncio.run(serve())
    elif args.register:
        register_protocol()
    elif args.unregister:
        unregister_protocol()
    elif args.query:
        if is_protocol_registered():
            print(f"1")
        else:
            print(f"0")
    else:
        url = args.url
        if len(url) == 0:
            exit(-1)

        print(f"Sending URL to server: {url}")
        asyncio.run(send_url(url))
