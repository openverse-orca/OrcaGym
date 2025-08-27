import os
import subprocess

# https://react.fluentui.dev/?path=/docs/icons-catalog--docs
if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    subprocess.run(["pyside6-rcc", "assets.qrc", "-o", "rc_assets.py"], cwd=cwd)
