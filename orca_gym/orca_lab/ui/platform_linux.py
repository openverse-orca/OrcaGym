import sys
from Xlib import X
from Xlib.display import Display

class PlatformHandlerLinux:
    def __init__(self):
        self.display = Display()