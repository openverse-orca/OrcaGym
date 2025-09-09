import inspect
import asyncio
from PySide6 import QtCore, QtWidgets, QtGui


def connect(signal: QtCore.SignalInstance, func):

    assert isinstance(signal, QtCore.SignalInstance)

    if inspect.iscoroutinefunction(func):

        def wrapper(*args, **kwargs):
            asyncio.ensure_future(func(*args, **kwargs))

        signal.connect(wrapper)

    elif inspect.isfunction(func) or inspect.ismethod(func) or inspect.isbuiltin(func):
        signal.connect(func)
    else:
        raise Exception("Invalid function type.")
