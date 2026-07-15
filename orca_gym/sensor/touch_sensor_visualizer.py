import time
import numpy as np
from collections import deque
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty


def _visualizer_main(queue, window_title, history_length):
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(window_title)
    history: dict[str, deque[float]] = {}

    try:
        while True:
            try:
                msg = queue.get(timeout=0.033)
                if msg is None:
                    return
                for name, val in msg.items():
                    if name not in history:
                        history[name] = deque(maxlen=history_length)
                    history[name].append(val)

                while not queue.empty():
                    msg = queue.get_nowait()
                    if msg is None:
                        return
                    for name, val in msg.items():
                        if name not in history:
                            history[name] = deque(maxlen=history_length)
                        history[name].append(val)
            except QueueEmpty:
                pass

            if history:
                ax.clear()
                for name, values in history.items():
                    x_data = np.arange(len(values))
                    ax.plot(x_data, list(values), label=name)
                ax.set_xlabel("Frame")
                ax.set_ylabel("Value")
                ax.set_title("Touch Sensor Values")
                ax.axhline(y=0, color='gray', linewidth=0.5)
                ax.legend(loc='upper right', fontsize=7)
                ax.set_xlim(left=0)
                fig.tight_layout()

            fig.canvas.draw()
            fig.canvas.flush_events()
    finally:
        plt.close(fig)


class TouchSensorVisualizer:
    def __init__(self, window_title: str = "Touch Sensor Visualizer",
                 history_length: int = 200):
        self._history_length = history_length
        self._queue: Queue = Queue()
        self._process = Process(
            target=_visualizer_main,
            args=(self._queue, window_title, history_length),
            daemon=True,
        )
        self._process.start()

    def update_data(self, sensor_data: dict[str, float]):
        self._queue.put(sensor_data)

    def draw(self):
        pass

    def close(self):
        self._queue.put(None)
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import random

    SENSOR_NAMES = [
        "touch_thumb",
        "touch_index",
        "touch_middle",
        "touch_ring",
        "touch_pinky",
    ]

    viz = TouchSensorVisualizer()
    try:
        for _ in range(10000):
            sensor_data = {
                name: random.uniform(-15, 15) for name in SENSOR_NAMES
            }
            viz.update_data(sensor_data)
            time.sleep(0.0166)
    finally:
        viz.close()