"""测试基础设施：只读环境下日志文件写入降级。

OrcaGym 的模块级 logger（orca_gym/log/orca_log.py）在 import 时就会
打开 ``<repo>/logs/orca_gym.log``。在受限沙箱或只读 checkout 中跑测试
时，这会让所有测试在收集阶段就因 PermissionError 失败。

此处 patch RotatingFileHandler：目标路径不可写（PermissionError）时
降级到临时目录；正常可写环境零影响（不触发降级分支）。
"""

import logging.handlers
import os
import tempfile

_orig_rotating_init = logging.handlers.RotatingFileHandler.__init__


def _sandbox_safe_rotating_init(self, filename, *args, **kwargs):
    try:
        _orig_rotating_init(self, filename, *args, **kwargs)
    except PermissionError:
        fallback_dir = os.path.join(tempfile.gettempdir(), "orcagym_test_logs")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback = os.path.join(fallback_dir, os.path.basename(filename))
        _orig_rotating_init(self, fallback, *args, **kwargs)


logging.handlers.RotatingFileHandler.__init__ = _sandbox_safe_rotating_init
