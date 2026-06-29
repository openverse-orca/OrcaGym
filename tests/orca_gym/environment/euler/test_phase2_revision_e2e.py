"""阶段二变更修订 — Step 7: 端到端验证(Lesson 1/2/3)。

CPU 离线测试可在 sandbox 内运行;GPU/在线测试标记为 skip,
需在宿主机手动运行(见文档 §7.1)。

对齐文档: docs/design/development/orca_gym_euler_phase2_revision_development.md Step 7
"""

import os
import pathlib
import subprocess
import sys
import unittest

_ORCA_PLAYGROUND = pathlib.Path(__file__).resolve().parents[4].parent / "OrcaPlayground"
_PYTHON = sys.executable


@unittest.skipUnless(
    _ORCA_PLAYGROUND.exists(),
    f"OrcaPlayground 不存在: {_ORCA_PLAYGROUND}",
)
class TestPhase2Lesson1HelloEuler(unittest.TestCase):
    """Lesson 1: Hello Euler 离线仿真(CPU,sandbox 内可运行)。"""

    def test_lesson1_runs_without_error(self):
        """Lesson 1 离线仿真完整运行,退出码 0。"""
        script = _ORCA_PLAYGROUND / "examples" / "euler" / "01_hello_euler" / "hello_euler.py"
        if not script.exists():
            self.skipTest(f"Lesson 1 脚本不存在: {script}")
        result = subprocess.run(
            [_PYTHON, str(script), "--steps", "50"],
            capture_output=True, text=True,
            cwd=str(_ORCA_PLAYGROUND),
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Lesson 1 失败:\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        # 验证输出含关键日志(模型加载 + 步进完成)
        self.assertIn("环境创建成功", result.stdout)
        self.assertIn("第 1 课验证通过", result.stdout)


@unittest.skipUnless(
    os.environ.get("ORCA_EULER_E2E_ONLINE") == "1",
    "在线测试需 ORCA_EULER_E2E_ONLINE=1 + OrcaStudio 运行",
)
class TestPhase2Lesson2OnlineRender(unittest.TestCase):
    """Lesson 2: 在线 gRPC 渲染(宿主机 + OrcaStudio)。

    手动运行命令(宿主机):
        cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python \\
            examples/euler/02_online_render/online_render.py
    """

    def test_lesson2_runs_without_error(self):
        script = _ORCA_PLAYGROUND / "examples" / "euler" / "02_online_render" / "online_render.py"
        result = subprocess.run(
            [_PYTHON, str(script), "--steps", "50"],
            capture_output=True, text=True,
            cwd=str(_ORCA_PLAYGROUND),
            timeout=300,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Lesson 2 失败:\nstdout={result.stdout}\nstderr={result.stderr}",
        )


@unittest.skipUnless(
    os.environ.get("ORCA_EULER_E2E_GPU") == "1",
    "GPU 训练需 ORCA_EULER_E2E_GPU=1 + 白名单旁路(AGENTS.md 规则 3)",
)
class TestPhase2Lesson3RlPpo(unittest.TestCase):
    """Lesson 3: RL PPO 训练(GPU,白名单旁路)。

    手动运行命令(宿主机,无管道,输出重定向到文件):
        cd <OrcaPlayground-root> && <conda-base>/envs/orca/bin/python \\
            examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000 \\
            > /tmp/lesson3_ppo.log 2>&1

    注意: 禁止使用 shell 管道(如 2>&1 | tail),会触发 sandbox 包裹,
    导致 CUDA_ERROR_304(AGENTS.md 规则 3)。
    """

    def test_lesson3_runs_without_error(self):
        script = _ORCA_PLAYGROUND / "examples" / "euler" / "03_rl_ppo" / "train_ppo.py"
        result = subprocess.run(
            [_PYTHON, str(script),
             "--total-timesteps", "20000",
             "--device", "cuda"],
            capture_output=True, text=True,
            cwd=str(_ORCA_PLAYGROUND),
            timeout=600,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Lesson 3 失败:\nstdout={result.stdout}\nstderr={result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
