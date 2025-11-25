#!/usr/bin/env python3
"""
XBot 性能测试脚本
测试不同设备（MUSA GPU, CUDA GPU, CPU）的推理性能

使用方法:
    python run_xbot_benchmark.py --device musa --warmup 100 --iterations 1000
    python run_xbot_benchmark.py --device auto --compare_all  # 对比所有可用设备
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
from collections import deque
import statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from envs.xbot_gym.xbot_simple_env import XBotSimpleEnv
from orca_gym.utils.device_utils import get_torch_device, get_gpu_info, print_gpu_info
import psutil
import gc


class PerformanceBenchmark:
    """性能测试类"""
    
    def __init__(self, policy, torch_device, device_name: str):
        self.policy = policy
        self.torch_device = torch_device
        self.device_name = device_name
        self.inference_times = []
        self.memory_usage = []
        
    def warmup(self, num_warmup: int = 100):
        """预热：运行多次推理以稳定性能"""
        print(f"🔥 预热中... ({num_warmup} 次推理)")
        dummy_obs = np.random.randn(705).astype(np.float32)
        
        for _ in range(num_warmup):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(dummy_obs).float().to(self.torch_device)
                _ = self.policy(obs_tensor)
        
        # 同步 GPU（如果有）
        if "musa" in str(self.torch_device) or "cuda" in str(self.torch_device):
            if "musa" in str(self.torch_device):
                torch.musa.synchronize()
            else:
                torch.cuda.synchronize()
        
        print("✓ 预热完成")
    
    def benchmark_single_inference(self, num_iterations: int = 1000):
        """单次推理性能测试"""
        print(f"\n📊 单次推理性能测试 ({num_iterations} 次迭代)")
        print("=" * 80)
        
        dummy_obs = np.random.randn(705).astype(np.float32)
        self.inference_times = []
        
        # 记录初始内存
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 测试循环
        for i in range(num_iterations):
            # 开始计时
            if "musa" in str(self.torch_device) or "cuda" in str(self.torch_device):
                if "musa" in str(self.torch_device):
                    torch.musa.synchronize()
                else:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
            
            # 推理
            with torch.no_grad():
                obs_tensor = torch.from_numpy(dummy_obs).float().to(self.torch_device)
                action_tensor = self.policy(obs_tensor)
                action = action_tensor.cpu().numpy()
            
            # 结束计时
            if "musa" in str(self.torch_device) or "cuda" in str(self.torch_device):
                if "musa" in str(self.torch_device):
                    torch.musa.synchronize()
                else:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
            else:
                end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            self.inference_times.append(inference_time)
            
            # 记录内存使用（每100次）
            if (i + 1) % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.memory_usage.append(current_memory - initial_memory)
        
        # 计算统计信息
        self._print_statistics()
    
    def benchmark_batch_inference(self, batch_sizes: list = [1, 4, 8, 16, 32]):
        """批量推理性能测试"""
        print(f"\n📊 批量推理性能测试")
        print("=" * 80)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n  测试批量大小: {batch_size}")
            dummy_obs = np.random.randn(batch_size, 705).astype(np.float32)
            batch_times = []
            
            # 预热
            for _ in range(10):
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(dummy_obs).float().to(self.torch_device)
                    _ = self.policy(obs_tensor)
            
            # 同步
            if "musa" in str(self.torch_device) or "cuda" in str(self.torch_device):
                if "musa" in str(self.torch_device):
                    torch.musa.synchronize()
                else:
                    torch.cuda.synchronize()
            
            # 测试
            num_iterations = 100
            for _ in range(num_iterations):
                if "musa" in str(self.torch_device) or "cuda" in str(self.torch_device):
                    if "musa" in str(self.torch_device):
                        torch.musa.synchronize()
                    else:
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                else:
                    start_time = time.perf_counter()
                
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(dummy_obs).float().to(self.torch_device)
                    action_tensor = self.policy(obs_tensor)
                    _ = action_tensor.cpu().numpy()
                
                if "musa" in str(self.torch_device) or "cuda" in str(self.torch_device):
                    if "musa" in str(self.torch_device):
                        torch.musa.synchronize()
                    else:
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                else:
                    end_time = time.perf_counter()
                
                batch_time = (end_time - start_time) * 1000  # 毫秒
                batch_times.append(batch_time)
            
            # 计算统计
            mean_time = np.mean(batch_times)
            std_time = np.std(batch_times)
            throughput = batch_size / (mean_time / 1000)  # 样本/秒
            time_per_sample = mean_time / batch_size  # 单样本时间（毫秒）
            
            results[batch_size] = {
                'mean_ms': mean_time,
                'std_ms': std_time,
                'throughput': throughput,
                'time_per_sample_ms': time_per_sample
            }
            
            print(f"    平均时间: {mean_time:.3f} ms ± {std_time:.3f} ms")
            print(f"    吞吐量: {throughput:.1f} 样本/秒")
            print(f"    单样本时间: {time_per_sample:.6f} ms ({time_per_sample*1000:.3f} μs)")
        
        return results
    
    def _print_statistics(self):
        """打印统计信息"""
        if not self.inference_times:
            return
        
        times_ms = np.array(self.inference_times)
        
        # 基本统计
        mean_ms = np.mean(times_ms)
        std_ms = np.std(times_ms)
        min_ms = np.min(times_ms)
        max_ms = np.max(times_ms)
        median_ms = np.median(times_ms)
        
        # 百分位数
        p50 = np.percentile(times_ms, 50)
        p95 = np.percentile(times_ms, 95)
        p99 = np.percentile(times_ms, 99)
        
        # FPS
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0
        
        # 内存使用
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        max_memory = np.max(self.memory_usage) if self.memory_usage else 0
        
        print(f"\n📈 性能统计 ({self.device_name}):")
        print(f"  ⏱️  推理时间:")
        print(f"     - 平均: {mean_ms:.3f} ms ± {std_ms:.3f} ms")
        print(f"     - 中位数: {median_ms:.3f} ms")
        print(f"     - 最小: {min_ms:.3f} ms")
        print(f"     - 最大: {max_ms:.3f} ms")
        print(f"     - P50: {p50:.3f} ms")
        print(f"     - P95: {p95:.3f} ms")
        print(f"     - P99: {p99:.3f} ms")
        print(f"  🚀 吞吐量:")
        print(f"     - FPS: {fps:.1f} 帧/秒")
        print(f"     - 吞吐量: {1000.0/mean_ms:.1f} 推理/秒")
        if avg_memory > 0:
            print(f"  💾 内存使用:")
            print(f"     - 平均: {avg_memory:.1f} MB")
            print(f"     - 最大: {max_memory:.1f} MB")
    
    def get_summary(self):
        """获取性能摘要"""
        if not self.inference_times:
            return None
        
        times_ms = np.array(self.inference_times)
        mean_ms = np.mean(times_ms)
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0
        
        return {
            'device': self.device_name,
            'mean_ms': mean_ms,
            'std_ms': np.std(times_ms),
            'p50_ms': np.percentile(times_ms, 50),
            'p95_ms': np.percentile(times_ms, 95),
            'p99_ms': np.percentile(times_ms, 99),
            'fps': fps,
            'throughput': 1000.0 / mean_ms
        }


def load_xbot_policy(policy_path: str, device: str = "auto"):
    """加载XBot策略"""
    if device == "auto":
        torch_device = get_torch_device(try_to_use_gpu=True)
        device_str = str(torch_device)
        if "musa" in device_str:
            device = "musa"
        elif "cuda" in device_str:
            device = "cuda"
        else:
            device = "cpu"
    else:
        if device == "musa":
            try:
                import torch_musa
                if torch.musa.is_available():
                    torch_device = torch.device("musa:0")
                else:
                    raise RuntimeError("MUSA GPU not available")
            except ImportError:
                raise RuntimeError("torch_musa not installed")
        elif device == "cuda":
            if torch.cuda.is_available():
                torch_device = torch.device("cuda:0")
            else:
                raise RuntimeError("CUDA not available")
        else:
            torch_device = torch.device("cpu")
    
    policy = torch.jit.load(policy_path, map_location=torch_device)
    policy.eval()
    policy.to(torch_device)
    
    return policy, torch_device, device


def benchmark_device(device: str, policy_path: str, warmup: int, iterations: int, batch_sizes: list):
    """对单个设备进行性能测试"""
    print(f"\n{'='*80}")
    print(f"🔬 性能测试: {device.upper()}")
    print(f"{'='*80}")
    
    try:
        # 加载策略
        policy, torch_device, device_name = load_xbot_policy(policy_path, device)
        print(f"✓ 策略已加载到设备: {torch_device}")
        
        # 创建测试对象
        benchmark = PerformanceBenchmark(policy, torch_device, device_name)
        
        # 预热
        benchmark.warmup(num_warmup=warmup)
        
        # 单次推理测试
        benchmark.benchmark_single_inference(num_iterations=iterations)
        
        # 批量推理测试
        if batch_sizes:
            benchmark.benchmark_batch_inference(batch_sizes=batch_sizes)
        
        # 清理
        del policy
        gc.collect()
        if "musa" in str(torch_device) or "cuda" in str(torch_device):
            if "musa" in str(torch_device):
                torch.musa.empty_cache()
            else:
                torch.cuda.empty_cache()
        
        return benchmark.get_summary()
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None


def compare_all_devices(policy_path: str, warmup: int, iterations: int, export_json: str = None):
    """对比所有可用设备"""
    print(f"\n{'='*80}")
    print(f"📊 设备性能对比测试")
    print(f"{'='*80}")
    
    # 检测可用设备
    available_devices = []
    
    # 检查 MUSA
    try:
        import torch_musa
        if torch.musa.is_available():
            available_devices.append("musa")
    except:
        pass
    
    # 检查 CUDA
    if torch.cuda.is_available():
        available_devices.append("cuda")
    
    # CPU 总是可用
    available_devices.append("cpu")
    
    print(f"\n可用设备: {', '.join(available_devices)}")
    
    # 测试每个设备
    results = {}
    for device in available_devices:
        summary = benchmark_device(device, policy_path, warmup, iterations, batch_sizes=[])
        if summary:
            results[device] = summary
        
        # 设备之间稍作延迟
        time.sleep(1)
    
    # 打印对比结果
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"📊 性能对比总结")
        print(f"{'='*80}")
        print(f"{'设备':<10} {'平均时间(ms)':<15} {'FPS':<10} {'吞吐量(推理/秒)':<20} {'P95(ms)':<10} {'P99(ms)':<10}")
        print(f"{'-'*90}")
        
        # 按平均时间排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_ms'])
        
        for device, summary in sorted_results:
            print(f"{device.upper():<10} {summary['mean_ms']:<15.3f} {summary['fps']:<10.1f} {summary['throughput']:<20.1f} {summary['p95_ms']:<10.3f} {summary['p99_ms']:<10.3f}")
        
        # 计算加速比
        if len(sorted_results) > 1:
            baseline = sorted_results[0][1]['mean_ms']  # 最快设备
            print(f"\n加速比 (相对于最快设备):")
            for device, summary in sorted_results:
                speedup = baseline / summary['mean_ms']
                print(f"  {device.upper()}: {speedup:.2f}x")
            
            # 详细分析
            print(f"\n📈 性能分析:")
            fastest = sorted_results[0]
            print(f"  - 最快设备: {fastest[0].upper()} ({fastest[1]['mean_ms']:.3f} ms)")
            
            if len(sorted_results) > 1:
                second = sorted_results[1]
                ratio = second[1]['mean_ms'] / fastest[1]['mean_ms']
                print(f"  - 第二快设备: {second[0].upper()} ({second[1]['mean_ms']:.3f} ms, {ratio:.2f}x 慢)")
            
            # 延迟稳定性分析
            print(f"\n⏱️  延迟稳定性 (P99/P50 比值，越小越稳定):")
            for device, summary in sorted_results:
                stability = summary['p99_ms'] / summary['p50_ms'] if summary['p50_ms'] > 0 else float('inf')
                print(f"  - {device.upper()}: {stability:.2f}x")
        
        # 导出 JSON（如果指定）
        if export_json:
            import json
            export_data = {
                'test_config': {
                    'warmup': warmup,
                    'iterations': iterations,
                    'policy_path': policy_path
                },
                'results': results,
                'summary': {
                    'fastest_device': sorted_results[0][0] if sorted_results else None,
                    'speedup_ratios': {
                        device: baseline / summary['mean_ms'] 
                        for device, summary in sorted_results
                    } if len(sorted_results) > 1 else {}
                }
            }
            with open(export_json, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"\n💾 结果已导出到: {export_json}")


def main():
    parser = argparse.ArgumentParser(description="XBot 性能测试脚本")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'musa', 'auto'], 
                       default='auto', help="测试设备 (默认: auto)")
    parser.add_argument("--policy_path", type=str, 
                       default=None, help="策略文件路径 (默认: config/policy_example.pt)")
    parser.add_argument("--warmup", type=int, default=100, 
                       help="预热迭代次数 (默认: 100)")
    parser.add_argument("--iterations", type=int, default=1000, 
                       help="测试迭代次数 (默认: 1000)")
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[1, 4, 8, 16, 32],
                       help="批量推理测试的批量大小 (默认: 1 4 8 16 32)")
    parser.add_argument("--compare_all", action='store_true',
                       help="对比所有可用设备的性能")
    parser.add_argument("--no_batch", action='store_true',
                       help="跳过批量推理测试")
    parser.add_argument("--export_json", type=str, default=None,
                       help="导出结果到 JSON 文件")
    
    args = parser.parse_args()
    
    # 默认策略路径
    if args.policy_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.policy_path = os.path.join(script_dir, "config", "policy_example.pt")
    
    if not os.path.exists(args.policy_path):
        print(f"❌ 策略文件不存在: {args.policy_path}")
        return
    
    print("="*80)
    print("🚀 XBot 性能测试")
    print("="*80)
    print(f"策略文件: {args.policy_path}")
    print(f"预热迭代: {args.warmup}")
    print(f"测试迭代: {args.iterations}")
    
    # 打印 GPU 信息
    print_gpu_info()
    
    if args.compare_all:
        # 对比所有设备
        compare_all_devices(args.policy_path, args.warmup, args.iterations, args.export_json)
    else:
        # 测试单个设备
        batch_sizes = [] if args.no_batch else args.batch_sizes
        benchmark_device(args.device, args.policy_path, args.warmup, args.iterations, batch_sizes)


if __name__ == "__main__":
    main()

