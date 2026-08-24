#!/usr/bin/env python3
"""CPU 内存监控脚本 - 用于监控 sft.py 训练进程的内存使用情况.

使用方法:
    python3 mem_monitor.py --interval 30 --output memory_log.csv

功能:
    1. 监控系统整体内存使用
    2. 监控 sft.py 进程组（主进程+所有子进程）的内存使用
    3. 检测内存上涨趋势（内存泄漏预警）
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any

import psutil


class MemoryMonitor:
    """内存监控器，用于监控系统整体和进程组的 CPU 内存使用情况."""

    def __init__(
        self,
        interval: float = 30.0,
        window_size: int = 20,
        output_file: str = "memory_log.csv",
        process_name: str = "sft",
    ) -> None:
        """
        Args:
            interval: 采样间隔（秒）
            window_size: 用于趋势分析的滑动窗口大小
            output_file: 输出 CSV 文件路径
            process_name: 用于识别训练进程的进程名关键词
        """
        self.interval = interval
        self.window_size = window_size
        self.output_file = output_file
        self.process_name = process_name

        # 历史数据队列（用于趋势分析）
        self.system_memory_history: deque[float] = deque(maxlen=window_size)
        self.process_memory_history: deque[float] = deque(maxlen=window_size)

        # 初始化 CSV 文件
        self._init_csv()

    def _init_csv(self) -> None:
        """初始化 CSV 文件并写入表头."""
        headers = [
            "timestamp",
            "system_used_gb",
            "system_total_gb",
            "system_percent",
            "sft_process_count",
            "sft_rss_gb",
            "sft_vms_gb",
            "leak_warning",
        ]

        with open(self.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def find_sft_processes(self) -> list[psutil.Process]:
        """查找所有 sft 相关的进程（包括主进程和子进程）.

        通过遍历所有进程，匹配进程名或命令行中包含 sft 关键词.
        """
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "ppid"]):
            try:
                info = proc.info
                name = info.get("name", "") or ""
                cmdline = info.get("cmdline", []) or []
                cmdline_str = " ".join(cmdline) if cmdline else ""

                # 匹配进程名或命令行中包含 sft 关键词
                if self.process_name in name or self.process_name in cmdline_str:
                    processes.append(proc)
                    continue

                # 检查父进程是否在 sft 进程树中
                try:
                    parent = proc.parent()
                    if parent:
                        parent_name = parent.name() or ""
                        parent_cmdline = " ".join(parent.cmdline() or [])
                        if self.process_name in parent_name or self.process_name in parent_cmdline:
                            processes.append(proc)
                            continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return processes

    def get_process_tree_memory(self, root_procs: list[psutil.Process]) -> dict[str, Any]:
        """获取进程树的总内存使用情况.

        遍历所有子进程，累加内存使用.
        """
        all_procs = set()
        rss_total = 0
        vms_total = 0

        for root in root_procs:
            try:
                # 获取进程及其所有子孙进程
                descendants = []
                try:
                    descendants = root.children(recursive=True)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                all_procs.add(root.pid)
                rss_total += root.memory_info().rss
                vms_total += root.memory_info().vms

                for child in descendants:
                    try:
                        if child.pid not in all_procs:
                            all_procs.add(child.pid)
                            mem_info = child.memory_info()
                            rss_total += mem_info.rss
                            vms_total += mem_info.vms
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            "count": len(all_procs),
            "rss_gb": rss_total / (1024**3),
            "vms_gb": vms_total / (1024**3),
        }

    def get_system_memory(self) -> dict[str, Any]:
        """获取系统整体内存使用情况."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "used_gb": (mem.total - mem.available) / (1024**3),
            "percent": mem.percent,
        }

    def analyze_trend(self, history: deque[float]) -> tuple[str, float]:
        """分析内存使用趋势.

        使用简单线性回归判断内存是否持续上涨.

        Returns:
            (趋势状态, 斜率) - 状态: "stable", "rising", "falling", "insufficient_data"
        """
        if len(history) < 5:
            return "insufficient_data", 0.0

        n = len(history)
        x = list(range(n))
        y = list(history)

        # 简单线性回归
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable", 0.0

        slope = numerator / denominator

        # 根据斜率判断趋势（单位: GB/采样周期）
        threshold = 0.1  # 每个采样周期增长超过 0.1 GB 认为是上涨
        if slope > threshold:
            return "rising", slope
        elif slope < -threshold:
            return "falling", slope
        else:
            return "stable", slope

    def check_leak_warning(self) -> str:
        """检查是否有内存泄漏警告."""
        warnings = []

        # 检查系统内存趋势
        if len(self.system_memory_history) >= 5:
            trend, slope = self.analyze_trend(self.system_memory_history)
            if trend == "rising":
                warnings.append(f"系统内存上涨({slope:.3f}GB/周期)")

        # 检查进程内存趋势
        if len(self.process_memory_history) >= 5:
            trend, slope = self.analyze_trend(self.process_memory_history)
            if trend == "rising":
                warnings.append(f"进程内存上涨({slope:.3f}GB/周期)")

        return "; ".join(warnings) if warnings else "normal"

    def sample(self) -> dict[str, Any] | None:
        """采集一次内存数据."""
        # 系统内存
        sys_mem = self.get_system_memory()

        # 进程组内存
        root_procs = self.find_sft_processes()
        if not root_procs:
            proc_mem = {"count": 0, "rss_gb": 0, "vms_gb": 0}
        else:
            proc_mem = self.get_process_tree_memory(root_procs)

        # 更新历史数据
        self.system_memory_history.append(sys_mem["used_gb"])
        self.process_memory_history.append(proc_mem["rss_gb"])

        # 检查内存泄漏
        leak_warning = self.check_leak_warning()

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system": sys_mem,
            "process": proc_mem,
            "leak_warning": leak_warning,
        }

    def print_status(self, data: dict[str, Any]) -> None:
        """打印当前状态."""
        sys_mem = data["system"]
        proc_mem = data["process"]

        print(f"\n[{data['timestamp']}]")
        print(
            f"  系统内存: {sys_mem['used_gb']:.2f} / {sys_mem['total_gb']:.2f} GB "
            f"({sys_mem['percent']:.1f}%)"
        )

        # 系统内存趋势
        if len(self.system_memory_history) >= 5:
            trend, slope = self.analyze_trend(self.system_memory_history)
            trend_str = {
                "rising": "↑ 上涨",
                "falling": "↓ 下降",
                "stable": "→ 稳定",
                "insufficient_data": "...",
            }.get(trend, "unknown")
            print(f"  系统趋势: {trend_str} ({slope:+.3f} GB/周期)")

        print(
            f"  进程组: {proc_mem['count']} 个进程, "
            f"RSS: {proc_mem['rss_gb']:.2f} GB, VMS: {proc_mem['vms_gb']:.2f} GB"
        )

        # 进程内存趋势
        if len(self.process_memory_history) >= 5:
            trend, slope = self.analyze_trend(self.process_memory_history)
            trend_str = {
                "rising": "↑ 上涨",
                "falling": "↓ 下降",
                "stable": "→ 稳定",
                "insufficient_data": "...",
            }.get(trend, "unknown")
            print(f"  进程趋势: {trend_str} ({slope:+.3f} GB/周期)")

        if data["leak_warning"] != "normal":
            print(f"  ⚠️  警告: {data['leak_warning']}")

    def write_to_csv(self, data: dict[str, Any]) -> None:
        """写入数据到 CSV."""
        sys_mem = data["system"]
        proc_mem = data["process"]

        row = [
            data["timestamp"],
            round(sys_mem["used_gb"], 3),
            round(sys_mem["total_gb"], 3),
            round(sys_mem["percent"], 1),
            proc_mem["count"],
            round(proc_mem["rss_gb"], 3),
            round(proc_mem["vms_gb"], 3),
            data["leak_warning"],
        ]

        with open(self.output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def run(self) -> None:
        """主循环."""
        print(f"开始监控 CPU 内存 (关键词: {self.process_name})")
        print(f"采样间隔: {self.interval} 秒")
        print(f"输出文件: {self.output_file}")
        print(f"按 Ctrl+C 停止监控\n")

        try:
            while True:
                data = self.sample()
                self.print_status(data)
                self.write_to_csv(data)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n监控已停止")
            # 输出最终统计
            if len(self.system_memory_history) > 0:
                print(f"\n总计采样 {len(self.system_memory_history)} 次")
                sys_start = self.system_memory_history[0]
                sys_end = self.system_memory_history[-1]
                print(f"系统内存: {sys_start:.2f} GB -> {sys_end:.2f} GB (变化: {sys_end - sys_start:+.2f} GB)")

                if len(self.process_memory_history) > 0:
                    proc_start = self.process_memory_history[0]
                    proc_end = self.process_memory_history[-1]
                    print(f"进程内存: {proc_start:.2f} GB -> {proc_end:.2f} GB (变化: {proc_end - proc_start:+.2f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="监控 sft.py CPU 内存使用情况")
    parser.add_argument(
        "--interval",
        type=float,
        default=30,
        help="采样间隔（秒），默认 30 秒",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="memory_log.csv",
        help="输出 CSV 文件路径，默认 memory_log.csv",
    )
    parser.add_argument(
        "--process",
        type=str,
        default="sft",
        help="进程识别关键词，默认 'sft'",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="趋势分析窗口大小，默认 20 个样本",
    )

    args = parser.parse_args()

    monitor = MemoryMonitor(
        interval=args.interval,
        window_size=args.window,
        output_file=args.output,
        process_name=args.process,
    )
    monitor.run()


if __name__ == "__main__":
    main()
