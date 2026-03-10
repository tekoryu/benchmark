#!/usr/bin/env python3
"""
Visualization Script for ML/AI Benchmark Comparisons
Reads all JSON results from output/ and generates comparison graphics
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

matplotlib.use('Agg')  # Use non-interactive backend


class BenchmarkVisualizer:
    """Visualizes benchmark results from multiple systems"""

    def __init__(self):
        self.results = {}
        self.graphics_dir = Path("graphics")
        self.graphics_dir.mkdir(exist_ok=True)

    def load_results(self):
        """Load all benchmark JSON files from output/ directory"""
        output_dir = Path("output")

        if not output_dir.exists():
            print("❌ No output/ directory found. Run benchmarks first with: benchmark")
            return False

        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            print("❌ No JSON files found in output/ directory")
            return False

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    computer_name = json_file.stem
                    self.results[computer_name] = data
                    print(f"✅ Loaded results for: {computer_name}")
            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")

        return len(self.results) > 0

    def extract_benchmark_data(self) -> Dict[str, Dict]:
        """Extract benchmark results organized by test name"""
        benchmark_data = {}

        for computer, data in self.results.items():
            benchmarks = data.get("benchmarks", {})
            for test_name, test_data in benchmarks.items():
                if test_name not in benchmark_data:
                    benchmark_data[test_name] = {}
                benchmark_data[test_name][computer] = test_data

        return benchmark_data

    def plot_time_comparison(self):
        """Generate bar chart comparing execution times"""
        benchmark_data = self.extract_benchmark_data()

        # Filter tests that have time_seconds
        time_tests = {}
        for test_name, systems_data in benchmark_data.items():
            times = {}
            for computer, test_data in systems_data.items():
                if "time_seconds" in test_data:
                    times[computer] = test_data["time_seconds"]
                elif "training_time_seconds" in test_data:
                    times[computer] = test_data["training_time_seconds"]

            if times:
                time_tests[test_name] = times

        if not time_tests:
            print("⚠️  No time comparison data available")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        computers = list(self.results.keys())
        x = np.arange(len(time_tests))
        width = 0.8 / len(computers)

        colors = plt.cm.Set3(np.linspace(0, 1, len(computers)))

        for idx, computer in enumerate(computers):
            times = [time_tests[test].get(computer, 0) for test in time_tests.keys()]
            ax.bar(x + idx * width, times, width, label=computer, color=colors[idx])

        ax.set_xlabel("Benchmark Test", fontsize=12, fontweight='bold')
        ax.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
        ax.set_title("ML/AI Benchmark Comparison - Execution Times", fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(computers) - 1) / 2)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in time_tests.keys()], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filepath = self.graphics_dir / "time_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filepath}")
        plt.close()

    def plot_speedup(self):
        """Generate speedup comparison (relative to slowest system)"""
        benchmark_data = self.extract_benchmark_data()

        # Filter tests with time_seconds
        time_tests = {}
        for test_name, systems_data in benchmark_data.items():
            times = {}
            for computer, test_data in systems_data.items():
                if "time_seconds" in test_data:
                    times[computer] = test_data["time_seconds"]
                elif "training_time_seconds" in test_data:
                    times[computer] = test_data["training_time_seconds"]

            if times and len(times) > 1:
                max_time = max(times.values())
                speedups = {comp: max_time / t for comp, t in times.items()}
                time_tests[test_name] = speedups

        if not time_tests:
            print("⚠️  Not enough systems for speedup comparison")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        computers = list(self.results.keys())
        x = np.arange(len(time_tests))
        width = 0.8 / len(computers)

        colors = plt.cm.Set3(np.linspace(0, 1, len(computers)))

        for idx, computer in enumerate(computers):
            speedups = [time_tests[test].get(computer, 1) for test in time_tests.keys()]
            ax.bar(x + idx * width, speedups, width, label=computer, color=colors[idx])

        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline')
        ax.set_xlabel("Benchmark Test", fontsize=12, fontweight='bold')
        ax.set_ylabel("Speedup (relative to slowest)", fontsize=12, fontweight='bold')
        ax.set_title("ML/AI Benchmark Comparison - Speedup Relative to Slowest", fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(computers) - 1) / 2)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in time_tests.keys()], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filepath = self.graphics_dir / "speedup_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filepath}")
        plt.close()

    def plot_system_specs(self):
        """Generate comparison of system specifications"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        computers = list(self.results.keys())

        # CPU Counts
        cpu_counts = []
        physical_cores = []
        logical_cores = []

        for computer in computers:
            system_info = self.results[computer].get("system_info", {})
            cpu_counts.append(system_info.get("cpu_count", 0))
            physical_cores.append(system_info.get("physical_cores", 0))
            logical_cores.append(system_info.get("logical_cores", 0))

        x = np.arange(len(computers))
        width = 0.25

        ax1.bar(x - width, physical_cores, width, label='Physical Cores', color='steelblue')
        ax1.bar(x, logical_cores, width, label='Logical Cores', color='lightcoral')
        ax1.set_ylabel("Core Count", fontweight='bold')
        ax1.set_title("CPU Cores Comparison", fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(computers)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # RAM
        ram_gb = []
        for computer in computers:
            system_info = self.results[computer].get("system_info", {})
            ram_gb.append(system_info.get("total_ram_gb", 0))

        ax2.bar(computers, ram_gb, color='mediumseagreen')
        ax2.set_ylabel("RAM (GB)", fontweight='bold')
        ax2.set_title("Total RAM Comparison", fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Platform
        platforms = []
        for computer in computers:
            system_info = self.results[computer].get("system_info", {})
            platform_str = system_info.get("platform", "Unknown").split()[0]
            platforms.append(platform_str)

        ax3.barh(computers, [1]*len(computers), color='mediumpurple')
        for idx, (comp, platform) in enumerate(zip(computers, platforms)):
            ax3.text(0.5, idx, platform, ha='center', va='center', fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.set_xticks([])
        ax3.set_title("Operating System", fontweight='bold')

        # Python & Processor
        processors = []
        for computer in computers:
            system_info = self.results[computer].get("system_info", {})
            proc = system_info.get("processor", "Unknown")[:30]
            processors.append(proc)

        ax4.axis('off')
        summary_text = "\n".join([
            f"{comp}:\n  {proc}\n"
            for comp, proc in zip(computers, processors)
        ])
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', family='monospace')
        ax4.set_title("Processor Info", fontweight='bold')

        fig.suptitle("System Specifications Comparison", fontsize=16, fontweight='bold')
        plt.tight_layout()
        filepath = self.graphics_dir / "system_specs.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filepath}")
        plt.close()

    def plot_summary_table(self):
        """Generate a summary table of all benchmarks"""
        benchmark_data = self.extract_benchmark_data()

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')

        computers = list(self.results.keys())

        # Build table data
        table_data = [['Benchmark'] + computers]

        for test_name, systems_data in sorted(benchmark_data.items()):
            row = [test_name.replace('_', ' ').title()]

            for computer in computers:
                test_data = systems_data.get(computer, {})

                # Try to get time metric
                time_val = None
                if "time_seconds" in test_data:
                    time_val = test_data["time_seconds"]
                elif "training_time_seconds" in test_data:
                    time_val = test_data["training_time_seconds"]

                if time_val is not None:
                    row.append(f"{time_val:.4f}s")
                else:
                    row.append("N/A")

            table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3] + [0.7/len(computers)]*len(computers))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)

        plt.title("Benchmark Results Summary", fontsize=14, fontweight='bold', pad=20)
        filepath = self.graphics_dir / "summary_table.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filepath}")
        plt.close()

    def generate_all_graphics(self):
        """Generate all comparison graphics"""
        if not self.load_results():
            return

        print("\n📊 Generating comparison graphics...")
        self.plot_time_comparison()
        self.plot_speedup()
        self.plot_system_specs()
        self.plot_summary_table()

        print(f"\n✅ All graphics saved to: {self.graphics_dir}/")


def main():
    """Main visualization function"""
    visualizer = BenchmarkVisualizer()
    visualizer.generate_all_graphics()


if __name__ == "__main__":
    main()
