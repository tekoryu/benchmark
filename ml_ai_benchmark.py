#!/usr/bin/env python3
"""
ML/AI Benchmark Suite v2 — Office Showdown Edition
Compara desempenho entre Dell G3 3590 e MacBook M4 Pro
"""

import argparse
import json
import multiprocessing
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------
try:
    from sklearn.datasets import load_digits, make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_RUNS = 3
WARMUP_RUNS = 1
MULTI_CORE_TASK_SIZE = 10_000


# ---------------------------------------------------------------------------
# Top-level function (required by multiprocessing)
# ---------------------------------------------------------------------------
def cpu_intensive_task(n: int) -> int:
    result = 0
    for _ in range(n):
        result += sum(j * j for j in range(100))
    return result


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------
class MLBenchmark:
    def __init__(self, machine_name: str | None = None):
        self.machine_name = machine_name or platform.node()
        self.results: dict = {
            "machine_name": self.machine_name,
            "system_info": self._get_system_info(),
            "benchmarks": {},
        }

    # -- helpers -------------------------------------------------------------

    def _get_system_info(self) -> dict:
        info = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": multiprocessing.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "gpu_cuda": None,
            "gpu_mps": False,
        }
        if HAS_TORCH:
            if torch.cuda.is_available():
                info["gpu_cuda"] = {
                    "name": torch.cuda.get_device_name(0),
                    "vram_gb": round(
                        torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
                    ),
                    "cuda_version": torch.version.cuda,
                }
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["gpu_mps"] = True
        return info

    def _run_timed(self, fn, *args, **kwargs) -> tuple[float, object]:
        """Warmup + NUM_RUNS, return (median_seconds, last_result)."""
        # warmup
        print("      Warming up...")
        for _ in range(WARMUP_RUNS):
            fn(*args, **kwargs)

        times: list[float] = []
        result = None
        for i in range(NUM_RUNS):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"      Run {i + 1}/{NUM_RUNS}: {elapsed:.4f}s")

        median = statistics.median(times)
        print(f"      Median: {median:.4f}s")
        return median, result

    def _get_torch_device(self) -> "torch.device":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _sync_device(device: "torch.device"):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except AttributeError:
                pass  # older PyTorch

    # -- benchmarks ----------------------------------------------------------

    def benchmark_cpu_single_core(self):
        """Round 1 — CPU Single-Core: Fibonacci(35)"""
        print("\n   Round 1: CPU Single-Core (Fibonacci)")

        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        median, result = self._run_timed(fibonacci, 35)
        self.results["benchmarks"]["cpu_single_core_fibonacci"] = {
            "test": "Fibonacci(35)",
            "time_seconds": median,
            "result": result,
        }

    def benchmark_cpu_multi_core(self):
        """Round 2 — CPU Multi-Core: Parallel Processing"""
        print("\n   Round 2: CPU Multi-Core (Parallel Processing)")
        num_workers = multiprocessing.cpu_count()

        def parallel_work():
            tasks = [MULTI_CORE_TASK_SIZE] * num_workers
            with multiprocessing.Pool(num_workers) as pool:
                return pool.map(cpu_intensive_task, tasks)

        median, results_list = self._run_timed(parallel_work)
        self.results["benchmarks"]["cpu_multi_core"] = {
            "test": "Parallel Processing",
            "workers": num_workers,
            "time_seconds": median,
            "tasks_completed": len(results_list),
        }

    def benchmark_numpy_operations(self):
        """Round 3 — NumPy Matrix Multiplication 2000x2000"""
        print("\n   Round 3: NumPy - Matrix Multiply (2000x2000)")
        size = 2000
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)

        def matmul():
            return np.dot(A, B)

        median, _ = self._run_timed(matmul)
        self.results["benchmarks"]["numpy_matmul"] = {
            "test": f"Matrix Multiply {size}x{size}",
            "time_seconds": median,
            "matrix_size": size,
        }

    def benchmark_sklearn_random_forest(self):
        """Round 4 — scikit-learn Random Forest"""
        if not HAS_SKLEARN:
            print("\n   Round 4: Random Forest - SKIPPED (scikit-learn not installed)")
            return
        print("\n   Round 4: scikit-learn - Random Forest")

        X, y = make_classification(
            n_samples=10_000,
            n_features=100,
            n_informative=50,
            n_redundant=20,
            random_state=42,
        )
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

        def train():
            rf.fit(X, y)
            return rf.score(X, y)

        median, accuracy = self._run_timed(train)
        print(f"      Accuracy: {accuracy:.4f}")
        self.results["benchmarks"]["sklearn_random_forest"] = {
            "test": "Random Forest (100 trees, 10k samples, 100 features)",
            "time_seconds": median,
            "accuracy": accuracy,
        }

    def benchmark_sklearn_logistic_regression(self):
        """Round 5 — scikit-learn Logistic Regression"""
        if not HAS_SKLEARN:
            print("\n   Round 5: Logistic Regression - SKIPPED (scikit-learn not installed)")
            return
        print("\n   Round 5: scikit-learn - Logistic Regression")

        digits = load_digits()
        X, y = digits.data, digits.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)

        def train():
            lr.fit(X_scaled, y)
            return lr.score(X_scaled, y)

        median, accuracy = self._run_timed(train)
        print(f"      Accuracy: {accuracy:.4f}")
        self.results["benchmarks"]["sklearn_logistic_regression"] = {
            "test": "Logistic Regression (digits dataset)",
            "time_seconds": median,
            "accuracy": accuracy,
        }

    def benchmark_pytorch_cpu(self):
        """Round 6 — PyTorch Neural Network on CPU"""
        if not HAS_TORCH:
            print("\n   Round 6: PyTorch CPU - SKIPPED (torch not installed)")
            return
        print("\n   Round 6: PyTorch - Neural Network (CPU)")

        device = torch.device("cpu")
        print(f"      Device: {device}")

        X_train = torch.randn(5000, 50, device=device)
        y_train = torch.randint(0, 2, (5000,), device=device)

        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(50, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),
                )

            def forward(self, x):
                return self.net(x)

        criterion = nn.CrossEntropyLoss()

        def train():
            model = SimpleNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for _ in range(50):
                optimizer.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward()
                optimizer.step()
            return loss.item()

        median, final_loss = self._run_timed(train)
        print(f"      Final loss: {final_loss:.4f}")
        self.results["benchmarks"]["pytorch_cpu_nn"] = {
            "test": "Neural Network (5k samples, 50 epochs, CPU)",
            "device": "cpu",
            "time_seconds": median,
            "final_loss": final_loss,
        }

    def benchmark_gpu_matmul(self):
        """Round 7 — GPU Matrix Multiplication"""
        if not HAS_TORCH:
            print("\n   Round 7: GPU MatMul - SKIPPED (torch not installed)")
            return
        device = self._get_torch_device()
        if device.type == "cpu":
            print("\n   Round 7: GPU MatMul - SKIPPED (no GPU available)")
            return
        print(f"\n   Round 7: GPU Matrix Multiply (4096x4096) [{device}]")

        size = 4096

        def gpu_matmul():
            A = torch.randn(size, size, device=device)
            B = torch.randn(size, size, device=device)
            C = torch.mm(A, B)
            self._sync_device(device)
            return C.shape

        median, _ = self._run_timed(gpu_matmul)
        self.results["benchmarks"]["gpu_matmul"] = {
            "test": f"GPU Matrix Multiply {size}x{size}",
            "device": str(device),
            "time_seconds": median,
        }

    def benchmark_pytorch_gpu_nn(self):
        """Round 8 — PyTorch Neural Network on GPU"""
        if not HAS_TORCH:
            print("\n   Round 8: PyTorch GPU NN - SKIPPED (torch not installed)")
            return
        device = self._get_torch_device()
        if device.type == "cpu":
            print("\n   Round 8: PyTorch GPU NN - SKIPPED (no GPU available)")
            return
        print(f"\n   Round 8: PyTorch - Neural Network (GPU: {device})")

        X_train = torch.randn(10_000, 100, device=device)
        y_train = torch.randint(0, 2, (10_000,), device=device)

        class LargerNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(100, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),
                )

            def forward(self, x):
                return self.net(x)

        criterion = nn.CrossEntropyLoss()

        def train():
            model = LargerNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for _ in range(100):
                optimizer.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward()
                optimizer.step()
            self._sync_device(device)
            return loss.item()

        median, final_loss = self._run_timed(train)
        print(f"      Final loss: {final_loss:.4f}")
        self.results["benchmarks"]["pytorch_gpu_nn"] = {
            "test": f"Neural Network (10k samples, 100 epochs, {device})",
            "device": str(device),
            "time_seconds": median,
            "final_loss": final_loss,
        }

    def benchmark_pandas_operations(self):
        """Round 9 — Pandas Data Processing"""
        if not HAS_PANDAS:
            print("\n   Round 9: Pandas - SKIPPED (pandas not installed)")
            return
        print("\n   Round 9: Pandas - Data Processing (100k rows)")

        df = pd.DataFrame(
            {
                "A": np.random.randn(100_000),
                "B": np.random.randn(100_000),
                "C": np.random.randn(100_000),
                "D": np.random.choice(["X", "Y", "Z"], 100_000),
            }
        )

        def process():
            _ = df.groupby("D")[["A", "B", "C"]].mean()
            _ = df[["A", "B", "C"]].corr()
            _ = df.describe()
            _ = df.select_dtypes(include="number").std()

        median, _ = self._run_timed(process)
        self.results["benchmarks"]["pandas_operations"] = {
            "test": "Pandas - groupby, correlation, describe, std (100k rows)",
            "time_seconds": median,
            "rows": 100_000,
        }

    # -- orchestration -------------------------------------------------------

    def run_all_benchmarks(self) -> dict:
        info = self.results["system_info"]
        gpu_label = "None"
        if info["gpu_cuda"]:
            gpu_label = info["gpu_cuda"]["name"]
        elif info["gpu_mps"]:
            gpu_label = "Apple MPS"

        print("=" * 60)
        print("   OFFICE BENCHMARK SHOWDOWN v2")
        print("   May the fastest silicon win!")
        print("=" * 60)
        print(f"\n   Contender: {self.machine_name}")
        print(f"   CPU:  {info['processor'] or info['platform']}")
        print(f"   GPU:  {gpu_label}")
        print(f"   RAM:  {info['total_ram_gb']} GB")
        print(
            f"   Cores: {info['physical_cores']} physical / {info['logical_cores']} logical"
        )
        print("-" * 60)

        # missing libs warning
        missing = []
        if not HAS_SKLEARN:
            missing.append("scikit-learn")
        if not HAS_TORCH:
            missing.append("torch")
        if not HAS_PANDAS:
            missing.append("pandas")
        if missing:
            print(f"\n   Missing optional libs: {', '.join(missing)}")
            print("   Some rounds will be skipped.")

        benchmarks = [
            self.benchmark_cpu_single_core,
            self.benchmark_cpu_multi_core,
            self.benchmark_numpy_operations,
            self.benchmark_sklearn_random_forest,
            self.benchmark_sklearn_logistic_regression,
            self.benchmark_pytorch_cpu,
            self.benchmark_gpu_matmul,
            self.benchmark_pytorch_gpu_nn,
            self.benchmark_pandas_operations,
        ]

        for bench_fn in benchmarks:
            try:
                bench_fn()
            except Exception as e:
                name = bench_fn.__name__.removeprefix("benchmark_")
                print(f"\n   FAILED: {name} -- {e}")
                self.results["benchmarks"][name] = {"error": str(e)}

        print("\n" + "=" * 60)
        print("   ALL ROUNDS COMPLETE!")
        print("=" * 60)

        return self.results

    def save_results(self) -> Path:
        safe_name = self.machine_name.replace(" ", "_").lower()
        filename = f"benchmark_results_{safe_name}.json"
        output_path = Path(filename)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n   Results saved to: {output_path}")
        print("   Now run the same on the other machine and use")
        print("   visualize.py to see who wins!")
        return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ML/AI Benchmark Suite v2 — Office Showdown Edition"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Machine name (defaults to hostname)",
    )
    args = parser.parse_args()

    benchmark = MLBenchmark(machine_name=args.name)
    benchmark.run_all_benchmarks()
    benchmark.save_results()


if __name__ == "__main__":
    main()
