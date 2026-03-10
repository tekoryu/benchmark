#!/usr/bin/env python3
"""
Machine Learning & AI Benchmark Suite
Compara desempenho single-core e multi-core entre Dell G3 3590 e MacBook M4 Pro
"""

import json
import time
import multiprocessing
import platform
import psutil
import numpy as np
from datetime import datetime
from pathlib import Path

# Tentar importar bibliotecas de ML
try:
    from sklearn.datasets import make_classification, load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn não instalado. Pulando testes de sklearn.")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch não instalado. Pulando testes de PyTorch.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  Pandas não instalado. Pulando testes de Pandas.")


def cpu_intensive_task(n):
    """Tarefa intensiva de CPU (função global para multiprocessing)"""
    result = 0
    for i in range(n):
        result += sum([j**2 for j in range(100)])
    return result


class MLBenchmark:
    """Suite de benchmarks de Machine Learning"""
    
    def __init__(self):
        self.results = {
            "system_info": self._get_system_info(),
            "benchmarks": {}
        }
        
    def _get_system_info(self):
        """Coleta informações do sistema"""
        return {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": multiprocessing.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
        }
    
    def benchmark_cpu_single_core(self):
        """Teste de CPU single-core: Fibonacci"""
        print("\n🔹 Teste 1: CPU Single-Core (Fibonacci)")
        
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        start = time.perf_counter()
        result = fibonacci(35)
        elapsed = time.perf_counter() - start
        
        print(f"   Fibonacci(35) = {result}")
        print(f"   Tempo: {elapsed:.4f}s")
        
        self.results["benchmarks"]["cpu_single_core_fibonacci"] = {
            "test": "Fibonacci(35)",
            "time_seconds": elapsed,
            "result": result
        }
        
        return elapsed
    
    def benchmark_cpu_multi_core(self):
        """Teste de CPU multi-core: Processamento paralelo"""
        print("\n🔹 Teste 2: CPU Multi-Core (Processamento Paralelo)")
        
        num_workers = multiprocessing.cpu_count()
        tasks = [1000000] * num_workers
        
        start = time.perf_counter()
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(cpu_intensive_task, tasks)
        elapsed = time.perf_counter() - start
        
        print(f"   Workers: {num_workers}")
        print(f"   Tempo: {elapsed:.4f}s")
        
        self.results["benchmarks"]["cpu_multi_core"] = {
            "test": "Processamento Paralelo",
            "workers": num_workers,
            "time_seconds": elapsed,
            "tasks_completed": len(results)
        }
        
        return elapsed
    
    def benchmark_numpy_operations(self):
        """Teste de operações NumPy (álgebra linear)"""
        print("\n🔹 Teste 3: NumPy - Álgebra Linear")
        
        # Multiplicação de matrizes grandes
        matrix_size = 2000
        A = np.random.randn(matrix_size, matrix_size)
        B = np.random.randn(matrix_size, matrix_size)
        
        start = time.perf_counter()
        C = np.dot(A, B)
        elapsed = time.perf_counter() - start
        
        print(f"   Multiplicação de matrizes {matrix_size}x{matrix_size}")
        print(f"   Tempo: {elapsed:.4f}s")
        
        self.results["benchmarks"]["numpy_matmul"] = {
            "test": f"Multiplicação de matrizes {matrix_size}x{matrix_size}",
            "time_seconds": elapsed,
            "matrix_size": matrix_size
        }
        
        return elapsed
    
    def benchmark_sklearn_random_forest(self):
        """Teste de Random Forest com scikit-learn"""
        if not HAS_SKLEARN:
            print("\n🔹 Teste 4: scikit-learn Random Forest - PULADO (não instalado)")
            return None
        
        print("\n🔹 Teste 4: scikit-learn - Random Forest")
        
        # Gerar dataset
        X, y = make_classification(
            n_samples=10000,
            n_features=100,
            n_informative=50,
            n_redundant=20,
            random_state=42
        )
        
        # Treinar Random Forest
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        
        start = time.perf_counter()
        rf.fit(X, y)
        elapsed = time.perf_counter() - start
        
        # Predição
        start_pred = time.perf_counter()
        predictions = rf.predict(X[:1000])
        elapsed_pred = time.perf_counter() - start_pred
        
        accuracy = rf.score(X, y)
        
        print(f"   Treino: {elapsed:.4f}s")
        print(f"   Predição (1000 samples): {elapsed_pred:.4f}s")
        print(f"   Acurácia: {accuracy:.4f}")
        
        self.results["benchmarks"]["sklearn_random_forest"] = {
            "test": "Random Forest (100 trees, 10k samples, 100 features)",
            "training_time_seconds": elapsed,
            "prediction_time_seconds": elapsed_pred,
            "accuracy": accuracy
        }
        
        return elapsed + elapsed_pred
    
    def benchmark_sklearn_logistic_regression(self):
        """Teste de regressão logística com scikit-learn"""
        if not HAS_SKLEARN:
            print("\n🔹 Teste 5: scikit-learn Logistic Regression - PULADO (não instalado)")
            return None
        
        print("\n🔹 Teste 5: scikit-learn - Logistic Regression")
        
        # Usar dataset digits
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Treinar
        lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
        
        start = time.perf_counter()
        lr.fit(X_scaled, y)
        elapsed = time.perf_counter() - start
        
        accuracy = lr.score(X_scaled, y)
        
        print(f"   Tempo de treino: {elapsed:.4f}s")
        print(f"   Acurácia: {accuracy:.4f}")
        
        self.results["benchmarks"]["sklearn_logistic_regression"] = {
            "test": "Logistic Regression (digits dataset)",
            "training_time_seconds": elapsed,
            "accuracy": accuracy
        }
        
        return elapsed
    
    def benchmark_pytorch_neural_network(self):
        """Teste de rede neural com PyTorch"""
        if not HAS_TORCH:
            print("\n🔹 Teste 6: PyTorch Neural Network - PULADO (não instalado)")
            return None

        print("\n🔹 Teste 6: PyTorch - Neural Network Training")

        # Configurar device (MPS para Mac, CUDA para Dell, CPU fallback)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"   Device: {device}")
        
        # Gerar dados
        X_train = torch.randn(5000, 50).to(device)
        y_train = torch.randint(0, 2, (5000,)).to(device)
        
        # Definir modelo
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(50, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 2)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = SimpleNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Treinar
        start = time.perf_counter()
        epochs = 10
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - start
        
        print(f"   Tempo de treino (10 epochs): {elapsed:.4f}s")
        print(f"   Loss final: {loss.item():.4f}")
        
        self.results["benchmarks"]["pytorch_nn"] = {
            "test": "Neural Network (5000 samples, 10 epochs)",
            "device": str(device),
            "training_time_seconds": elapsed,
            "final_loss": float(loss.item())
        }
        
        return elapsed
    
    def benchmark_pandas_operations(self):
        """Teste de operações com Pandas"""
        if not HAS_PANDAS:
            print("\n🔹 Teste 7: Pandas Operations - PULADO (não instalado)")
            return None
        
        print("\n🔹 Teste 7: Pandas - Data Processing")
        
        # Criar DataFrame grande
        df = pd.DataFrame({
            'A': np.random.randn(100000),
            'B': np.random.randn(100000),
            'C': np.random.randn(100000),
            'D': np.random.choice(['X', 'Y', 'Z'], 100000)
        })
        
        start = time.perf_counter()

        # Operações típicas de ML
        result1 = df.groupby('D')[['A', 'B', 'C']].mean()
        result2 = df[['A', 'B', 'C']].corr()
        result3 = df.describe()
        result4 = df.select_dtypes(include='number').std()

        elapsed = time.perf_counter() - start
        
        print(f"   Operações: groupby, correlação, describe, std")
        print(f"   Tempo total: {elapsed:.4f}s")
        
        self.results["benchmarks"]["pandas_operations"] = {
            "test": "Pandas - groupby, correlation, describe (100k rows)",
            "time_seconds": elapsed,
            "rows": 100000
        }
        
        return elapsed
    
    def run_all_benchmarks(self):
        """Executa todos os benchmarks"""
        print("=" * 60)
        print("🚀 MACHINE LEARNING & AI BENCHMARK SUITE")
        print("=" * 60)
        print(f"\n📊 Sistema: {self.results['system_info']['platform']}")
        print(f"🔧 Processador: {self.results['system_info']['processor']}")
        print(f"💾 RAM: {self.results['system_info']['total_ram_gb']:.1f} GB")
        print(f"🧵 Cores: {self.results['system_info']['physical_cores']} físicos / {self.results['system_info']['logical_cores']} lógicos")
        
        try:
            self.benchmark_cpu_single_core()
            self.benchmark_cpu_multi_core()
            self.benchmark_numpy_operations()
            self.benchmark_sklearn_random_forest()
            self.benchmark_sklearn_logistic_regression()
            self.benchmark_pytorch_neural_network()
            self.benchmark_pandas_operations()
        except Exception as e:
            print(f"\n❌ Erro durante execução: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("✅ Benchmark concluído!")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, filename=None):
        """Salva resultados em JSON"""
        if filename is None:
            # Use computer name to avoid overwrites
            computer_name = platform.node()
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            filename = output_dir / f"{computer_name}.json"

        output_path = Path(filename)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Resultados salvos em: {output_path}")
        return output_path


def main():
    """Função principal"""
    benchmark = MLBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.save_results()


if __name__ == "__main__":
    main()
