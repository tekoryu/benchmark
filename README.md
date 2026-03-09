# Machine Learning & AI Benchmark Suite

Benchmark comparativo de desempenho para tarefas de Machine Learning e Inteligência Artificial.

## Máquinas Testadas

| Máquina | CPU | Cores | Ano | Arquitetura |
|---------|-----|-------|-----|-------------|
| Dell G3 3590 | Intel Core i7-9750H | 6 físicos / 12 threads | 2019 | 9nm |
| MacBook M4 Pro | Apple M4 Pro | 10-14 cores | 2024 | 3nm |

## Testes Inclusos

1. **CPU Single-Core** — Fibonacci(35)
2. **CPU Multi-Core** — Processamento paralelo
3. **NumPy** — Multiplicação de matrizes 2000x2000
4. **scikit-learn** — Random Forest (100 trees, 10k samples)
5. **scikit-learn** — Logistic Regression (digits dataset)
6. **PyTorch** — Neural Network (5000 samples, 10 epochs)
7. **Pandas** — Data Processing (groupby, correlation, describe)

## Como Rodar

```bash
# Instalar dependências
pip3 install scikit-learn numpy pandas psutil torch

# Executar benchmark
python3 ml_ai_benchmark.py
```

Os resultados são salvos em `benchmark_results.json`.

## Arquivos

- `ml_ai_benchmark.py` — Script principal do benchmark
- `BENCHMARK_README.md` — Instruções detalhadas
- `benchmark_specs.md` — Especificações técnicas das máquinas

---

*Benchmark criado com Manus AI — Março 2026*
