# Office Benchmark Showdown v2

Benchmark comparativo de ML/AI: **Dell G3 3590** vs **MacBook M4 Pro**.

## Machines

| Machine | CPU | GPU | RAM | Year |
|---------|-----|-----|-----|------|
| Dell G3 3590 | Intel i7-9750H (6C/12T) | GTX 1650/1660 Ti | 16 GB DDR4 | 2019 |
| MacBook M4 Pro | Apple M4 Pro (10-14C) | Integrated (MPS) | 24-36 GB LPDDR5x | 2024 |

## Benchmarks (9 rounds)

1. **CPU Single-Core** — Fibonacci(35)
2. **CPU Multi-Core** — Parallel processing (multiprocessing pool)
3. **NumPy** — Matrix multiply 2000x2000
4. **Random Forest** — scikit-learn, 100 trees, 10k samples
5. **Logistic Regression** — scikit-learn, digits dataset
6. **PyTorch CPU** — Neural network, 5k samples, 50 epochs
7. **GPU MatMul** — 4096x4096 matrix multiply (CUDA or MPS)
8. **PyTorch GPU** — Neural network, 10k samples, 100 epochs
9. **Pandas** — groupby, correlation, describe (100k rows)

Each test runs **3 times** after a warmup, reporting the **median** time.

## Setup

```bash
# Install base dependencies
pip install -e .

# Install PyTorch (pick one)
pip install torch                         # CPU only
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA (Dell)
pip install torch                         # MPS is included by default (Mac)
```

## Usage

```bash
# Run on each machine (use --name for a friendly label)
python ml_ai_benchmark.py --name "Dell G3 3590"
python ml_ai_benchmark.py --name "MacBook M4 Pro"

# Compare results and generate charts
python visualize.py benchmark_results_dell_g3_3590.json benchmark_results_macbook_m4_pro.json
```

## Output

- `benchmark_results_<machine>.json` — raw results per machine
- `comparison_times.png` — side-by-side bar chart
- `speedup_ratios.png` — speedup ratio per test
- `winner_summary.png` — final scoreboard

## Files

| File | Description |
|------|-------------|
| `ml_ai_benchmark.py` | Main benchmark suite (v2) |
| `visualize.py` | Chart generator and comparison tool |
| `benchmark_specs.md` | Detailed hardware specs |
