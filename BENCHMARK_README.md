# Machine Learning & AI Benchmark Suite

## 📋 Descrição

Este benchmark compara o desempenho de máquinas para tarefas de Machine Learning e Inteligência Artificial, com foco em desempenho single-core e multi-core.

**Máquinas sendo testadas:**
- Dell G3 3590 (Intel Core i7-9750H, 6 cores/12 threads, 2019)
- MacBook M4 Pro (Apple M4 Pro, 10-14 cores, 2024)

## 🧪 Testes Inclusos

O benchmark executa 7 testes diferentes:

1. **CPU Single-Core**: Cálculo de Fibonacci(35) - mede desempenho de um único core
2. **CPU Multi-Core**: Processamento paralelo - mede escalabilidade entre cores
3. **NumPy - Álgebra Linear**: Multiplicação de matrizes 2000x2000
4. **scikit-learn - Random Forest**: Treinamento de 100 árvores em 10k amostras
5. **scikit-learn - Logistic Regression**: Classificação no dataset digits
6. **PyTorch - Neural Network**: Treinamento de rede neural (se disponível)
7. **Pandas - Data Processing**: Operações de groupby, correlação e estatísticas

## 🚀 Como Rodar

### Pré-requisitos

- Python 3.7+
- pip ou pip3

### Instalação de Dependências

```bash
# No macOS (recomendado usar Homebrew)
pip3 install scikit-learn numpy pandas psutil

# No Windows/Linux
pip3 install scikit-learn numpy pandas psutil

# Opcional: Para testes de PyTorch (recomendado)
# macOS com M4 Pro:
pip3 install torch torchvision torchaudio

# Windows/Linux com GPU NVIDIA:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only:
pip3 install torch torchvision torchaudio
```

### Executar o Benchmark

```bash
python3 ml_ai_benchmark.py
```

O script vai:
1. Coletar informações do sistema
2. Executar todos os 7 testes
3. Salvar resultados em `benchmark_results.json`

**Tempo estimado**: 5-15 minutos (dependendo da máquina)

### Exemplo de Saída

```
============================================================
🚀 MACHINE LEARNING & AI BENCHMARK SUITE
============================================================
📊 Sistema: Darwin-23.3.0-arm64-with-glibc2.17
🔧 Processador: arm
💾 RAM: 36.0 GB
🧵 Cores: 10 físicos / 10 lógicos

🔹 Teste 1: CPU Single-Core (Fibonacci)
   Fibonacci(35) = 9227465
   Tempo: 0.8234s

🔹 Teste 2: CPU Multi-Core (Processamento Paralelo)
   Workers: 10
   Tempo: 2.1456s

[... mais testes ...]

✅ Benchmark concluído!
💾 Resultados salvos em: benchmark_results.json
```

## 📊 Resultados

Os resultados são salvos em `benchmark_results.json` com a seguinte estrutura:

```json
{
  "system_info": {
    "timestamp": "2026-03-09T14:59:16.969950",
    "platform": "Darwin-23.3.0-arm64-with-glibc2.17",
    "processor": "arm",
    "cpu_count": 10,
    "physical_cores": 10,
    "logical_cores": 10,
    "total_ram_gb": 36.0
  },
  "benchmarks": {
    "cpu_single_core_fibonacci": {
      "test": "Fibonacci(35)",
      "time_seconds": 0.8234,
      "result": 9227465
    },
    ...
  }
}
```

## 📈 Interpretação dos Resultados

### Single-Core Performance
- **Fibonacci(35)**: Quanto menor o tempo, melhor. Mede velocidade bruta de um core.
- Importante para: Desenvolvimento interativo, debugging, testes rápidos

### Multi-Core Performance
- **Processamento Paralelo**: Quanto menor o tempo, melhor. Mede escalabilidade.
- Importante para: Treinamento de modelos em paralelo, processamento em batch

### ML/AI Workloads
- **Random Forest, Logistic Regression, Neural Networks**: Quanto menor o tempo, melhor.
- **Acurácia**: Deve ser similar entre máquinas (validação de correção)

## 💡 Dicas de Otimização

### Para Dell G3 3590 (Windows/Linux)
- Feche programas desnecessários antes de rodar
- Plugue na tomada (não use bateria)
- Considere usar `taskset` no Linux para isolar cores

### Para MacBook M4 Pro (macOS)
- Feche aplicações em background
- Considere usar Activity Monitor para monitorar recursos
- O PyTorch pode usar Metal (GPU) automaticamente

## 🔧 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip3 install scikit-learn
```

### Erro: "Can't pickle local object"
- Certifique-se de estar usando a versão mais recente do script
- Tente: `pip3 install --upgrade scikit-learn`

### Benchmark muito lento
- Máquina pode estar com muitos programas abertos
- Tente fechar navegadores, IDEs, etc.
- Considere rodar em modo seguro/safe mode

## 📝 Notas

- Os testes usam dados aleatórios gerados, não dados reais
- A ordem dos testes é fixa para consistência
- Resultados podem variar entre execuções (±5-10% é normal)
- Para comparações mais precisas, rode 3 vezes e calcule a média

## 📧 Próximos Passos

1. Execute o benchmark em ambas as máquinas
2. Envie os dois arquivos `benchmark_results.json`
3. Vou gerar gráficos comparativos e um relatório visual

---

**Versão**: 1.0  
**Data**: Março 2026  
**Autor**: Manus AI Benchmark Suite
