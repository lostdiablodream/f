# Архитектура агрегации NVIDIA B200 8x (DGX B200 + масштабирование)

## 📐 1. ФИЗИЧЕСКАЯ ТОПОЛОГИЯ: DGX B200 (Single-Node 8x B200)

```
┌─────────────────────────────────────────────────────────────┐
│                    DGX B200 (10U, 14kW)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              NVSwitch Fabric Layer                   │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │  NVSwitch 5 #1     NVSwitch 5 #2           │     │   │
│  │  │  (28.8 Tb/s ea)    (28.8 Tb/s ea)          │     │   │
│  │  │  [72 ports ea]     [72 ports ea]           │     │   │
│  │  │                                             │     │   │
│  │  │  ↓ (144 cumulative ports = 72 GPU × 2)     │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  │           18 NVLink 5 connections per GPU             │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │             GPU Compute Layer (8 GPU)                │   │
│  │                                                       │   │
│  │  Bianca #1        Bianca #2      ...   Bianca #4   │   │
│  │  ┌────────────┐   ┌────────────┐       ┌────────────┐ │   │
│  │  │B200 GPU #0 │   │B200 GPU #2 │  ...  │B200 GPU #6 │ │   │
│  │  │192GB HBM3e │   │192GB HBM3e │       │192GB HBM3e │ │   │
│  │  │8 TB/s BW   │   │8 TB/s BW   │       │8 TB/s BW   │ │   │
│  │  └────────────┘   └────────────┘       └────────────┘ │   │
│  │  ┌────────────┐   ┌────────────┐       ┌────────────┐ │   │
│  │  │B200 GPU #1 │   │B200 GPU #3 │  ...  │B200 GPU #7 │ │   │
│  │  │192GB HBM3e │   │192GB HBM3e │       │192GB HBM3e │ │   │
│  │  │8 TB/s BW   │   │8 TB/s BW   │       │8 TB/s BW   │ │   │
│  │  └────────────┘   └────────────┘       └────────────┘ │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│           14.4 TB/s ALL-TO-ALL (NVSwitch)                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        CPU/Memory/Networking Layer                   │   │
│  │  • 2x Intel Xeon 8570 (56 cores ea = 112 cores)     │   │
│  │  • 4TB DDR5 RAM (CPU memory)                        │   │
│  │  • 8x ConnectX-7 400Gbps InfiniBand NICs (scale)    │   │
│  │  • 2x BlueField-3 DPU (storage/management)          │   │
│  │  • NVMe: 2x 1.92TB (OS) + 8x 3.84TB (cache)         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔗 2. NVLink 5 + NVSwitch 5 ДЕТАЛИ

| Параметр | Значение |
|----------|----------|
| **NVLink 5 версия** | 5-е поколение (200Gbps физически = 100GB/s логически) |
| **Ports per B200** | 18 (9 к NVSwitch #1 + 9 к NVSwitch #2) |
| **Bandwidth per port** | 100 GB/s unidirectional (200Gbps bidirectional) |
| **Total per GPU** | 18 × 100 GB/s = 1.8 TB/s (50GB/s × 18 в Hopper) |
| **Aggregate (all 8)** | 8 × 1.8 TB/s = 14.4 TB/s |
| **Topology** | **Non-blocking all-to-all** (any GPU ↔ any GPU at full 1.8TB/s) |
| **Hop count** | 1 hop (через NVSwitch) |
| **SerDes** | 2x 224Gbps PAM4 per port = 400Gbps per port (network notation) |
| **Latency** | <1 μs (microsecond) |

**NVSwitch 5 specs:**
- 2 NVSwitch ASICs per DGX B200
- 72 ports per ASIC (144 total)
- 28.8 Tb/s throughput per ASIC
- Non-blocking fabric (full bisection bandwidth)

---

## 📊 3. ПАРАЛЛЕЛИЗМ ВЫПОЛНЕНИЯ для Qwen3-Coder-480B

### Сценарий: Tensor Parallelism (TP) на 8 GPUs

```
480B параметров / 8 GPU = 60B параметров на GPU

Каждый B200 GPU получает:
- 60B параметры ≈ 45GB (FP8) или 90GB (FP16)
- Оставшееся: ~100-140GB для activations, KV cache, gradients
- ВЫВОД: ✓ Полностью помещается в 192GB HBM3e
```

### Распределение вычисления (TP8):

```
Forward Pass (один token):
┌─────────────────────────────────────────────────────────┐
│  GPU-0: Layer 0 (60B params)   → output tensor         │
│         AllReduce (1.8TB/s) ← GPU-1..7 outputs         │
│────────────────────────────────────────────────────────│
│  GPU-1: Layer 1 (60B params)   → output tensor         │
│         AllReduce (1.8TB/s) ← GPU-0,2..7 outputs       │
│  ...                                                    │
│  GPU-7: Layer 47 (60B params)  → output tensor         │
│         AllReduce (1.8TB/s) ← GPU-0..6 outputs         │
└─────────────────────────────────────────────────────────┘

Latency на слой:
- Compute: (60B × 2 × token_len) / (20 PFLOPS / 8) ≈ 0.24ms
- AllReduce: ~1-2ms (overlap с next layer)
- ИТОГО: ~1.5-2ms per layer × 48 layers = 72-96ms
```

---

## 🎯 4. AGGRЕГАЦИЯ: МАСШТАБИРОВАНИЕ ВЫШЕ 8x

### DGX B200 → DGX SuperPOD (72x GPU)

**Конфигурация NVL72:**

```
┌────────────────────────────────────────────────────────┐
│          NVL72 Rack (72 GPUs, ~124kW)                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────┐  ┌─────────────┐      ┌─────────────┐│
│  │ DGX B200 #0 │  │ DGX B200 #1 │  ... │ DGX B200 #8 ││
│  │  8 GPUs     │  │  8 GPUs     │      │  8 GPUs     ││
│  │ 1.44TB mem  │  │ 1.44TB mem  │      │ 1.44TB mem  ││
│  │ 14.4TB/s    │  │ 14.4TB/s    │      │ 14.4TB/s    ││
│  └─────────────┘  └─────────────┘      └─────────────┘│
│       ↓                ↓                    ↓          │
│  ┌────────────────────────────────────────────────┐   │
│  │     Scale-Out Layer: Jupiter DCN Network       │   │
│  │     (400Gbps × 8 NICs per DGX = 3.2Tbps)      │   │
│  │     InfiniBand (default) / 400GbE (alt)        │   │
│  └────────────────────────────────────────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │   NVSwitch Fabric (9 trays × 2 NVSwitch ea)     │  │
│  │   Connects all 72 GPUs: 1-hop all-to-all        │  │
│  │   Bisection BW: 259.2 Tb/s (all 72 GPUs)        │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
└────────────────────────────────────────────────────────┘

Topology: Flat NVLink (не hierarchical)
- Любой GPU ↔ любой GPU через 1 NVSwitch hop
- ~1 microsecond latency
```

**Vs. NVL36x2 (72 GPUs, 2 racks, предпочтительнее):**

```
Rack #1: 36 GPUs (NVL36)    ↑ 1 hop (NVSwitch)
Rack #2: 36 GPUs (NVL36)    ↑ 2 hops (inter-rack via DCN)

Плюсы: 
- Холоднее (10kW меньше)
- Дешевле (нет 72-GPU ограничений)
- Инте для GenAI workloads

Минусы:
- Inter-rack: +latency, но для training не критично
```

---

## ⚡ 5. ПРОИЗВОДИТЕЛЬНОСТЬ: Qwen3-Coder-480B на 8x B200

### Расчёт времени генерации 128KB (32K tokens output)

```
Параметры:
- Модель: 480B (TP8 = 60B per GPU)
- FP8 inference
- Batch size: 1 (autoregressive generation)
- Sequence length: 32K input + 32K output (context window)

Throughput (8 GPUs, TP8):
- Peak: 144 PFLOPS (FP4 with sparsity) per GPU × 8 = 1.152 EXAFLOPS
- Realistic (FP8, no sparsity): ~20 PFLOPS per GPU
- Per-token latency (MLPs + Attention): ~2ms

Генерация 32K tokens:
- 32,000 tokens × 2ms = 64,000ms = 64 сек
- Но: батч из 32-128 токенов обрабатывается параллельно
- Реальное: ~8-15 сек (с batching + pipeline overlap)
```

### Сравнение: TPU v6e vs B200

| Метрика | TPU v6e (10 чипов) | B200 8x | Победитель |
|---------|-------------------|---------|-----------|
| Требуемо чипов для 480B | 10 | 8 | **B200** |
| Пропускная способность | 16 TB/s | 14.4 TB/s | **TPU v6e** |
| Время генерации 128KB | 5-10 сек | 8-15 сек | **TPU v6e** |
| Топология | Jupiter DCN | NVSwitch | **B200** (проще) |
| Стоимость/месяц | $4,015 comm | ~$48K (приблизительно) | **TPU v6e** |
| Доступность | Google Cloud | AWS/GCP | **Tie** |

---

## 🔀 6. ПАРАЛЛЕЛИЗМ: АЛЬТЕРНАТИВНЫЕ СХЕМЫ

### Pipeline Parallelism (PP)

```
8 GPUs → 4 stages × 2 micro-batches

Stage 0: Layers 0-11 (GPU-0, GPU-1)
Stage 1: Layers 12-23 (GPU-2, GPU-3)
Stage 2: Layers 24-35 (GPU-4, GPU-5)
Stage 3: Layers 36-47 (GPU-6, GPU-7)

Advantages:
✓ Меньше AllReduce overhead
✓ Лучше для inference (latency)

Disadvantages:
✗ Pipeline bubbles
✗ Сложнее балансировка
```

### Data Parallelism (DP) + Tensor Parallelism (TP)

```
2-way DP × 4-way TP = 8 GPU total

Batch 0 → TP4 (GPUs 0-3)
Batch 1 → TP4 (GPUs 4-7)

Pros: Лучше для training
Cons: Дополнительные AllReduce (gradients)
```

---

## 🌐 7. СЕТЕВАЯ КОММУНИКАЦИЯ

### Intra-Pod (NVSwitch):
- **Bandwidth:** 14.4 TB/s (fully synchronized)
- **Latency:** <1 microsecond
- **Protocol:** NVLink 5 (proprietary coherent)
- **Use:** AllGather, Reduce, Broadcast, AllReduce

### Inter-Pod (InfiniBand DCN):
- **Bandwidth:** 8 × 400 Gbps = 3.2 Tbps (per DGX)
- **Latency:** ~1-5 microseconds (same-rack), ~10-20μs (cross-rack)
- **Protocol:** RDMA (InfiniBand SEND/WRITE)
- **Use:** Gradient synchronization (training) или model serving

### Bandwidth Utilization (480B, TP8):

```
Forward pass (1 token):
- Compute: 480B × 2 × seq_len = ~1-2 EXAFLOPS (1.2-2.4 trillion ops)
- Memory I/O: 480B × 2 bytes = 960GB read/write
- Required bandwidth: 960GB / 0.1ms ≈ 9.6 TB/s
- Available: 14.4 TB/s per GPU ÷ 8 (sharing) = 1.8 TB/s

✓ Достаточно (headroom: 1.8 vs 9.6 это компромисс)
  (Реально используется ~10-50% в зависимости от seq_len)
```

---

## 💾 8. ПАМЯТЬ: ДЕТАЛЬНОЕ РАСПРЕДЕЛЕНИЕ

### Per GPU (192GB HBM3e):

```
Model weights (480B TP8):
- 60B params × 2 bytes (FP16) = 120GB
- Остаток: 72GB

Активации + KV Cache:
- Seq length: 32,768
- Per token activation: ~120KB
- 32K tokens × 120KB = 3.84GB
- KV cache: 2 × (48 layers × 32K seq × 60B/8 × 2) ≈ 45GB
- Остаток: 72 - 3.84 - 45 = ~23GB (safe margin)

ИТОГО: ✓ ХВАТАЕТ с запасом
```

### DGX B200 (1.44TB совокупно):

```
Total model (480B FP16): 960GB
Activations (batch=8): 30GB
Optimizers/gradients (training): 960GB
KV cache (batch=128, seq=32K): 360GB

Training: 960 + 30 + 960 = 1,950GB (OVERFLOW)
Inference: 960 + 30 + 360 = 1,350GB ✓ (fits barely)
```

---

## 🔧 9. SOFTWARE FRAMEWORKS ПОДДЕРЖКА

### Tensor Parallelism реализация:

```python
# PyTorch Distributed (native):
torch.distributed.init_process_group("nccl")

# Megatron-LM (Meta):
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=1
)

# Hugging Face Transformers + FSDP:
from torch.distributed.fsdp import FullyShardedDataParallel
model = FullyShardedDataParallel(model)

# vLLM (Inference, optimized):
from vllm import LLM
llm = LLM(model="qwen/Qwen3-Coder-480B",
    tensor_parallel_size=8,  # Auto-shard across 8 GPUs
    dtype="float8")

# TensorRT-LLM (NVIDIA optimized):
- Supports B200 FP4 Transformer Engine
- ~2-4x speedup vs stock inference
```

---

## 📈 10. МАСШТАБИРОВАНИЕ ВЫШЕ: 72 GPU (NVL72)

```
Qwen3-Coder-480B на 72 GPU (TP72):
- Per GPU: 480B ÷ 72 = 6.67B params
- Memory: ~13GB weights + 180GB activations = 193GB ✓

Time (32K output):
- Per-token latency: 0.5-1ms (меньше из-за TP72)
- 32K tokens × 0.5ms = 16 сек (batch overlap)
```

---

## 💰 11. СТОИМОСТЬ (AWS P5 инстансы, 2025)

| Конфигурация | GPUs | Monthly (on-demand) | Instance |
|-------------|------|------------------|----------|
| Single B200 | 1 | ~$6,000 | p5.1xlarge |
| DGX B200 | 8 | ~$48,000 | p5.8xlarge |
| 2x DGX (16) | 16 | ~$96,000 | p5.16xlarge |
| NVL72 | 72 | ~$432,000 | Super Pod |
| 1-yr commit (NVL72) | 72 | ~$250,000 | Super Pod |

---

## ✅ ВЕРДИКТ для Qwen3-Coder-480B

**Лучший выбор: NVIDIA B200 8x (DGX B200)**

| Критерий | Оценка |
|----------|--------|
| Может ли поместиться | ✓ Да (60B per GPU) |
| Скорость (128KB) | 8-15 сек |
| Параллелизм | TP8 (лучший для 480B) |
| Сложность setup | Умеренная (PyTorch native) |
| Стоимость | ~$48K/месяц (AWS) |

**Vs TPU v6e (10 чипов):**
- TPU быстрее на 30-40%
- B200 проще в программировании (CUDA/PyTorch)
- TPU дешевле ($4K vs $48K на Google Cloud)
- B200 доступнее (AWS/GCP vs только Google Cloud)
