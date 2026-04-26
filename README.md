# T-GMA: Thermal-Aware GPU Memory Allocator

> A low-level GPU memory management engine built directly on the NVIDIA CUDA Driver API and NVML — designed to eliminate external VRAM fragmentation in real-time by responding to hardware thermal telemetry.

---

![T-GMA Memory Compaction Event](fragmentation_graph.png)

*The graph above shows a live compaction event: fragmentation score spikes after artificial holes are introduced, then drops to **zero** after the thermal watchdog triggers active memory consolidation.*

---

## The Problem

Standard GPU memory APIs like `cudaMalloc` abstract away physical VRAM placement. Under sustained ML workloads, this causes:

- **External Fragmentation** — free memory exists, but in scattered, non-contiguous chunks that cannot satisfy large allocation requests.
- **Thermal Hotspots** — uncontrolled physical data layout causes uneven heat distribution across VRAM silicon.
- **Thermal Throttling** — the GPU reduces clock speeds to survive, silently degrading training throughput with no obvious error.

These issues compound silently — the driver reports "enough free VRAM" while the system cannot actually service the next large tensor allocation.

---

## The Solution

T-GMA takes control at the hardware level using **Virtual Memory Management (VMM)** from the CUDA Driver API — the same low-level interface used by frameworks like PyTorch's `ExpandingAllocator`.

The core insight is the **decoupling of Virtual Addresses from Physical VRAM frames**:

- A static **1 GB Virtual Address hallway** is reserved at startup via `cuMemAddressReserve`.
- Physical **2 MB VRAM pages** are minted on demand via `cuMemCreate` and mapped into virtual slots via `cuMemMap`.
- A background **Watchdog Thread** polls GPU temperature at 1 Hz via NVML.
- When a thermal threshold is crossed, the engine performs **Active Memory Compaction** — physically relocating scattered data into contiguous pages using a Device-to-Device `cuMemcpyDtoD`.
- **The application's virtual pointers never change.** Zero disruption to the running workload.

---

## Key Technical Achievements

| Feature | Implementation Detail |
|---|---|
| **VMM-Based Page Manager** | `cuMemAddressReserve` + `cuMemCreate` + `cuMemMap` — bypasses the CUDA Runtime entirely |
| **Zero-Copy Remapping** | Physical frames are relocated without invalidating existing virtual pointers |
| **Lock-Free Watchdog** | Detached `std::thread` using `std::atomic<bool>` for signal-safe shutdown |
| **Thread-Safe Allocator** | `std::lock_guard<std::mutex>` guards all allocation, free, and compaction paths |
| **Data Integrity Proof** | A sentinel value (`1337`) written to a virtual address before compaction is verified to survive physical migration |
| **Fragmentation Metric** | Real-time score: `S = 1 - (largest_contiguous_free / total_free)` — logged to CSV per simulation tick |
| **Telemetry Dashboard** | Python (`matplotlib` + `pandas`) auto-generates a fragmentation timeline graph from the CSV |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   T-GMA Engine                               │
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────────┐   │
│  │  Main Thread         │    │  Watchdog Thread          │   │
│  │  (Workload Sim)      │    │  (Telemetry Daemon)       │   │
│  │                      │    │                           │   │
│  │  allocate()          │    │  NVML Poll @ 1 Hz         │   │
│  │  free()  → Hole Gen  │    │  Temp ≥ Threshold?        │   │
│  │  log_memory_state()  │    │  └─→ defragment()         │   │
│  └──────────┬───────────┘    └──────────┬────────────────┘   │
│             │                           │                     │
│             └───────────┬───────────────┘                     │
│                         │  (mutex-guarded)                    │
│                         ▼                                     │
│           ┌─────────────────────────┐                        │
│           │   ThermalAllocator      │                        │
│           │   (VMM Core)            │                        │
│           │                         │                        │
│           │  1GB Virtual Hallway    │                        │
│           │  [slot0][slot1]...[N]   │                        │
│           │     ↑        ↑          │                        │
│           │  cuMemMap  cuMemMap     │                        │
│           │  (2MB phys)(2MB phys)   │                        │
│           │                         │                        │
│           │  defragment():          │                        │
│           │  cuMemcpyDtoD(hole←src) │                        │
│           │  Remap physical handle  │                        │
│           └─────────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
T-GMA/
├── ThermalAllocator.h      # Allocator interface: PageNode linked list, VMM handles
├── ThermalAllocator.cpp    # Core VMM engine: allocate, free, defragment, log
├── main.cpp                # Workload simulator + NVML watchdog thread
├── Makefile                # Build system (g++, CUDA Driver API, NVML)
├── plot_frag.py            # Python telemetry dashboard (matplotlib)
├── fragmentation_log.csv   # Live output: timestep, active pages, free pages, frag score
└── fragmentation_graph.png # Auto-generated compaction event graph
```

---

## Build & Run

### Prerequisites

- Linux with an NVIDIA GPU
- CUDA Toolkit (Driver API headers: `cuda.h`)
- NVML (`libnvidia-ml`)
- `g++` with C++11 support
- Python 3 with `matplotlib` and `pandas` (for the dashboard)

### Build

```bash
make
```

### Run

```bash
./allocator_engine
```

The engine will:
1. Boot the VMM allocator and reserve the 1 GB virtual address space.
2. Launch the NVML watchdog thread.
3. Allocate 5 tensor-sized (2 MB) memory pages and write a sentinel value.
4. Deliberately free alternating pages to induce external fragmentation.
5. Wait for the watchdog to detect a thermal threshold breach and trigger compaction.
6. Verify that the sentinel value survived physical migration.
7. Write all telemetry to `fragmentation_log.csv`.

### Generate the Dashboard

```bash
python3 plot_frag.py
```

Outputs `fragmentation_graph.png` — a plot of the fragmentation score over time, showing the compaction event as a sharp drop to zero.

---

## How the Compaction Works (Step-by-Step)

1. **Scan** — Walk the `PageNode` linked list to find the first free "hole" followed by the first active block.
2. **Copy** — Execute `cuMemcpyDtoD(hole->v_addr, active->v_addr, 2MB)` — a direct silicon-to-silicon transfer.
3. **Remap** — Transfer the physical `CUmemGenericAllocationHandle` from the source node to the hole node.
4. **Update** — Patch the `active_allocations` map so future `free()` calls resolve correctly.
5. **Verify** — After migration, the application reads back from its **original virtual address** and confirms data integrity.

The virtual pointer held by the application is never touched. This is the fundamental guarantee of VMM-based allocation.

---

## Why This Matters for Production ML

In production LLM training and inference:
- Activations, KV-caches, and optimizer states are allocated and freed constantly at different granularities.
- Over hours-long runs, fragmentation silently degrades effective VRAM utilization.
- A proactive, thermal-aware allocator can reclaim fragmented VRAM **without stopping the training loop**, increasing effective batch sizes and sustained throughput.

T-GMA is a proof-of-concept for exactly this class of system-level memory management.

---

## Technologies

`C++11` · `CUDA Driver API (VMM)` · `NVML` · `cuMemMap` · `cuMemCreate` · `cuMemcpyDtoD` · `std::thread` · `std::mutex` · `std::atomic` · `Python` · `matplotlib` · `pandas`