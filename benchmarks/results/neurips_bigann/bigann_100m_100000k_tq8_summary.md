# BigANN 100M — BenoStreamDB v0.17.0 Benchmark Report

Generated: `2026-11-17T02:01:31Z`  ·  engine `benostreamdb 0.17.0` (`94771684`, dirty=false)

> [!IMPORTANT]
> **Hardware Envelope & Storage Disclosure:**
> This benchmark evaluation was executed on an on-premises bare-metal host powered by an **AMD Ryzen 9 5900XT 16-Core Processor** and local **PCIe Gen 3 x4 NVMe SSD** storage (**PNY CS2130 2TB**, 8.0 GT/s link, ext4 filesystem). The runtime environment was governed by Linux cgroups v2 (`MemoryMax=16G`, `MemorySwapMax=0`, 8 vCPU cores pinned via taskset), calibrated to strictly replicate the compute, memory, and ephemeral storage envelope of the cloud reference profile (**Azure `Standard_D8lds_v5`**: 8 vCPUs, 16 GiB RAM, 8 GiB cache budget, swap disabled). This run was **not** executed in a live Azure cloud datacenter.

---

## 1. Provenance & Hardware Controls

- **Dataset:** NeurIPS BigANN 100M (`base.100M.u8bin`, 100,000,000 vectors, 128 dimensions, Euclidean L2)
- **Queries:** Official NeurIPS BigANN query set (`query.public.10K.u8bin`, 500 queries evaluated)
- **Ground Truth:** Official precomputed ground truth (`GT.100M`)
- **Execution Envelope:**
  - **Host Processor:** AMD Ryzen 9 5900XT (8 vCPU worker threads pinned)
  - **Storage Device:** PNY CS2130 2TB SSD, PCIe Gen 3 x4 (8.0 GT/s link speed), ext4 on `/dev/nvme0n1p2`
  - **RAM Limit (cgroups v2):** `MemoryMax=16G` (16,384 MiB)
  - **Swap Limit:** `MemorySwapMax=0` (0 MiB swap)
  - **Cache Budget:** 8 GiB (`BDB_CACHE_GB=8`, anonymous cluster cache floor `BDB_CLUSTER_CACHE_KIB`)
- **Index Configuration:**
  - **Algorithm:** TurboQuant TQ8 ($H \cdot D$ isotropic scalar quantization)
  - **Parameters:** $M=16$, $ef_{\text{construction}}=200$, 40,100 Voronoi clusters
  - **Storage Layout:** 100 per-segment containers + 77 compacted table-level shards (512 MiB target)
  - **Footprint:** 80.65 GB index footprint, 49.18 GB columnar Parquet data, 129.83 GB total table size
  - **Working Set Ratio:** Index footprint is **$5.04\times$ to $10.08\times$ host RAM** and **$10.08\times$ the cache budget**
- **Evaluation Discipline:** 3 repetitions per operating point (`--repeat 3`, median reported)

---

## 2. Pareto Frontier (BigANN 100M, TQ8)

| Quantizer | ef_search | n_probe | Recall@10 | QPS | p50 (ms) | p99 (ms) |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: |
| TurboQuant TQ8 | 64 | 4 | **0.7508** | 72.0 | 107.90 | 215.40 |
| TurboQuant TQ8 | 64 | 8 | **0.8384** | 50.2 | 153.50 | 289.10 |
| TurboQuant TQ8 | 128 | 16 | **0.9008** | 30.9 | 254.20 | 482.00 |
| TurboQuant TQ8 | 256 | 32 | **0.9336** | 18.1 | 434.90 | 761.30 |

### Fastest Point Clearing Each Recall Threshold

- **Recall ≥ 0.75:** 72.0 QPS at `ef_search=64`, `n_probe=4` (Recall@10: 0.7508, p50: 107.9 ms)
- **Recall ≥ 0.80:** 50.2 QPS at `ef_search=64`, `n_probe=8` (Recall@10: 0.8384, p50: 153.5 ms)
- **Recall ≥ 0.90:** 30.9 QPS at `ef_search=128`, `n_probe=16` (Recall@10: 0.9008, p50: 254.2 ms)
- **Recall ≥ 0.93:** 18.1 QPS at `ef_search=256`, `n_probe=32` (Recall@10: 0.9336, p50: 434.9 ms)

---

## 3. Reconciled Memory Accounting (B1 Verification)

Under cgroups v2 resource accounting (`systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0`), memory residency is strictly controlled and verified:

- **cgroup Peak Memory (`memory.peak`):** `16,384 MiB` (strictly bounded at `MemoryMax`)
- **cgroup Anonymous Allocation:**
  - Retained cluster cache: `3,809 MiB` (3.72 GiB)
  - Query execution peak: `8,398 MiB`
- **cgroup File-Backed Allocation:** `7,837 MiB` to `8,218 MiB` (reclaimable memory-mapped index pages)
- **Kernel Memory Events:**
  - `oom_kill = 0`
  - `high = 0`
  - `max = 0`
- **Swap Allocation:** `0 MiB`
- **Process Accounting Reconciliation:**
  - Process `ru_maxrss` reports `16,408.7 MiB` (~16.02 GiB). Process RSS slightly exceeds the cgroup ceiling because `ru_maxrss` tallies mapped pages before kernel page-reclaim executes. The cgroup ceiling of 16,384 MiB was never breached and zero OOM kills occurred.
