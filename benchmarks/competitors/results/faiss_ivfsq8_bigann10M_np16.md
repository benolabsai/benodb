# Competitor: faiss_ivfsq8 / vector_ann

| Metric | Value |
|---|---|
| engine | faiss_ivfsq8 |
| index_type | ivf_sq8 |
| index_precision | sq |
| dataset | bigann-10M |
| workload | vector_ann |
| device | cpu |
| n | 10000000 |
| dim | 128 |
| index | m=16, ef_construction=200, ef_search=200, metric=l2, nlist=1024, nprobe=16, pq_m=16, pq_nbits=8, refine=False, mmap_invlists=False |
| queries | 10000 |
| k | 10 |
| build_s | 165.228 |
| index_mb | 1360.53 |
| recall_at_k | 0.7923 |
| pure_index_qps | 345.5 |
| refine_factor | 1 |
| pure_p50_ms | 2.689 |
| pure_p99_ms | 5.372 |
| full_row_qps | 344.5 |
| full_p50_ms | 2.699 |
| full_p99_ms | 5.381 |
| qps | 344.5 |
| p50_ms | 2.699 |
| p99_ms | 5.381 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
