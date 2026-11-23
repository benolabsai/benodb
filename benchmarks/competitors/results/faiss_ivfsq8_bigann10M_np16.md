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
| build_s | 142.734 |
| index_mb | 1360.53 |
| recall_at_k | 0.9389 |
| pure_index_qps | 344.0 |
| refine_factor | 1 |
| pure_p50_ms | 2.718 |
| pure_p99_ms | 5.455 |
| full_row_qps | 342.7 |
| full_p50_ms | 2.73 |
| full_p99_ms | 5.474 |
| qps | 342.7 |
| p50_ms | 2.73 |
| p99_ms | 5.474 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-43-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
