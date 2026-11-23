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
| index | m=16, ef_construction=200, ef_search=200, metric=l2, nlist=1024, nprobe=64, pq_m=16, pq_nbits=8, refine=False, mmap_invlists=False |
| queries | 10000 |
| k | 10 |
| build_s | 143.324 |
| index_mb | 1360.53 |
| recall_at_k | 0.7996 |
| pure_index_qps | 93.9 |
| refine_factor | 1 |
| pure_p50_ms | 10.224 |
| pure_p99_ms | 15.572 |
| full_row_qps | 93.8 |
| full_p50_ms | 10.228 |
| full_p99_ms | 15.624 |
| qps | 93.8 |
| p50_ms | 10.228 |
| p99_ms | 15.624 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-43-generic-x86_64-with-glibc2.41, python=3.12.15, containerized=True, gpus=[] |
