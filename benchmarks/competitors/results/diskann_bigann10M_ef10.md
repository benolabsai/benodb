# Competitor: diskann / vector_ann

| Metric | Value |
|---|---|
| engine | diskann |
| index_type | disk |
| index_precision | pq-128 |
| dataset | bigann-10M |
| workload | vector_ann |
| device | cpu |
| n | 10000000 |
| dim | 128 |
| index | m=32, ef_construction=64, ef_search=10, metric=l2, nlist=1024, nprobe=0, pq_m=16, pq_nbits=8, refine=False, mmap_invlists=False |
| queries | 10000 |
| k | 10 |
| build_s | 1261.73 |
| index_mb | 6584.0 |
| recall_at_k | 0.6991 |
| pure_index_qps | 535.3 |
| refine_factor | 1 |
| pure_p50_ms | 1.809 |
| pure_p99_ms | 3.28 |
| full_row_qps | 534.7 |
| full_p50_ms | 1.812 |
| full_p99_ms | 3.257 |
| qps | 534.7 |
| p50_ms | 1.812 |
| p99_ms | 3.257 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-43-generic-x86_64-with-glibc2.41, python=3.11.18, containerized=True, gpus=[] |
