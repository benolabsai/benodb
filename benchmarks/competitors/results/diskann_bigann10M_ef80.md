# Competitor: diskann / vector_ann

| Metric | Value |
|---|---|
| engine | diskann |
| index_type | disk |
| index_precision | sq |
| dataset | bigann-10M |
| workload | vector_ann |
| device | cpu |
| n | 10000000 |
| dim | 128 |
| index | m=32, ef_construction=64, ef_search=80, metric=l2 |
| queries | 10000 |
| k | 10 |
| build_s | 1261.73 |
| index_mb | 6584.0 |
| recall_at_k | 0.7934 |
| pure_index_qps | 274.9 |
| refine_factor | 1 |
| pure_p50_ms | 3.591 |
| pure_p99_ms | 4.875 |
| full_row_qps | 274.5 |
| full_p50_ms | 3.594 |
| full_p99_ms | 4.871 |
| qps | 274.5 |
| p50_ms | 3.594 |
| p99_ms | 4.871 |
| env | cpu_model=AMD Ryzen 9 5900XT 16-Core Processor, cores=8.0, ram_gb=16.0, host_cores=32, host_ram_gb=121.4, os=Linux-7.0.0-42-generic-x86_64-with-glibc2.41, python=3.11.18, containerized=True, gpus=[] |
