import os
import benodb as bs
import numpy as np

db_path = os.environ.get("BENODB_TEST_PATH", "benchmark_db")
t = bs.open_table(db_path)
res = t.vector_search_scored("embedding", [0.5] * 128, 10)
print(f"Results length: {len(res)}")
