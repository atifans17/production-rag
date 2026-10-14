import random
import sys

from src.loaders import load_directory

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
records = load_directory("data")
for r in random.sample(records, min(n, len(records))):
    m = r["metadata"]
    print("=" * 80)
    print(f"{m['source_file']}  ({m['section']})")
    print(r["text"][:700])