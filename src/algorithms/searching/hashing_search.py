from ...instrumentation.decorators import instrument


class SimpleHashTable:
    def __init__(self, size):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.collision_count = 0
        self.num_items = 0

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, index):
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]
        if len(bucket) > 0:
            self.collision_count += 1
        bucket.append((key, index))
        self.num_items += 1

    def lookup(self, key, metrics):
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]
        for k, idx in bucket:
            metrics.probe()
            if k == key:
                return idx
        return -1

    def load_factor(self):
        return self.num_items / self.size


@instrument
def hashing_search(arr, metrics, target):
    table_size = max(1, len(arr) // 2)  # deliberately smaller to force some collisions
    table = SimpleHashTable(table_size)

    for i, val in enumerate(arr):
        table.insert(val, i)

    found_index = table.lookup(target, metrics)

    return {
        "found": found_index != -1,
        "index": found_index,
        "load_factor": round(table.load_factor(), 4),
        "collision_count": table.collision_count,
    }