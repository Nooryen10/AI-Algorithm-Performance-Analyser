"""
Computes dataset-level features used as ML model inputs:
- presortedness_score: how close to fully sorted the array already is
  (1.0 = perfectly sorted, 0.0 = perfectly reverse-sorted)
- duplicate_ratio: fraction of elements that are duplicates of another
- value_range: max - min of the array
"""


def presortedness_score(arr):
    """
    Computed as 1 - (normalized inversion count).
    An inversion is a pair (i, j) where i < j but arr[i] > arr[j].
    Fully sorted -> 0 inversions -> score 1.0
    Fully reverse-sorted -> max inversions -> score 0.0

    Uses a merge-sort-based O(n log n) inversion count so this stays fast
    even at size 2000, rather than the O(n^2) naive pairwise check.
    """
    n = len(arr)
    if n <= 1:
        return 1.0

    max_inversions = n * (n - 1) / 2

    def merge_count(a):
        if len(a) <= 1:
            return a, 0
        mid = len(a) // 2
        left, inv_left = merge_count(a[:mid])
        right, inv_right = merge_count(a[mid:])

        merged = []
        i = j = inv_split = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inv_split += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])

        return merged, inv_left + inv_right + inv_split

    _, inversions = merge_count(arr)
    return 1.0 - (inversions / max_inversions)


def duplicate_ratio(arr):
    """Fraction of elements that are NOT unique (i.e., repeated at least once)."""
    n = len(arr)
    if n == 0:
        return 0.0
    unique_count = len(set(arr))
    duplicate_count = n - unique_count
    return duplicate_count / n


def value_range(arr):
    if not arr:
        return 0
    return max(arr) - min(arr)


def extract_sorting_features(arr):
    return {
        "presortedness_score": round(presortedness_score(arr), 4),
        "duplicate_ratio": round(duplicate_ratio(arr), 4),
        "value_range": value_range(arr),
    }