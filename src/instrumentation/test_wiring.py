from .decorators import instrument

@instrument
def dummy_sort(arr, metrics):
    metrics.compare()
    metrics.swap()
    return sorted(arr)

if __name__ == "__main__":
    output = dummy_sort([3, 1, 2])
    print(output)