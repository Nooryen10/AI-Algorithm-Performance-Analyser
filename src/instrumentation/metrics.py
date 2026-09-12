"""
Metrics object passed into every algorithm to count comparisons and
swaps/shifts/probes as the algorithm runs. Time and memory are measured
separately by the @instrument decorator, since those wrap the whole call.
"""


class Metrics:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0        # also used for "shifts" in insertion sort etc.
        self.probes = 0       # used by searching algorithms

    def compare(self, n=1):
        self.comparisons += n

    def swap(self, n=1):
        self.swaps += n

    def probe(self, n=1):
        self.probes += n

    def as_dict(self):
        return {
            "comparisons": self.comparisons,
            "swaps": self.swaps,
            "probes": self.probes,
        }