class Benchmark:

    def __init__(self):
        self.results = []

    def add_result(self, strategy, metrics):
        self.results.append({
            "strategy": strategy,
            **metrics
        })

    def get_results(self):
        return self.results