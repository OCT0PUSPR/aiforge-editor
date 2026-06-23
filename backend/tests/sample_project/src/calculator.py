"""A tiny calculator module used as RAG/index test fixture."""


def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def subtract(a, b):
    """Return the difference of two numbers."""
    return a - b


class Calculator:
    """Stateful calculator that accumulates a running total."""

    def __init__(self):
        self.total = 0

    def fibonacci(self, n):
        """Compute the nth Fibonacci number iteratively."""
        prev, curr = 0, 1
        for _ in range(n):
            prev, curr = curr, prev + curr
        return prev
