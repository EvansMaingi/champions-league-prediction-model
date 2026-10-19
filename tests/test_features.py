"""Tests for src/features/build_features.py.

Planned (Milestones 4-5) - the most important tests in the project (leakage):
- `result` is "H" / "D" / "A" correctly derived from goals
- a match's own score never affects its own rolling features (shift works)
- rolling features use only earlier matches, even when input is unsorted
- teams with little history are handled explicitly, not silently filled
- `result` is not among the feature columns
"""
