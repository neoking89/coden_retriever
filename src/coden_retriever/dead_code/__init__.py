"""Dead code detection analysis.

Identifies functions and methods with no incoming calls in the call graph.
"""

from .detector import calculate_confidence, detect_unused_functions

__all__ = ["detect_unused_functions", "calculate_confidence"]
