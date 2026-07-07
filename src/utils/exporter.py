import json
from typing import Any


def export_results(results: Any, filename: str = "results.json") -> None:
    """
    Export simulation results to a formatted JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results exported to {filename}")


def save_report(report: Any, filename: str) -> None:
    """
    Save the simulation report data structure into a JSON file.
    Reuses export_results to eliminate code duplication.
    """
    export_results(results=report, filename=filename)
