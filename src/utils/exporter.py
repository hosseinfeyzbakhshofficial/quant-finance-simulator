import json
from typing import Any

def export_results(results: Any, filename: str = "results.json") -> None:
    """
    Export simulation results to a formatted JSON file or CSV file based on input/extension.
    """
    # Detect if output format is CSV or if results is a pandas DataFrame
    if filename.endswith(".csv") or hasattr(results, "to_csv"):
        if hasattr(results, "to_csv"):
            results.to_csv(filename, index=False)
        else:
            import pandas as pd
            pd.DataFrame(results).to_csv(filename, index=False)
        print(f"Results exported to CSV: {filename}")
    else:
        # Default behavior for standard structures and JSON format
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Results exported to JSON: {filename}")


def save_report(report: Any, filename: str) -> None:
    """
    Save the simulation report data structure into a JSON file.
    Reuses export_results to eliminate code duplication.
    """
    export_results(results=report, filename=filename)