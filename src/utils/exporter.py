import json


def export_results(results, filename="results.json"):
    """
    Export simulation results to JSON file.
    """

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results exported to {filename}")