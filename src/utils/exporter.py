import json


def export_results(data, filename="results.json"):
    """
    Export simulation results to JSON file.
    """

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results exported to {filename}")