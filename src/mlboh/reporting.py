import json


def save_report(
    report,
    filename,
):
    with open(
        filename,
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )
