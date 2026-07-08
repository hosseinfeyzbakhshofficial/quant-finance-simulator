import os

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

IGNORE_FILES = {".DS_Store", "make_tree.py"}


def generate_tree(dir_path, prefix=""):
    try:
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        return

    entries = [e for e in entries if e not in IGNORE_DIRS and e not in IGNORE_FILES]

    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "

        print(f"{prefix}{connector}{entry}")

        if os.path.isdir(path):
            indent = "    " if is_last else "│   "
            generate_tree(path, prefix + indent)


if __name__ == "__main__":
    print(".")
    generate_tree(".")
