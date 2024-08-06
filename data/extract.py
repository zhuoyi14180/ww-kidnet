import os
import sys

current_file_path = os.path.abspath(__file__)

project_root = os.path.dirname(os.path.dirname(current_file_path))

sys.path.append(project_root)
from config import PediatricConfig, AdultConfig

def extract_tree(root):
    leaves = []

    for cur, dirs, _ in os.walk(root):
        if not dirs:
            leaves.append(os.path.relpath(cur, root))

    return leaves


def save_tree_to_file(tree, file_path):
    with open(file_path, 'w') as f:
        for line in tree:
            f.write(f"{line}\n")


if __name__ == "__main__":
    train = PediatricConfig().BRATS_VALID
    save_tree_to_file(extract_tree(train["dir"]), os.path.join(train["dir"], train["list"]))