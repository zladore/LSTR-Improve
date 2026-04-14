cd /home/hbxz_lzl/LSTR-Trimodal

python3 - <<'PY' > structure_trimmed.txt
import os

root = "/home/hbxz_lzl/LSTR-Trimodal"

special_dirs = {
    os.path.join(root, "data/THUMOS/flow_kinetics_bninception"),
    os.path.join(root, "data/THUMOS/rgb_kinetics_resnet50"),
    os.path.join(root, "data/THUMOS/target_perframe"),
}

ignore_names = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}

ignore_suffixes = (
    ".pyc", ".pyo", ".swp", ".swo",
)

def should_ignore(name):
    if name in ignore_names:
        return True
    for suf in ignore_suffixes:
        if name.endswith(suf):
            return True
    return False

def walk(path, indent=""):
    name = os.path.basename(path)
    if path == root:
        print("LSTR-Trimodal/")
    else:
        print(f"{indent}{name}/" if os.path.isdir(path) else f"{indent}{name}")

    if not os.path.isdir(path):
        return

    if path in special_dirs:
        try:
            items = sorted([x for x in os.listdir(path) if not should_ignore(x)])
        except Exception as e:
            print(indent + "    " + f"[ERROR] {e}")
            return

        for item in items[:10]:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(indent + "    " + item + "/")
            else:
                print(indent + "    " + item)

        if len(items) > 10:
            print(indent + "    ...")
        return

    try:
        items = sorted([x for x in os.listdir(path) if not should_ignore(x)])
    except Exception as e:
        print(indent + "    " + f"[ERROR] {e}")
        return

    for item in items:
        walk(os.path.join(path, item), indent + "    ")

walk(root)
PY

cat structure_trimmed.txt
