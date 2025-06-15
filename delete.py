import os
import shutil
import sys

def delete_if_exists(path):
    if os.path.isdir(path):
        print(f"Borrando carpeta: {path}")
        shutil.rmtree(path)
    elif os.path.isfile(path):
        print(f"Borrando archivo: {path}")
        os.remove(path)
    else:
        print(f"No se encontró: {path}")

def main(experiment):
    base = "data"
    paths_to_delete = [
        os.path.join(base, "masks", experiment),
        os.path.join(base, "plots", experiment),
        os.path.join(base, "processed", experiment),
        os.path.join(base, "processed", f"{experiment}_masks"),
        os.path.join(base, "raw", f"{experiment}.tif"),
        os.path.join(base, "results", experiment),
    ]

    for path in paths_to_delete:
        delete_if_exists(path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete.py <video_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    main(experiment_name)
