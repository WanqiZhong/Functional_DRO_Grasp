import os

root_directory = "/data/zwq/code/DRO-Grasp/data/data_urdf/object/oakink"  # Replace with the actual path

for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        if "_real" in filename or "_virtual" in filename:
            old_path = os.path.join(dirpath, filename)
            new_filename = filename.replace("_real", "").replace("_virtual", "")
            new_path = os.path.join(dirpath, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

print("Renaming complete.")