import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, Manager
from concurrent.futures import ThreadPoolExecutor


def coacd_worker(task):
    """
    Worker function for generating coacd_xxx.obj using coacd.
    Handles individual tasks in subprocess.
    """
    input_file, output_file, log_queue = task
    try:
        process = subprocess.run(
            ["python", "data_utils/subprocess_prepare_coacd.py", "-i", input_file, "-o", output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if process.returncode == 0:
            log_queue.put(f"Successfully generated {output_file}.")
        else:
            log_queue.put(f"Failed to generate {output_file}. Error: {process.stderr.strip()}")
    except Exception as e:
        log_queue.put(f"Error processing {input_file} -> {output_file}: {str(e)}")


def coacd_task_manager(input_dir, max_processes=50):
    """
    Manages coacd_xxx.obj generation tasks using a pool of subprocesses.
    Skips tasks if the output file already exists.
    Logs progress dynamically.
    """
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".obj") and not file.startswith("coacd_"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(root, f"coacd_{file}")
                # Skip task if coacd output file already exists
                if os.path.exists(output_file):
                    print(f"Skipping existing file: {output_file}")
                    continue
                tasks.append((input_file, output_file))

    # Using multiprocessing with a managed queue for logging
    with Manager() as manager:
        log_queue = manager.Queue()
        with Pool(processes=max_processes) as pool:
            # Enqueue tasks and monitor progress using tqdm
            results = pool.imap_unordered(coacd_worker, [(task[0], task[1], log_queue) for task in tasks])
            for _ in tqdm(results, total=len(tasks), desc="Generating COACD Files", unit="file"):
                while not log_queue.empty():
                    print(log_queue.get())  # Print log messages as they arrive


def urdf_task(obj_file, coacd_file, standard_urdf_path, modified_urdf_path):
    """
    Task for generating URDF files.
    """
    try:
        # Generate standard URDF
        standard_urdf_content = f"""<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{os.path.basename(obj_file)}" scale="1.00 1.00 1.00"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{os.path.basename(obj_file)}" scale="1.00 1.00 1.00"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        with open(standard_urdf_path, "w") as f:
            f.write(standard_urdf_content)

        # Generate modified URDF
        modified_urdf_content = f"""<?xml version="1.0"?>
<robot name="root">
  <link name="link_original">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{os.path.basename(obj_file)}" scale="1.0000E+00 1.0000E+00 1.0000E+00"/>
      </geometry>
    </visual>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="4.74E-05" ixy="-1.66E-07" ixz="1.03E-06" iyy="1.23E-04" iyz="-7.90E-09" izz="1.59E-04"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{os.path.basename(coacd_file)}" scale="1.0000E+00 1.0000E+00 1.0000E+00"/>
      </geometry>
    </collision>
  </link>
  <link name="link_original"/>
</robot>
"""
        with open(modified_urdf_path, "w") as f:
            f.write(modified_urdf_content)
        return f"Successfully generated URDF files for {obj_file}."
    except Exception as e:
        return f"Failed to generate URDF files for {obj_file}. Error: {e}"


def generate_urdf(input_dir):
    """
    Generate URDF files for all eligible .obj files.
    """
    urdf_tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".obj") and not file.startswith("coacd_"):
                obj_file = os.path.join(root, file)
                coacd_file = os.path.join(root, f"coacd_{file}")
                base_name = os.path.splitext(file)[0]
                standard_urdf_path = os.path.join(root, f"{base_name}.urdf")
                modified_urdf_path = os.path.join(root, f"coacd_decomposed_object_one_link_{base_name}.urdf")
                urdf_tasks.append((obj_file, coacd_file, standard_urdf_path, modified_urdf_path))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(lambda args: urdf_task(*args), urdf_tasks), total=len(urdf_tasks), desc="Generating URDFs"))
    for result in results:
        print(result)


def main():
    # Base directory to traverse
    input_directory = "/data/zwq/code/DRO-Grasp/data/data_urdf/object/oakink"

    # Step 1: Generate URDF files (fast step)
    # print("Starting URDF generation...")
    # generate_urdf(input_directory)

    # Step 2: Generate COACD files with efficient subprocess handling
    print("Starting COACD generation...")
    coacd_task_manager(input_directory, max_processes=20)


if __name__ == "__main__":
    main()