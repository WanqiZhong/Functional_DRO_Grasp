import os
import viser
import trimesh
import numpy as np

DIR_1 = "/data/zwq/code/DexGraspBench/output/dro_eval_500_early_version_shadow_custom/vis_obj/"
DIR_2 = "/data/zwq/code/DexGraspBench/output/dro_eval_500_early_version_shadow_multiple/vis_obj/"
DIR_3 = "/data/zwq/code/DexGraspBench/output/dro_eval_500_shadow_custom/vis_obj/"
DIR_4 = "/data/zwq/code/DexGraspBench/output/dro_eval_500_shadow_multiple/vis_obj/"
DIR_5 = "/data/zwq/code/DexGraspBench/output/diffusion_eval_500_shadow_custom/vis_obj/"
DIR_6 = "/data/zwq/code/DexGraspBench/output/diffusion_eval_500_shadow_multiple/vis_obj/"

# Set comparison items directly in the code
item1 = "DRO_Early_Custom"
item2 = "DRO_Early_Multiple"
item3 = "DRO_Custom"
item4 = "DRO_Multiple"
item5 = "Diffusion_Custom"
item6 = "Diffusion_Multiple"

TMP_DIR = "/data/zwq/code/Visualize/"
copy_flag = True
# Copy the select data file (.npy) to new folder, named as the item name
def copy_select_file(object_name, src_file, item_name, suffix=".npy"):
    dst_dir = os.path.join(TMP_DIR, object_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_file = os.path.join(dst_dir, item_name + suffix)
    if os.path.exists(dst_file):
        os.remove(dst_file)
    os.system(f"cp {src_file} {dst_file}")

def main():
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    print("http://127.0.0.1:8080")
    object_folders = {}
    
    for object_name in os.listdir(DIR_1):
        object_path = os.path.join(DIR_1, object_name)
        if not os.path.isdir(object_path):
            continue
            
        object_folders[object_name] = {}
        
        for object_id in os.listdir(object_path):
            object_id_path = os.path.join(object_path, object_id)
            if not os.path.isdir(object_id_path):
                continue
                
            object_folders[object_name][object_id] = {}
            
            grasp_files = [f for f in os.listdir(object_id_path) if f.endswith('_grasp_0.obj')]
            obj_files = [f for f in os.listdir(object_id_path) if f.endswith('_obj.obj')]
            
            for grasp_file in grasp_files:
                index = grasp_file.split('_')[0]
                obj_file = f"{index}_obj.obj"
                
                if obj_file in obj_files:
                    full_path = os.path.join(object_name, object_id)
                    pair_info = {
                        'object_name': object_name,
                        'object_id': object_id,
                        'grasp_file': grasp_file,
                        'obj_file': obj_file,
                        'full_path': full_path
                    }
                    object_folders[object_name][object_id][index] = pair_info
    
    object_count = len(object_folders)
    print(f"Found {object_count} objects")
    
    current_object_name = list(object_folders.keys())[0] if object_folders else ""
    current_object_id = ""
    current_pair_number = ""
    
    if current_object_name and object_folders[current_object_name]:
        current_object_id = list(object_folders[current_object_name].keys())[0]
        
    if current_object_name and current_object_id and object_folders[current_object_name][current_object_id]:
        current_pair_number = list(object_folders[current_object_name][current_object_id].keys())[0]
    
    def update_visualization():
        server.scene.reset()
        
        if not current_object_name or not current_object_id or not current_pair_number:
            print("Invalid selection")
            return
        
        try:
            pair_info = object_folders[current_object_name][current_object_id][current_pair_number]
        except KeyError:
            print(f"Cannot find: {current_object_name}/{current_object_id}/{current_pair_number}")
            return
            
        gui_info.value = f"Object: {current_object_name}\nID: {current_object_id}\nNumber: {current_pair_number}"
        
        success_text = ""
        
        if item1:
            try:
                path_1 = os.path.join(DIR_1, pair_info['full_path'], pair_info['grasp_file'])
                mesh_1 = trimesh.load_mesh(path_1)
                data_1_path = os.path.join(DIR_1.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
                print(f"Loaded {item1} data: {data_1_path}")
                data_1 = np.load(data_1_path, allow_pickle=True).item()
                succ_1 = data_1['succ_flag']
                server.scene.add_mesh_simple(
                    f'{item1}_mesh',
                    mesh_1.vertices,
                    mesh_1.faces,
                    color=(192, 102, 255), 
                    opacity=0.7
                )
                success_text += f"{item1}: {succ_1}\n"
            except Exception as e:
                print(f"Error loading {item1} mesh: {e}")

        # Load and display Diffusion mesh
        if item2:
            try:
                path_2 = os.path.join(DIR_2, pair_info['full_path'], pair_info['grasp_file'])
                mesh_2 = trimesh.load_mesh(path_2)
                data_2_path = os.path.join(DIR_2.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
                print(f"Loaded {item2} data: {data_2_path}")
                data_2 = np.load(data_2_path, allow_pickle=True).item()
                succ_2 = data_2['succ_flag']
                server.scene.add_mesh_simple(
                    f'{item2}_mesh',
                    mesh_2.vertices,
                    mesh_2.faces,
                    color=(192, 255, 102),
                    opacity=0.7
                )
                success_text += f"{item2}: {succ_2}\n"
            except Exception as e:
                print(f"Error loading {item2} mesh: {e}")
        
        if item3:
            try:
                path_3 = os.path.join(DIR_3, pair_info['full_path'], pair_info['grasp_file'])
                mesh_3 = trimesh.load_mesh(path_3)
                data_3_path = os.path.join(DIR_3.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
                print(f"Loaded {item3} data: {data_3_path}")
                data_3 = np.load(data_3_path, allow_pickle=True).item()
                succ_3 = data_3['succ_flag']
                server.scene.add_mesh_simple(
                    f'{item3}_mesh',
                    mesh_3.vertices,
                    mesh_3.faces,
                    color=(102, 255, 192),
                    opacity=0.7
                )
                success_text += f"{item3}: {succ_3}\n"
            except Exception as e:
                print(f"Error loading {item3} mesh: {e}")


        if item4:   
            try:
                path_4 = os.path.join(DIR_4, pair_info['full_path'], pair_info['grasp_file'])
                mesh_4 = trimesh.load_mesh(path_4)
                data_4_path = os.path.join(DIR_4.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
                print(f"Loaded {item4} data: {data_4_path}")
                data_4 = np.load(data_4_path, allow_pickle=True).item()
                succ_4 = data_4['succ_flag']
                server.scene.add_mesh_simple(
                    f'{item4}_mesh',
                    mesh_4.vertices,
                    mesh_4.faces,
                    color=(255, 192, 102),
                    opacity=0.7
                )
                success_text += f"{item4}: {succ_4}\n"
            except Exception as e:
                print(f"Error loading {item4} mesh: {e}")


        if item5:
            try:
                path_5 = os.path.join(DIR_5, pair_info['full_path'], pair_info['grasp_file'])
                mesh_5 = trimesh.load_mesh(path_5)
                data_5_path = os.path.join(DIR_5.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
                print(f"Loaded {item5} data: {data_5_path}")
                data_5 = np.load(data_5_path, allow_pickle=True).item()
                succ_5 = data_5['succ_flag']
                server.scene.add_mesh_simple(
                    f'{item5}_mesh',
                    mesh_5.vertices,
                    mesh_5.faces,
                    color=(255, 102, 192),
                    opacity=0.7
                )
                success_text += f"{item5}: {succ_5}\n"
            except Exception as e:
                print(f"Error loading {item5} mesh: {e}")

        if item6:
            try:
                path_6 = os.path.join(DIR_6, pair_info['full_path'], pair_info['grasp_file'])
                mesh_6 = trimesh.load_mesh(path_6)
                data_6_path = os.path.join(DIR_6.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
                print(f"Loaded {item6} data: {data_6_path}")
                data_6 = np.load(data_6_path, allow_pickle=True).item()
                succ_6 = data_6['succ_flag']
                server.scene.add_mesh_simple(
                    f'{item6}_mesh',
                    mesh_6.vertices,
                    mesh_6.faces,
                    color=(102, 192, 255),
                    opacity=0.7
                )
                success_text += f"{item6}: {succ_6}\n"
            except Exception as e:
                print(f"Error loading {item6} mesh: {e}")

        # Display success information
        if success_text:
            server.scene.add_label(
                "Succ",
                success_text.strip(),
                wxyz=(1, 0, 0, 0),
                position=(1, 1, 1),
            )


        if copy_flag:
            copy_select_file(f"{current_object_name}_{current_object_id}", data_1_path, f"{item1}_{succ_1}")
            copy_select_file(f"{current_object_name}_{current_object_id}", data_2_path, f"{item2}_{succ_2}")
            copy_select_file(f"{current_object_name}_{current_object_id}", data_3_path, f"{item3}_{succ_3}")
            copy_select_file(f"{current_object_name}_{current_object_id}", data_4_path, f"{item4}_{succ_4}")
            copy_select_file(f"{current_object_name}_{current_object_id}", data_5_path, f"{item5}_{succ_5}")
            copy_select_file(f"{current_object_name}_{current_object_id}", data_6_path, f"{item6}_{succ_6}")
            

        # Load and display object mesh
        try:
            obj_path = os.path.join(DIR_1, pair_info['full_path'], pair_info['obj_file'])
            obj_mesh = trimesh.load_mesh(obj_path)
            server.scene.add_mesh_simple(
                'obj_mesh',
                obj_mesh.vertices,
                obj_mesh.faces,
                color=(239, 132, 167),  # Pink
                opacity=1.0
            )
            copy_select_file(f"{current_object_name}_{current_object_id}", obj_path, "object", ".obj")
            print(f"Successfully loaded object mesh: {obj_path}")
        except Exception as e:
            print(f"Error loading object mesh: {e}")

    
    with server.gui.add_folder("Dataset"):
        gui_info = server.gui.add_text("Info", initial_value="")
        
        object_names = list(object_folders.keys())
        object_names.sort()
        
        def on_object_name_changed(name):
            name = name.target.value

            nonlocal current_object_name, current_object_id, current_pair_number
            
            current_object_name = name
            
            object_ids = list(object_folders[name].keys())
            object_ids.sort()
            object_id_dropdown.options = object_ids
            
            if object_ids:
                current_object_id = object_ids[0]
                object_id_dropdown.value = current_object_id
                
                update_pair_options()
            else:
                current_object_id = ""
                current_pair_number = ""
        
        object_name_dropdown = server.gui.add_dropdown(
            "Object Name",
            options=object_names,
            initial_value=current_object_name
        )
        object_name_dropdown.on_update(on_object_name_changed)
        
        def on_object_id_changed(id_value):
            id_value = id_value.target.value

            nonlocal current_object_id, current_pair_number
            
            current_object_id = id_value
            
            update_pair_options()
        
        initial_object_ids = list(object_folders[current_object_name].keys()) if current_object_name else []
        initial_object_ids.sort()
        
        object_id_dropdown = server.gui.add_dropdown(
            "Object ID",
            options=initial_object_ids,
            initial_value=current_object_id
        )
        object_id_dropdown.on_update(on_object_id_changed)
        
        def update_pair_options():
            nonlocal current_pair_number
            
            if not current_object_name or not current_object_id:
                return
                
            pair_numbers = list(object_folders[current_object_name][current_object_id].keys())
            pair_numbers.sort(key=int)  
            
            if pair_numbers:
                pair_slider.min = 0
                pair_slider.max = len(pair_numbers) - 1
                pair_slider.value = 0
                
                current_pair_number = pair_numbers[0]
                
                pair_label.value = f"Number: {current_pair_number} / {len(pair_numbers)}"
                
                update_visualization()
            else:
                current_pair_number = ""
                pair_label.value = "No available number"
        
        pair_label = server.gui.add_text("Number", initial_value="")
        
        def on_pair_slider_changed(slider_value):
            nonlocal current_pair_number
            
            if not current_object_name or not current_object_id:
                return
                
            pair_numbers = list(object_folders[current_object_name][current_object_id].keys())
            pair_numbers.sort(key=int)
            
            if pair_numbers and 0 <= slider_value < len(pair_numbers):
                current_pair_number = pair_numbers[int(slider_value)]
                
                pair_label.value = f"Number: {current_pair_number} / {len(pair_numbers)}"
                
                update_visualization()
        
        pair_slider = server.gui.add_slider(
            "Browse Number",
            min=0,
            max=20,
            step=1,
            initial_value=0
        )
        pair_slider.on_update(on_pair_slider_changed)
        
        def on_prev_clicked(event=None):
            if pair_slider.value > pair_slider.min:
                pair_slider.value = pair_slider.value - 1
                on_pair_slider_changed(pair_slider.value)
        
        def on_next_clicked(event=None):
            if pair_slider.value < pair_slider.max:
                pair_slider.value = pair_slider.value + 1
                on_pair_slider_changed(pair_slider.value)
        
        prev_button = server.gui.add_button("Previous")
        prev_button.on_click(on_prev_clicked)
        
        next_button = server.gui.add_button("Next")
        next_button.on_click(on_next_clicked)
    
    if current_object_name and current_object_id:
        update_pair_options()
    
    print("Server is running. Press Ctrl+C to exit.")
    try:
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Server is closing...")
        server.close()

if __name__ == "__main__":
    main()