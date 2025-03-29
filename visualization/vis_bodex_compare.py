import os
import viser
import trimesh
import numpy as np

BASE_DIR = "/data/zwq/code/DexGraspBench/output/oakink_gold_eval_500_shadow/vis_obj/"
DRO_DIR = "/data/zwq/code/DexGraspBench/output/dro_eval_500_shadow/vis_obj/"
DIFFUSION_DIR = "/data/zwq/code/DexGraspBench/output/diffusion_eval_500_shadow/vis_obj/"

def main():
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    print("http://127.0.0.1:8080")
    
    object_folders = {}
    
    for object_name in os.listdir(BASE_DIR):
        object_path = os.path.join(BASE_DIR, object_name)
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
        
        try:
            grasp_path = os.path.join(BASE_DIR, pair_info['full_path'], pair_info['grasp_file'])
            grasp_mesh = trimesh.load_mesh(grasp_path)
            grasp_data = os.path.join(BASE_DIR.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
            print('grasp_path', grasp_data)
            grasp_data = np.load(grasp_data, allow_pickle=True).item()
            grasp_succ = grasp_data['succ_flag']
            
            server.scene.add_mesh_simple(
                'gold_mesh',
                grasp_mesh.vertices,
                grasp_mesh.faces,
                color=(102, 192, 255),  # Light blue
                opacity=0.7
            )

            dro_path = os.path.join(DRO_DIR, pair_info['full_path'], pair_info['grasp_file'])
            dro_mesh = trimesh.load_mesh(dro_path)
            dro_data = os.path.join(DRO_DIR.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
            dro_data = np.load(dro_data, allow_pickle=True).item()
            dro_succ = dro_data['succ_flag']
            server.scene.add_mesh_simple(
                'dro_mesh',
                dro_mesh.vertices,
                dro_mesh.faces,
                color=(192, 102, 255), 
                opacity=0.7
            )

            diffusion_path = os.path.join(DIFFUSION_DIR, pair_info['full_path'], pair_info['grasp_file'])
            diffusion_mesh = trimesh.load_mesh(diffusion_path)
            diffusion_data = os.path.join(DIFFUSION_DIR.split('vis_obj/')[0], 'evaluation', pair_info['full_path'], pair_info['grasp_file'].split('_grasp_0.obj')[0] + '.npy')
            diffusion_data = np.load(diffusion_data, allow_pickle=True).item()
            diffusion_succ = diffusion_data['succ_flag']
            server.scene.add_mesh_simple(
                'diffusion_mesh',
                diffusion_mesh.vertices,
                diffusion_mesh.faces,
                color=(192, 255, 102),
                opacity=0.7
            )

            server.scene.add_label(
                "Succ",
                f"Dataset: {grasp_succ} \n DRO: {dro_succ} \n Diffusion: {diffusion_succ}",
                wxyz=(1, 0, 0, 0),
                position=(1, 1, 1),
            )

        except Exception as e:
            print(f"Error loading grasp mesh: {e}")
        
        try:
            obj_path = os.path.join(BASE_DIR, pair_info['full_path'], pair_info['obj_file'])
            obj_mesh = trimesh.load_mesh(obj_path)
            server.scene.add_mesh_simple(
                'obj_mesh',
                obj_mesh.vertices,
                obj_mesh.faces,
                color=(239, 132, 167),  # Pink
                opacity=1.0
            )
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