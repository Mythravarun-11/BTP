import polyscope as ps
import polyscope.imgui as psim
import trimesh
import argparse
from tqdm import tqdm
import glob
from natsort import natsorted
import itertools


def parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_paths", type=str, help="mesh paths", required=True)
    parser.add_argument("--reference_paths", type=str, help="mesh paths")
    args = parser.parse_args()
    return args

def main(): 
    args = parser()

    all_meshes = natsorted(glob.glob(f'{args.mesh_paths}/*.obj'))

    if args.reference_paths: 
        all_reference_meshes = natsorted(glob.glob(f'{args.reference_paths}/*.obj'))
        if len(all_meshes) > len(all_reference_meshes): 
            extended_reference_meshes = list(itertools.islice(itertools.cycle(all_reference_meshes), len(all_meshes)))
            all_reference_meshes = extended_reference_meshes  # Update with extended list
            assert len(all_reference_meshes) == len(extended_reference_meshes)

    # Debug: print the list of mesh paths
    print(f"Found {len(all_meshes)} mesh files.")
    
    if len(all_meshes) > 0: 
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_autoscale_structures(False)

        # State variables for play, pause, and reset
        play = False
        reset = False
        current_mesh_index = 0

        def load_mesh(index):
            """Function to load and display a mesh by index."""
            if index < len(all_meshes):
                mesh_path = all_meshes[index]
                print(f"Loading mesh: {mesh_path}")
                if args.reference_paths: 
                    reference_path = all_reference_meshes[index]
                    print(f"Loading reference mesh: {reference_path}")

                try:
                    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
                    ps.register_surface_mesh('mesh', mesh.vertices, mesh.faces)

                    if args.reference_paths: 
                        mesh = trimesh.load(reference_path, process=False, maintain_order=True)
                        ps.register_surface_mesh('reference_mesh', mesh.vertices, mesh.faces)

                    print(f"Displaying mesh {index + 1}/{len(all_meshes)}")
                except Exception as e:
                    print(f"Failed to load mesh {mesh_path}: {e}")

        # Register callback for GUI buttons
        def callback():
            nonlocal play, reset, current_mesh_index

            # ImGui button logic
            if psim.Button("Play"):
                play = True
                reset = False
                print("Play button clicked")

            if psim.Button("Pause"):
                play = False
                print("Pause button clicked")

            if psim.Button("Reset"):
                play = False
                reset = True
                current_mesh_index = 0
                ps.remove_all_structures()
                print("Reset button clicked")
                load_mesh(current_mesh_index)

            if play:
                if current_mesh_index < len(all_meshes):
                    load_mesh(current_mesh_index)
                    current_mesh_index += 1
                else:
                    play = False  # Stop when all meshes are displayed

        ps.set_user_callback(callback)

        # Initial display
        load_mesh(current_mesh_index)
        
        ps.show()  # Start the Polyscope interactive window

if __name__ == '__main__': 
    main()


