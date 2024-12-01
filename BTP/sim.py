import trimesh
import os
import cv2
import numpy as np

def visualize_obj_sequence_to_video(directory, output_video, frame_rate=24, resolution=(640, 480)):
    # Get a list of .obj files sorted in order
    obj_files = sorted([f for f in os.listdir(directory) if f.endswith('.obj')])

    # Prepare the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, resolution)

    for obj_file in obj_files:
        # Load the mesh
        mesh = trimesh.load(os.path.join(directory, obj_file))
        
        # Render the mesh to an image
        image_data = mesh.scene().save_image(resolution=resolution)
        
        # Convert the image data to a numpy array
        img = np.frombuffer(image_data, dtype=np.uint8)

        try:
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # Decode it to OpenCV format
        except Exception as e:
            print(f"Error decoding image for {obj_file}: {e}")
            continue

        # Resize the image to match the desired resolution if necessary
        img = cv2.resize(img, resolution)

        # Write the frame to the video
        out.write(img)

    # Release the video writer
    out.release()
    print(f"Video saved at {output_video}")

# Directory where your .obj files are stored
output_dir = './outputs/gpbd_exps/check'

# Output video file path
output_video = './output_video.mp4'

# Call the function to create the video
visualize_obj_sequence_to_video(output_dir, output_video)
