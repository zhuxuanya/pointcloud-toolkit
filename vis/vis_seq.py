import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def get_point_cloud_files(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    return [os.path.join(folder_path, f) for f in files]

def load_point_cloud(file_path, remove_intensity=True):
    points = np.load(file_path)
    pcd = o3d.geometry.PointCloud()
    if not remove_intensity and points.shape[1] == 4:
        # use the fourth column as intensity values
        colors = plt.get_cmap("terrain")(points[:, 3] / np.max(points[:, 3]))[:, :3]  # normalize and convert to RGB
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    return pcd

def visualize_point_clouds(file_paths, remove_intensity):
    if not file_paths:
        print("No point clouds found in the directory.")
        return
    
    print(f"Total number of files to visualize: {len(file_paths)}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    idx = 0
    current_pcd = load_point_cloud(file_paths[idx], remove_intensity)
    vis.add_geometry(current_pcd)
    print(f"Visualizing: {os.path.basename(file_paths[idx])}")

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    def load_next_point_cloud(vis):
        nonlocal idx, current_pcd, camera_params
        if idx < len(file_paths) - 1:
            idx += 1
            current_pcd = load_point_cloud(file_paths[idx], remove_intensity)
            vis.clear_geometries()
            vis.add_geometry(current_pcd)
            view_control.convert_from_pinhole_camera_parameters(camera_params)
            print(f"Visualizing: {os.path.basename(file_paths[idx])}")
        else:
            vis.destroy_window()  # signal to close the visualizer

    def load_previous_point_cloud(vis):
        nonlocal idx, current_pcd, camera_params
        if idx > 0:
            idx -= 1
            current_pcd = load_point_cloud(file_paths[idx], remove_intensity)
            vis.clear_geometries()
            vis.add_geometry(current_pcd)
            view_control.convert_from_pinhole_camera_parameters(camera_params)
            print(f"Visualizing: {os.path.basename(file_paths[idx])}")

    def save_camera_params(vis):
        nonlocal camera_params
        camera_params = view_control.convert_to_pinhole_camera_parameters()

    vis.register_key_callback(262, load_next_point_cloud)  # right arrow key
    vis.register_key_callback(263, load_previous_point_cloud)  # left arrow key
    vis.register_animation_callback(save_camera_params)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    vis.run()

def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud data")
    parser.add_argument("--points_path", type=str, help="specify the point cloud data directory")
    parser.add_argument("--remove_intensity", action="store_true", help="draw point cloud data without intensity")
    args = parser.parse_args()

    file_paths = get_point_cloud_files(args.points_path)
    visualize_point_clouds(file_paths, args.remove_intensity)

if __name__ == "__main__":
    main()
