# pointcloud-toolkit

The toolkit provides scripts for exploratory data analysis, converting ROS bag files, and visualizing point cloud data.

## Data Structures

The scripts require that data be organized as follows within a directory (e.g., `custom`). 

```
├── custom
│   ├── points
│   │   └── 001.npy
│   ├── labels
│   │   └── 001.txt
```

The `points` folder should contain the point cloud data files in `.npy` format. Each line within these files represents a point with the following format: `x y z intensity`. Here:

- `x, y, z` are the spatial coordinates,
- `intensity` represents the reflectivity strength of the surface.

The `labels` folder should contain the label data files in `.txt` format. Each line within these files defines a bounding box with the following format: `x y z dx dy dz yaw track_id type`. Here:
- `x, y, z` are the centroid coordinates of the bounding box,
- `dx, dy, dz` denote the dimensions of the box (length, width, height),
- `yaw` indicates the rotation around the vertical axis,
- `track_id` is a unique ID used for tracking objects across frames,
- `type` describes the class of the object.

Each point cloud and its corresponding label from each frame must share the same base filename, differing only in their file extension. This ensures that each point cloud data file is correctly associated with its corresponding label data.

## ROS Conversion

Use `rosbag2npy.py` to convert ROS bag files into NumPy arrays.

```bash
python rosbag2npy.py --bag_path ${rosbag_path} --output_path ${output_path} --lidar_topic ${lidar_topic}
```

- `--bag_path`: Path to the ROS bag file.
- `--output_path`: Directory where the output `.npy` files should be saved.
- `--lidar_topic`: ROS topic name that contains the lidar data.

## Data Analysis

Use `eda.py` to perform exploratory data analysis on point cloud data.

```bash
python eda.py --seq_list ${sequence_name} --seq_path ${sequence_path}
```

- `--seq_list`: A list of folder names, where each name corresponds to a sequence that is analyzed.
- `--seq_path`: Directory path that contains the folders for the sequences specified in the `--seq_list`.
- `--eda_name`: Name of the EDA output (default: `custom`).
- `--sample`: Number of bounding boxes to sample for analysis (default: 500). If this parameter is not specified, all bounding boxes in each sequence will be analyzed.
- `--bins`: Method to calculate histogram bins (default: `sqrt`).
- `--xmax`: Maximum value of the x-dimension (default: 2000).

## Visualization

Use `vis_seq.py` to visualize point cloud data sequences.

```bash
python vis_seq.py --points_path ${points_path} --remove_intensity
```

- `--points_path`: Path to the directory containing the point cloud data.
- `--remove_intensity`: Draw point cloud data without intensity values.

Use `vis_gt.py` to visualize ground truth bounding boxes in point cloud data. 

```bash
python vis_gt.py --data_path ${data_path} --remove_intensity
```

- `--data_path`: Directory path containing both `points` and `labels` folders. If this parameter is used, `--points_path` and `--labels_path` will be set to the `points` and `labels` subfolders within this directory, respectively.
- `--points_path`: Specific path to the point cloud data file or directory. This is only needed if `--data_path` is not used.
- `--labels_path`: Specific path to the labels file or directory. This is only needed if `--data_path` is not used.
- `--ext`: File extension for point cloud data files (default: `.npy`).
- `--remove_intensity`: Draw point cloud data without intensity values.
- `--use_tracking`: Draw traces by using tracking IDs from labels.
- `--bins`: Filter bounding boxes based on the number of points inside. If one number is provided, it sets the range between 0 and this number. If two numbers are provided, it sets the range between these two numbers.
- `--all`: Used in conjunction with `--bins`. Draw selected bounding boxes in red color with other boxes in green.
