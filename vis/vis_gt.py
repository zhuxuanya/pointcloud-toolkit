import os
import argparse
import logging
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def translate_boxes_to_o3d_instance(gt_boxes):
    '''
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    '''
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d

def should_display_box(box3d_num_points, bins):
    if bins:
        bins_range = list(map(int, bins.split(',')))
        if len(bins_range) == 1:
            return box3d_num_points <= bins_range[0]
        elif len(bins_range) == 2:
            return bins_range[0] <= box3d_num_points <= bins_range[1]
        else:
            logging.error('Expected at most 2 inputs for bins')
            return
    return True

def draw_track_history(vis, track_history, track_id, current_point, color):
    # extract the x, y coordinates for 2D trajectory visualization
    point_2d = current_point[:3]

    if track_id not in track_history:
        track_history[track_id] = {'points': [], 'color': color}
    track_history[track_id]['points'].append(point_2d)

    # draw track history for track ID
    if len(track_history[track_id]['points']) > 1:
        track_points = np.array(track_history[track_id]['points'])
        if track_points.ndim == 1:
            track_points = np.expand_dims(track_points, axis=0)
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.hstack([track_points])),
            lines=o3d.utility.Vector2iVector(np.vstack([np.arange(track_points.shape[0]-1), np.arange(1, track_points.shape[0])]).T)
        )
        line.paint_uniform_color(track_history[track_id]['color'])
        vis.add_geometry(line)

def draw_box_with_track(vis, pts, gt_boxes, color=None, bins=None, all=None, track_ids=None, track_history=None):
    vis.add_geometry(pts)

    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_o3d_instance(gt_boxes[i])
        box3d_num_points = len(pts.crop(box3d).points)
        
        if not should_display_box(box3d_num_points, bins) and not all:
            continue
        
        gt_color = (0, 1, 0)  # default box color: green
        
        if bins and should_display_box(box3d_num_points, bins):
            gt_color = (1, 0, 0)  # filtered box color: red
        
        if color:
            gt_color = color[i]  # if use tracking
        
        line_set.paint_uniform_color(gt_color)
        vis.add_geometry(line_set)

        if track_ids and len(track_ids) > i and track_history is not None:
            track_id = track_ids[i]
            current_center = box3d.get_center()
            draw_track_history(vis, track_history, track_id, current_center, gt_color)
    
    return vis

class Dataset():
    def __init__(self, points_path, labels_path, trackingInfoPresent=False, ext='.npy'):
        self.label_list = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])
        self.points_path = points_path
        self.labels_path = labels_path
        self.trackingInfoPresent = trackingInfoPresent
        self.colour_dict = {}
        self.ext = ext

    def __len__(self):
        return len(self.label_list)
    
    def get_colour(self, track_id):
        if track_id not in self.colour_dict:
            self.colour_dict[track_id] = np.array([random.random(), random.random(), random.random()], dtype=np.float64)
        
        return self.colour_dict[track_id]
        
    def __getitem__(self, index):
        data_dict = {}
        label_filename = self.label_list[index]
        label_filepath = os.path.join(self.labels_path, label_filename)
        with open(label_filepath, 'r') as f:
            labels = f.readlines()
        labels = [x.strip() for x in labels]
        labels = [label.split(' ') for label in labels]
        gt_boxes = np.asarray([label[:7] for label in labels], dtype=np.float64)

        frame_id = label_filename.split('.')[0]
        if self.points_path:
            if self.ext == '.npy':
                data_filename = frame_id + '.npy'
                data_filepath = os.path.join(self.points_path, data_filename)
                data = np.load(data_filepath)
            elif self.ext == '.pcd':
                data_filename = frame_id + '.pcd'
                data_filepath = os.path.join(self.points_path, data_filename)
                pcd = o3d.io.read_point_cloud(data_filepath)
                data = np.asarray(pcd.points)
            else:
                raise NotImplementedError(f'Extension {self.ext} not supported')
        else:
            data = np.zeros((1, 4))
        
        data_dict['frame_id'] = frame_id
        data_dict['points'] = data
        data_dict['gt_boxes'] = gt_boxes

        if self.trackingInfoPresent:
            track_ids = [label[-2] for label in labels]
            data_dict['track_ids'] = track_ids
            data_dict['colours'] = [self.get_colour(track_id) for track_id in track_ids]
        
        return data_dict

def get_points_labels(pts, test_set, idx, remove_intensity):
    data_dict = test_set[idx]
    # data_dict = test_set.collate_batch([data_dict])
    gt_boxes = data_dict['gt_boxes']
    
    if 'colours' in data_dict:
        colours = data_dict['colours']
    else:
        colours = None

    points = data_dict['points']
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    if not remove_intensity and points.shape[1] == 4:
        # use the fourth column as intensity values
        colors = plt.get_cmap('terrain')(points[:, 3] / np.max(points[:, 3]))[:, :3]  # normalize and convert to RGB
        pts.colors = o3d.utility.Vector3dVector(colors)
    else:
        pts.colors = o3d.utility.Vector3dVector(np.ones((data_dict['points'].shape[0], 3)))
    
    if 'track_ids' in data_dict:
        track_ids = data_dict['track_ids']
    else:
        track_ids = None

    return pts, test_set[idx]['frame_id'], gt_boxes, colours, track_ids

def visualize_point_clouds(test_set, remove_intensity, bins=None, all=None):
    logger = logging.getLogger('visualize_gt')
    if len(test_set) == 0:
        logger.error('No point cloud found in the directory.')
        return
    
    logger.info(f'Total number of files to visualize: {len(test_set)}')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    pts = o3d.geometry.PointCloud()
    track_history = {}

    idx = 0
    pts, frame_id, gt_boxes, colours, track_ids = get_points_labels(pts, test_set, idx, remove_intensity)
    draw_box_with_track(vis, pts, gt_boxes, colours, bins=bins, all=all, track_ids=track_ids, track_history=track_history)
    logger.info(f'Visualizing: {frame_id}')

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    def load_next_point_cloud(vis):
        nonlocal idx, pts, camera_params
        if idx < len(test_set) - 1:
            idx += 1
            pts, frame_id, gt_boxes, colours, track_ids = get_points_labels(pts, test_set, idx, remove_intensity)
            vis.clear_geometries()
            vis = draw_box_with_track(vis, pts, gt_boxes, colours, bins=bins, all=all, track_ids=track_ids, track_history=track_history)
            view_control.convert_from_pinhole_camera_parameters(camera_params)
            logger.info(f'Visualizing: {frame_id}')
        else:
            vis.destroy_window()  # signal to close the visualizer

    def load_previous_point_cloud(vis):
        nonlocal idx, pts, camera_params
        if idx > 0:
            idx -= 1
            pts, frame_id, gt_boxes, colours, track_ids = get_points_labels(pts, test_set, idx, remove_intensity)
            vis.clear_geometries()
            for track_id in track_history:
                track_history[track_id]['points'] = track_history[track_id]['points'][:-2]
            vis = draw_box_with_track(vis, pts, gt_boxes, colours, bins=bins, all=all, track_ids=track_ids, track_history=track_history)
            view_control.convert_from_pinhole_camera_parameters(camera_params)
            logger.info(f'Visualizing: {frame_id}')

    def save_camera_params(vis):
        nonlocal camera_params
        camera_params = view_control.convert_to_pinhole_camera_parameters()

    vis.register_key_callback(262, load_next_point_cloud)  # right arrow key
    vis.register_key_callback(263, load_previous_point_cloud)  # left arrow key
    vis.register_animation_callback(save_camera_params)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    vis.run()

def parse_config():
    parser = argparse.ArgumentParser(description='Visualize ground truth annotations on point cloud data')

    parser.add_argument('--data_path', type=str, default=None, help='specify directory including both points and labels folder')
    parser.add_argument('--points_path', type=str, default=None, help='specify the point cloud data file or directory')
    parser.add_argument('--labels_path', type=str, default=None, help='specify the labels file or directory')

    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of point cloud data file')
    parser.add_argument('--remove_intensity', action='store_true', help='draw point cloud data without intensity')
    parser.add_argument('--use_tracking', action='store_true', help='use the tracking id')

    parser.add_argument('--bins', type=str, default=None, help='filter bounding boxes by the number of points inside')
    parser.add_argument('--all', action='store_true', help='draw selected bboxes in red color with other bboxes in green')

    args = parser.parse_args()
    return args

def main():
    args = parse_config()

    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger('visualize_gt')
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    logger.info('-----------------Visualizing Ground Truth Data-------------------------')
    
    if args.data_path:
        test_set = Dataset(os.path.join(args.data_path, 'points'), os.path.join(args.data_path, 'labels'), trackingInfoPresent=args.use_tracking, ext=args.ext)
    else:
        test_set = Dataset(args.points_path, args.labels_path, trackingInfoPresent=args.use_tracking, ext=args.ext)

    num_samples = len(test_set)
    logger.info(f'Total number of samples: {num_samples}')

    visualize_point_clouds(test_set, args.remove_intensity, args.bins, args.all)

if __name__ == '__main__':
    main()
