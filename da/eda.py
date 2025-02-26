import os
import json
import argparse
import logging
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.stats import describe

SAVE_PATH = "./results"

class Dataset:
    def __init__(self, path):
        self.path = path

    def load_pcd(self, pcd_path):
        np_pcd = np.load(pcd_path, allow_pickle=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pcd[:,:3])
        return pcd
    
    def load_bboxes(self, bbox_path, has_id=False):
        with open(bbox_path, 'r') as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            line = line.strip().split(' ')
            if has_id:
                bbox = BBox(float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), str(line[7]), str(line[8]))
            else:
                bbox = BBox(float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), "NA", str(line[7]))
            bboxes.append(translate_boxes_to_o3d_instance(bbox.get_gt_boxes())[1])
        return bboxes

class BBox:
    def __init__(self, x, y, z, dx, dy, dz, yaw, id, cls):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.yaw = yaw
        self.id = id    
        self.cls = cls
        self.gt_boxes = self.get_gt_boxes()

    def get_gt_boxes(self):
        return np.array([self.x, self.y, self.z, self.dx, self.dy, self.dz, self.yaw])

def translate_boxes_to_o3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
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

def load_data(dataset_path, data_name):
    data = Dataset(dataset_path)
    pcd = data.load_pcd(dataset_path + f"points/{data_name}.npy")
    bboxes = data.load_bboxes(dataset_path + f"labels/{data_name}.txt")
    return data, pcd, bboxes

def calculate_stats(pcd, bboxes):
    pts_lst = []
    dense_lst = []
    lwh_lst = []
    
    for i in range(0, len(bboxes)):
        points_num = len(pcd.crop(bboxes[i]).points)
        pts_lst.append(points_num)
        dense_lst.append(points_num/(bboxes[i].volume()))
        lwh_lst.append(bboxes[i].extent)

    return pts_lst, dense_lst, lwh_lst

def process_dataset(seq, dataset_path, sample_size, output_file):
    dataset = Dataset(dataset_path)
    pcd_files = os.listdir(os.path.join(dataset_path, 'points'))

    all_point_counts = []  # points per bbox
    excluded_bboxes = []  # bboxes without points

    all_dense_counts = []  # points per volume
    all_lwh = []  # extent per bbox

    maxs, mins = [], []
    
    for filename in pcd_files:
        pcd_path = os.path.join(dataset_path, 'points', filename)
        bbox_path = os.path.join(dataset_path, 'labels', filename.replace('.npy', '.txt'))
        pcd = dataset.load_pcd(pcd_path)
        points = np.asarray(pcd.points)
        maxs.append(points.max(axis=0))
        mins.append(points.min(axis=0))
        bboxes = dataset.load_bboxes(bbox_path)
        point_counts, dense_counts, lwh = calculate_stats(pcd, bboxes)

        for idx, cnt in enumerate(point_counts):
            if cnt > 0:
                all_point_counts.append(cnt)
                all_dense_counts.append(dense_counts[idx])
                all_lwh.append(lwh[idx])
            else:
                excluded_bboxes.append((filename, idx))
    
    maxs = np.array(maxs)
    mins = np.array(mins)
    max_info = maxs.max(axis=0)
    mins_info = mins.min(axis=0)
    med_max = np.median(maxs, axis=0)
    med_min = np.median(mins, axis=0)
    minmax_info = {
        "maxs": max_info,
        "mins": mins_info,
        "median_maxs": med_max,
        "median_mins": med_min
    }

    if sample_size:
        paired = list(zip(all_point_counts, all_dense_counts, all_lwh))
        sample_pairs = random.sample(paired, min(len(all_point_counts), sample_size))
        all_point_counts, all_dense_counts, all_lwh = zip(*sample_pairs)

    stats_pts = describe(all_point_counts)
    stats_dense = describe(all_dense_counts)
    stats_lwh = describe(all_lwh)
    
    # exclude boxes without points
    if output_file:
        with open(f"{output_file}/{seq}_exclude.txt", 'w') as f:
            f.write(f"{len(excluded_bboxes)}\n{excluded_bboxes}")
    
    return stats_pts, all_point_counts, stats_dense, all_dense_counts, stats_lwh, all_lwh, minmax_info

def plot_histogram(lst, title, x_label, y_label, save_path, bins, x_max):
    plt.clf()
    plt.hist(lst, bins, range=(0, x_max), color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"{save_path}/{title}.png")
    plt.show()

def split_lwh(arrays):
    l, w, h = [], [], [] 
    for arr in arrays:
        l.append(arr[0])
        w.append(arr[1])
        h.append(arr[2])
    return l, w, h

def calculate_info(stats, cnts, flag=None):
    describe_dict = {
        "nobs": stats.nobs,
        "minmax": stats.minmax,
        "mean": stats.mean,
        "variance": stats.variance,
        "skewness": stats.skewness,
        "kurtosis": stats.kurtosis
    }
    if flag:
        l, w, h = split_lwh(cnts)
        stats_info = {
            "Describe": describe_dict,
            "Median": [np.median(l).item(), np.median(w).item(), np.median(h).item()],
            "Average": describe_dict["mean"],
            "5th Percentile": [np.percentile(l, 5).item(), np.percentile(w, 5).item(), np.percentile(h, 5).item()],
            "95th Percentile": [np.percentile(l, 95).item(), np.percentile(w, 95).item(), np.percentile(h, 95).item()],
            "Standard Deviation": (describe_dict["variance"]**0.5)
        }
        return stats_info
        
    stats_info = {
        "Describe": describe_dict,
        "Median": np.median(cnts).item(),
        "Average": stats.mean.item(),
        "5th Percentile": np.percentile(cnts, 5).item(),
        "95th Percentile": np.percentile(cnts, 95).item(),
        "Standard Deviation": (stats.variance**0.5).item()
    }
    return stats_info

def numpy_json_serializer(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type '{type(obj)}' is not JSON serializable")

def eda(eda_name, seq_list, seq_path, sample, bins, xmax):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    save_path = os.path.join(SAVE_PATH, eda_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results = {}
    for seq in seq_list:
        seq_path = os.path.join(seq_path, seq)
        if seq not in results:
            results[seq] = {}

        logging.info("Processing {}".format(seq))
        seq_pts_stats, seq_pts_cnts, seq_dens_stats, seq_dens_cnts, seq_lwh_stats, seq_lwh, minmax_info = process_dataset(seq, seq_path, sample_size=sample, output_file=save_path)
        results[seq]["point cloud info"] = minmax_info

        seq_pts_info = calculate_info(seq_pts_stats, seq_pts_cnts)
        plot_histogram(seq_pts_cnts, f"{seq} (sample={sample}) point counts in bboxes", "number of points in bboxes", "number of bboxes", save_path, bins, xmax)
        results[seq]["point counts in bboxes"] = seq_pts_info

        seq_dens_info = calculate_info(seq_dens_stats, seq_dens_cnts)
        plot_histogram(seq_dens_cnts, f"{seq} (sample={sample}) points per volume", "points per volume", "number", save_path, bins, xmax)
        results[seq]["points per volume"] = seq_dens_info

        seq_lwh_info = calculate_info(seq_lwh_stats, seq_lwh, flag=1)
        results[seq]["extent per bbox"] = seq_lwh_info

    json_path = os.path.join(save_path, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, default=numpy_json_serializer, indent=4)

def get_sequence_list(seq_list):
    seq_list = seq_list.split(',')
    seq_list = [seq.strip() for seq in seq_list]
    return seq_list

def check_data_exist(seq_list, seq_path):
    for seq_name in seq_list:
        seq_path = os.path.join(seq_path, seq_name)
        if not os.path.exists(seq_path):
            logging.error(f"{seq_name} does not exist")
            exit(1)
    return True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--eda_name', type=str, default='custom')
    parser.add_argument('--seq_list', type=str)
    parser.add_argument('--seq_path', type=str)
    parser.add_argument('--sample', type=int, default=500)
    parser.add_argument('--bins', type=str, default='sqrt')
    parser.add_argument('--xmax', type=int, default=2000)

    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO)
    
    if not os.path.exists(args.seq_path):
        logging.error("Sequence path does not exist")
        exit(1)

    seq_path = os.path.abspath(args.seq_path)

    logging.info("EDA: {}".format(args.eda_name))
    logging.info("Sequence path: {}".format(seq_path))
    logging.info("Sequence list: {}".format(args.seq_list))

    eda_name = args.eda_name
    bins = args.bins
    xmax = args.xmax
    sample = args.sample

    seq_list = get_sequence_list(args.seq_list)
    check_data_exist(seq_list, seq_path)

    eda(eda_name, seq_list, seq_path, sample, bins, xmax)

    logging.info("Success")

if __name__ == '__main__':
    main()
