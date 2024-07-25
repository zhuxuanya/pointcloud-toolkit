import os
import argparse
import numpy as np
import rosbag
from pypcd.numpy_pc2 import pointcloud2_to_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--lidar_topic', type=str)
    args = parser.parse_args()

    bag_path = args.bag_path
    out_dir = args.output_path

    if not os.path.exists(bag_path):
        print(f"[ERROR] Bag file {bag_path} not found")
        return

    bag = rosbag.Bag(bag_path)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    lidar_topic = args.lidar_topic

    counter = 0
    # iterate through the messages in the bag file
    for topic, msg, t in bag.read_messages():
        if topic != lidar_topic:
            # print(topic)
            continue
        
        structured_point_cloud = pointcloud2_to_array(msg)[['x','y','z','intensity']]
        point_cloud = np.column_stack((
            structured_point_cloud['x'].reshape(-1, 1), 
            structured_point_cloud['y'].reshape(-1, 1), 
            structured_point_cloud['z'].reshape(-1, 1), 
            structured_point_cloud['intensity'].reshape(-1, 1),
        ))

        point_cloud[:, 2] = -point_cloud[:, 2]

        file_name = str(t) + ".npy"
        file_path = os.path.join(out_dir, file_name)
        np.save(file_path, point_cloud)
        counter += 1

    bag.close()

    if counter == 0:
        print(f"[ERROR] No messages with selected topic {lidar_topic}")
    else:
        print(f"Found {counter} messages from topic {lidar_topic}")

if __name__ == "__main__":
    main()