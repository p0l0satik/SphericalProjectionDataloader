import numpy as np
import open3d as o3d

from pathlib import Path

import loader.constants as constants


class LaserScan:
    """Class that contains LaserS
    can with x,y,z,r,sem_label"""

    def __init__(self, heigh=64, width=1024, fov_up=3.0, fov_down=-25.0, visualise_ransac=False):
        """Function take as input desirable height and width of the image, and fov of the lidar rays"""
        self.proj_height = heigh
        self.proj_width = width
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.visualise_ransac = visualise_ransac
        self.reset()

    def reset(self):
        """Reset scan members."""
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_height, self.proj_width), -1, dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_height, self.proj_width, 3), -1, dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_height, self.proj_width), -1, dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros(
            (self.proj_height, self.proj_width), dtype=np.int32
        )  # [H,W] mask

    def size(self):
        """Return the size of the point cloud."""
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename: Path):
        """Open raw scan and fill in attributes"""
        # reset just in case there was an open structure
        self.reset()

        # if all goes well, open pointcloud
        if filename.suffix == constants.POINTCLOUD_EXT[0]:
            scan = np.fromfile(filename, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            points = scan[:, 0:3]  # get xyz
        elif filename.suffix == constants.POINTCLOUD_EXT[1]:
            pcd = o3d.io.read_point_cloud(filename)
            points = np.asarray(pcd.points)  # get xyz
        else:
            raise RuntimeError("Filename extension is not valid scan file.")

        self.set_points(points)

    def set_points(self, points):
        """Set scan attributes (instead of opening from file)"""
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        self.points = points  # get xyz
        self.do_range_projection()

    def do_range_projection(self):
        """Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_width  # in [0.0, W]
        proj_y *= self.proj_height  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.clip(proj_x, 0, self.proj_width - 1).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.clip(proj_y, 0, self.proj_height - 1).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)

    def open_label(self, filename: Path, ransac=True):
        """Open labels and so label rpojection"""
        # if all goes well, open label
        if filename.suffix == constants.LABELS_EXT[0]:
            label = np.load(filename)
        elif filename.suffix == constants.LABELS_EXT[1]:
            label = np.fromfile(filename, dtype=np.uint32)
        else:
            raise RuntimeError("Label filename extension is not valid scan file.")

        label = label.reshape((-1))

        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16  # instance id in upper half
        else:
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        if not (sem_label + (inst_label << 16) == label).all():
            raise ValueError("An error occurred during labels parsing")

        if ransac:
            sem_label = apply_ransac_filtering(
                points=self.points,
                labels=sem_label,
                visualisation=self.visualise_ransac,
            )

        proj_sem_label = np.zeros((self.proj_height, self.proj_width), dtype=np.int32)
        mask = self.proj_idx >= 0
        proj_sem_label[mask] = sem_label[self.proj_idx[mask]]
        return sem_label, proj_sem_label


def apply_ransac_filtering(points, labels, visualisation=False):
    # init new labels new label
    new_labels = np.zeros_like(labels)
    # calculating unique label values
    unique_labels = np.unique(labels)

    for label_inst in unique_labels:
        # skip empty points
        if label_inst == 0:
            continue

        # indexes of labels we need
        filtered_idx = np.argwhere(labels == label_inst)
        if len(filtered_idx) < constants.RANSAC_N:
            continue
        # filtered_lables = labels[filtered_idx]
        filtered_point = points[filtered_idx, :].squeeze()
        # creating pointcloud from points belonging to filtered labels and
        # fitting plane with RANSAC
        pcd_f = o3d.geometry.PointCloud()
        pcd_f.points = o3d.utility.Vector3dVector(filtered_point)
        _, inliers = pcd_f.segment_plane(
            distance_threshold=constants.RANSAC_THRESHOLD,
            ransac_n=constants.RANSAC_N,
            num_iterations=constants.RANSAC_N_ITER,
        )
        # adding filtered labels as "planes"
        new_labels[filtered_idx[inliers]] = 1

    if visualisation:
        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(points)
        filtered_colors = np.zeros_like(points)
        filtered_colors[:, 0] = 255 * new_labels
        filtered_colors[:, 1] = 255 * labels
        filtered.colors = o3d.utility.Vector3dVector(filtered_colors)
        o3d.visualization.draw_geometries([filtered])

    return new_labels
