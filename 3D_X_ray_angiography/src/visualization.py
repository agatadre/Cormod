from pypcd import pypcd
import numpy as np
import open3d as o3d

def show_point_cloud(pcd_file='cat.pcd', rgb=False):
    """
    Shows visualization of point cloud given in pcd file
    :param pcd_file: path to pcd file that contains point cloud
    :param rgb: whether given file has rgb column or not
    :return:
    """
    pcd = o3d.io.read_point_cloud(pcd_file)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[-2, -2, -2])
    o3d.visualization.draw_geometries([pcd, mesh_frame])


def generate_random_rgb_colors(num_of_colors):
    """
    Generates array of random rgb colors
    :param num_of_colors: number of colors to generate
    :return: array (n, 3) of random rgb colors
    """
    arr = np.empty([num_of_colors, 3])
    for num in range(num_of_colors):
        arr[num, :] = np.array(np.random.choice(range(256), size=3))
    return arr


def save_and_show_point_cloud(points_array, point_cloud_name='test', rgb_colors=None):
    """
    Saves points to file as point cloud and visualize it
    :param points_array: array of xyz points coordinates
    :param point_cloud_name: name of newly created file
    :param rgb_colors: array with rgb colors corresponding with points at the same index in points_array
    :return:
    """
    rgb_enabled = False
    if rgb_colors is not None:
        encoded_colors = pypcd.encode_rgb_for_pcl(rgb_colors.astype(np.uint8))
        point_cloud = pypcd.make_xyz_rgb_point_cloud(np.hstack((points_array.astype(np.float32), encoded_colors[:, np.newaxis])))
        rgb_enabled = True
    else:
        point_cloud = pypcd.make_xyz_point_cloud(points_array)
    point_cloud.save(point_cloud_name + '.pcd')
    show_point_cloud(point_cloud_name + '.pcd', rgb_enabled)


def visualize_all(shadow_1, shadow_2, real_object, rays_source_1, rays_source_2, view_name, corresponding_points):
    """
    Visualize three views (shadows with F points, real object, shadows with real object and F points)
    :param shadow_1: array with xyz1 coordinates that create first shadow
    :param shadow_2: array with xyz1 coordinates that create second shadow
    :param real_object: array with xyz1 coordinates that create real object
    :param rays_source_1: xyz1 coordinate of first ray source
    :param rays_source_2: xyz1 coordinate of second ray source
    :param view_name: view name appended to filename
    :param corresponding_points: array that contains indexes of corresponding points in shadow_1 and shadow_2 arrays
            so that we can mark them with same colors
    :return:
    """
    # stack points for visualization
    all_shadow_points = np.vstack((shadow_1, shadow_2))

    # remove additional dimension
    all_shadow_points_3d = all_shadow_points[:, :-1]
    real_object_points_array_3d = real_object[:, :-1]
    all_shadow_points_3d = np.vstack((all_shadow_points_3d, rays_source_1[:-1], rays_source_2[:-1]))

    # assign random colors to proper corresponding points
    num_of_points_in_one_shadow = len(corresponding_points)
    shadow_random_colors = generate_random_rgb_colors(shadow_1.shape[0] + 1)
    shadows_rgb_colors = np.empty([2 * shadow_random_colors.shape[0], 3])
    object_rgb_colors = np.empty([num_of_points_in_one_shadow, 3])
    for i in range(num_of_points_in_one_shadow):
        shadows_rgb_colors[i, :] = shadow_random_colors[i, :]
        shadows_rgb_colors[num_of_points_in_one_shadow + corresponding_points[i], :] = shadow_random_colors[i, :]
        object_rgb_colors[i, :] = shadow_random_colors[i, :]
    # add ray_sources colors
    shadows_rgb_colors[-1, :] = shadow_random_colors[-1, :]
    shadows_rgb_colors[-2, :] = shadow_random_colors[-1, :]

    save_and_show_point_cloud(all_shadow_points_3d, 'shadows_' + view_name, shadows_rgb_colors)
    save_and_show_point_cloud(real_object_points_array_3d, 'object_' + view_name, object_rgb_colors)

    # show everything together
    all_points = np.vstack((all_shadow_points_3d, real_object_points_array_3d))
    save_and_show_point_cloud(all_points, 'all_points_' + view_name, np.vstack((shadows_rgb_colors, object_rgb_colors)))
