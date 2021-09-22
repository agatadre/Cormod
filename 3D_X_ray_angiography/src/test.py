import dicomTools as dcm
import deviceConfiguration as configuration
import pointsCollecting as points
import math as m
import numpy as np
import pointCloudTools as pct
import centerline_extraction as centerline
import branch_detection as branch
import borders
import open3d as o3d
import cv2
import splineInterpolation as splint
import matplotlib.pyplot as plt


def test_cube_with_calibration():
    transparent_cube_points_1 = np.array(
        [[8./3, -8./3], [8./3, 8./3], [-8./3, 8./3], [-8./3, -8./3],    # big rectangle outside
         [8./5, -8./5], [8./5, 8./5], [-8./5, 8./5], [-8./5, -8./5]])   # small rectangle inside

    transparent_cube_points_2 = np.array(
        [[8./3, -8./3], [8./3, 8./3], [-8./3, 8./3], [-8./3, -8./3],    # big rectangle outside
         [8./5, -8./5], [8./5, 8./5], [-8./5, 8./5], [-8./5, -8./5]])   # small rectangle inside

    sid = 8
    sod = 4
    alpha_a90 = m.pi/2
    beta_a90 = 0
    alpha_b90 = 0
    beta_b90 = m.pi/2

    # map A1,B1... points to A2,B2...

    # corresponding points for alpha=90
    # view rotated to LAO
    corresponding_points_a90 = [0, 1, 2, 3, 4, 5, 6, 7]
    corresponding_points_a90 = [1, 5, 6, 2, 0, 4, 7, 3]

    # corresponding points for alpha=0, beta=90
    # view rotated to CRA
    corresponding_points_b90 = [4, 5, 1, 0, 7, 6, 2, 3]

    # cube1 = image.generate_3d_view(transparent_cube_points_1, transparent_cube_points_2, alpha_a90, beta_a90,
    #                                sid, sod, sid, sod, corresponding_points_a90, None, 'a90', False)
    # cube2 = image.generate_3d_view(transparent_cube_points_1, transparent_cube_points_2, alpha_b90, beta_b90,
    #                                sid, sod, sid, sod, corresponding_points_b90, None, 'b90', False)

    # GENETIC ALGORITHM TESTING
    conf = configuration.DeviceConfiguration(transparent_cube_points_1, transparent_cube_points_2,
                                             corresponding_points_a90, sid, sod, sid, sod, alpha_a90, beta_a90)
    param_dict = conf.get_calibration_params()
    # optimized_variables = conf.get_var_dict(param_dict.get('variable'))

    # generate cube with parameters acquired in device calibration
    conf.run_configured_3d_view_generation('a90', True)


def test_pyramid(select):
    # pyramid_points_1 = np.array([[-8./3, -8], [8./3, -8], [-8./5, -24./5], [8./5, -24./5], [0, 3]])
    # pyramid_points_2 = np.array([[-8./3, -8], [8./3, -8], [-8./5, -24./5], [8./5, -24./5], [0, 3]])
    # sid = 8
    # sod = 4
    # alpha_a90 = 0
    # beta_a90 = m.pi/2
    # corresponding_points_a90 = [1, 3, 0, 2, 4]
    # pyramid = image.generate_3d_view(pyramid_points_1, pyramid_points_2, alpha_a90, beta_a90,
    #                                  sid, sod, sid, sod, corresponding_points_a90, None, 'a90', True)

    # POINTS COLLECTING
    points_collector = points.PointsCollecting()
    png_paths = ['../res/pyramidTest/pyramidShadow_mask_1.png', '../res/pyramidTest/pyramidShadow_mask_2.png']
    png_paths = ['../res/masks/pyramidShadow_maska_fatt_sharpen.png', '../res/masks/pyramidShadow_maska_fatt_sharpen.png']
    csv_path = '../res/corr_points_pyramid.csv'
    # select points and save them to file if it is required
    if select is True:
        points_collector.save_corresponding_points_from_images(png_paths[0], png_paths[1], csv_path)

    # load selected points from file
    pngs_points_list = points_collector.load_corresponding_points(csv_path)
    if len(pngs_points_list[0]) != len(pngs_points_list[1]):
        print('Numbers of corresponding points must be equal')
        return 1

    png_points = np.array(pngs_points_list, dtype=np.float32)
    sid = 8
    sod = 4
    alpha_a90 = 0
    beta_a90 = m.pi/2
    pixel_spacings = 1./20
    corresponding_points = list(range(len(pngs_points_list[0])))
    image_sizes = [points_collector.get_image_size(png_paths[0]), points_collector.get_image_size(png_paths[1])]
    if image_sizes[0] != image_sizes[1]:
        print('Use images of the same size.\n Different image sizes are not implemented yet')
        return 2
    device = configuration.DeviceConfiguration(png_points[0], png_points[1], corresponding_points,
                                               sid, sod, sid, sod, alpha_a90, beta_a90,
                                               image_sizes[0], [pixel_spacings, pixel_spacings])
    # no calibration
    # opt_params = device.get_calibration_params()

    device.run_configured_3d_view_generation('test', False)
    return device


def test_dicom_reading():
    dicom_file = dcm.DicomFile('../res/dicoms/exam1.dcm')
    dicom_file.dicom_to_pngs('../res/pngs/test1')
    print(dicom_file.get_sid())
    print(dicom_file.get_sod())
    print(dicom_file.get_alpha())
    print(dicom_file.get_beta())
    print(dicom_file.get_pixel_spacing())
    dicom_file.save_metadata_to_file('../res/pngs/test1/metadata.txt')


def test_dicom_bifurcations_to_3d(png_paths=None, csv_path=None, dicom_paths=None, select=False):
    # POINTS COLLECTING
    points_collector = points.PointsCollecting()
    if png_paths is None:
        png_paths = ['../res/badania_oznaczone/corresponding/3_10.jpeg', '../res/badania_oznaczone/corresponding/6_16.jpeg']
    # csv_paths = ['../res/points_exam1.csv', '../res/points_exam2.csv']
    if csv_path is None:
        csv_path = '../res/corr_points_test.csv'

    # select points and save them to file
    if select is True:
        points_collector.save_corresponding_points_from_images(png_paths[0], png_paths[1], csv_path)
    # points_collector.save_points_from_image(png_paths[0], csv_paths[0])
    # points_collector.save_points_from_image(png_paths[1], csv_paths[1])

    # load selected points from file
    pngs_points_list = points_collector.load_corresponding_points(csv_path)
    # png_1_points = points_collector.load_points(csv_paths[0])
    # png_2_points = points_collector.load_points(csv_paths[1])
    if len(pngs_points_list[0]) != len(pngs_points_list[1]):
        print('Numbers of corresponding points must be equal')
        return 1

    png_points = np.array(pngs_points_list, dtype=np.float32)

    # OTHER PARAMS COLLECTING FROM DICOM FILE
    if dicom_paths is None:
        dicom_paths = ['../res/dicoms/exam1.dcm', '../res/dicoms/exam2.dcm']
    dicom_files = [dcm.DicomFile(dicom_paths[0]), dcm.DicomFile(dicom_paths[1])]
    sids = [dicom_files[0].get_sid(), dicom_files[1].get_sid()]
    sods = [dicom_files[0].get_sod(), dicom_files[1].get_sod()]
    alpha = np.deg2rad(dicom_files[1].get_alpha() - dicom_files[0].get_alpha())
    beta = np.deg2rad(dicom_files[1].get_beta() - dicom_files[0].get_beta())
    corresponding_points = list(range(len(pngs_points_list[0])))
    image_sizes = [points_collector.get_image_size(png_paths[0]), points_collector.get_image_size(png_paths[1])]
    if image_sizes[0] != image_sizes[1]:
        print('Use images of the same size.\n Different image sizes are not implemented yet')
        return 2
    pixel_spacings = [dicom_files[0].get_pixel_spacing(), dicom_files[1].get_pixel_spacing()]
    if pixel_spacings[0] != pixel_spacings[1]:
        print('Use images with the same pixel spacing.\n Image with different pixel spacings are not supported yet')
        return 3

    device = configuration.DeviceConfiguration(png_points[0], png_points[1], corresponding_points,
                                               sids[0], sods[0], sids[1], sods[1], alpha, beta,
                                               image_sizes[0], pixel_spacings[0])
    opt_params = device.get_calibration_params()
    # device.run_configured_3d_view_generation('test', False)
    return device


def borders_test(select):
    mask_path = '../res/masks/0016.png'
    cent_path = '../res/Inputs/centerline16.csv'
    points_collector = points.PointsCollecting()
    if select:
        points_collector.save_points_from_image(mask_path, cent_path)
    centerline_points = points_collector.load_points(cent_path)
    centerline_points = np.array(centerline_points)
    # centerline_test_points = np.array([[100, 130], [120, 125], [141, 127]])
    border_points = borders.get_2d_border_points(centerline_points, mask_path)
    # in the next step diameter should be scaled by ratio dist(F, 3D_point)/dist(F, 2D_point)

    # centerline_points = centerline.vesselSegmentation(mask_path, corr_path, None)
    print('hi')


def test_pyramid_centerlines(select):
    # calibration
    device = test_pyramid(False)
    mask_path = '../res/masks/pyramidShadow_maska_fatt.png'
    cent_path = '../res/Inputs/pyramid/pyr_centerline_corr.csv'

    # collect centerline points
    points_collector = points.PointsCollecting()
    if select:
        points_collector.save_corresponding_points_from_images(mask_path, mask_path, cent_path)
    cent_points_list = points_collector.load_corresponding_points(cent_path)
    cent_points = np.array(cent_points_list, dtype=np.float32)
    device.change_points(cent_points[0], cent_points[1], list(range(len(cent_points_list[0]))))
    device.run_configured_3d_view_generation('cent_with_corr', True)
    print('bbb')


def show_centerline_points(c1, c2, mask_paths):
    mask1 = cv2.imread(mask_paths[0], 1)
    for i, segment in enumerate(c1):
        for c in segment:
            x, y = list(c)
            mask1[y - 2:y + 2, x - 2:x + 2] = [255, 0, 0]

    mask2 = cv2.imread(mask_paths[1], 1)
    for segment in c2:
        for c in segment:
            x, y = list(c)
            mask2[y - 2:y + 2, x - 2:x + 2] = [0, 255, 0]

    cv2.imshow("c1", mask1)
    cv2.imshow("c2", mask2)
    cv2.waitKey(0)

def show_segments_with_bifurs(c1, c2, mask_paths, bifurs_list1):
    mask1 = cv2.imread(mask_paths[0], 1)
    for i, segment in enumerate(c1):
        for c in segment:
            x, y = list(c)
            mask1[y - 2:y + 2, x - 2:x + 2] = [255, 0, 0]
        for bifur in bifurs_list1[i]:
            x, y = list(bifur)
            mask1[y - 4:y + 4, x - 4:x + 4] = [0, 0, 255]
        cv2.imshow("seg " + str(i), mask1)

        for c in segment:
            x, y = list(c)
            mask1[y - 2:y + 2, x - 2:x + 2] = [255, 255, 255]
        for bifur in bifurs_list1[i]:
            x, y = list(bifur)
            mask1[y - 4:y + 4, x - 4:x + 4] = [255, 255, 255]

    cv2.waitKey(0)


def test_point_cloud(test_num, select):

    # choosing files for wanted test
    if test_num == 1:
        mask_paths = ['../res/masks/0019_01.png', '../res/masks/0025_01.png']
        csv_path = '../res/corr_points_pyramid19_01_25_01.csv'
        dicom_paths = ['../res/dicoms/exam19.dcm', '../res/dicoms/exam25.dcm']
    elif test_num == 2:
        mask_paths = ['../res/masks/0016_01.png', '../res/masks/0023_02.png']
        csv_path = '../res/corr_points_pyramid16_01_23_02.csv'
        dicom_paths = ['../res/dicoms/exam16.dcm', '../res/dicoms/exam23.dcm']
    elif test_num == 3:
        mask_paths = ['../res/masks/0025_1_01.png', '../res/masks/0034_01.png']
        csv_path = '../res/corr_points_pyramid25_1_01_34_01.csv'
        dicom_paths = ['../res/dicoms/exam25_1.dcm', '../res/dicoms/exam34.dcm']
    elif test_num == 4:
        mask_paths = ['../res/masks/0021_01.png', '../res/masks/0023_001.png']
        csv_path = '../res/corr_points_pyramid21_01_23_001.csv'
        dicom_paths = ['../res/dicoms/exam21.dcm', '../res/dicoms/exam23_1.dcm']
    elif test_num == 5:
        mask_paths = ['../res/reduced_vessel_test/vessel_6_reduced_mask.png',
                      '../res/reduced_vessel_test/vessel_18_reduced_mask.png']
        dicom_paths = ['../res/reduced_vessel_test/vessel_6.dcm',
                       '../res/reduced_vessel_test/vessel_18.dcm']
        csv_path = '../res/reduced_vessel_test/reduced_vessel_corr_points.csv'
    else:
        print('Wrong test number')
        return

    # calibrate device and marking bifurcation points
    device = test_dicom_bifurcations_to_3d(mask_paths, csv_path, dicom_paths, select)

    # collect centerline points
    points_collector = points.PointsCollecting()
    if select:
        points_collector.save_points_from_image(mask_paths[0], csv_path[0])
        points_collector.save_points_from_image(mask_paths[1], csv_path[1])

    # vessel segmentation and matching segments from two views
    order, c1, c2, bifurs1, bifurs2, g1, g2 = centerline.prepareInputs(mask_paths[0], mask_paths[1], csv_path)
    c1, c2 = centerline.compose_full_centerline((c1, c2), (bifurs1, bifurs2), (g1, g2))
    bif_orders = centerline.getBifursOrder(g1, bifurs1, bifurs2)

    pyramid_pcd = []
    for i, pair in enumerate(order):
        centerline_points_1 = c1[pair[0]]
        centerline_points_2 = c2[pair[1]]

        centerline_points_1 = np.array(centerline_points_1, dtype=np.float64)
        centerline_points_2 = np.array(centerline_points_2, dtype=np.float64)

        # reconstruct 3d centerline
        cent_3d_points, cent_3d_spline = pct.minimize_reprojection_error(device, [centerline_points_1,
                                                                                  centerline_points_2], (bif_orders[0][i], bif_orders[1][i]), mask_paths)

        pcd = o3d.geometry.PointCloud()
        to_draw = cent_3d_points
        if cent_3d_spline is not None:
            to_draw = np.vstack([cent_3d_points, cent_3d_spline])
        pcd.points = o3d.utility.Vector3dVector(to_draw)
        pyramid_pcd.append(pcd)

    o3d.visualization.draw_geometries(pyramid_pcd)


def test_reduce():
    mask_path = '../res/masks/0016_1.png'
    branch.reduceVessel(mask_path)


def getClosetsProjections():
    dir_path = '../../../data/documents/1.3.6.1.4.1.19291.2.1.1.11401331441172059354581860543423'
    dicoms = dcm.get_closest_projections(dir_path)

    print(dicoms[0][0].getPath())
    print(dicoms[0][1].getPath())


def test_new_splines():
    x_axis = [1, 2, 3, 4, 9]
    y_axis = [2, 3, 4, 5, 9]
    z_axis = [3, 4, 7, 5, 9]
    spline = splint.cubicSplineInterpolate(x_axis, y_axis, z_axis)
    fig = plt.figure()
    my_ax = fig.add_subplot(111, projection='3d')
    my_ax.plot(x_axis, y_axis, z_axis, 'ro')
    my_ax.plot(spline[0], spline[1], spline[2], color="red")
    plt.show()
    print('spline')


if __name__ == "__main__":
    test_point_cloud(1, False)
