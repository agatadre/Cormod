# *3D_X_ray_angiography 0.1.0*

This is the software which aim is to reconstruct in 3D space a vessel of coronary arteries from it's two 2D XRA projections.

### *Flow of the program*
Our program consists of three parts:
1. Preparing inputs for reconstruction like masks of a vessel and dicoms from both projections.
2. Retrieving of the most important items from the inputs like bifurcations, vessel's centerline, segments, centerline's points etc.
3. Reconstruction of the vessel's centerline as 3D curve.

### 1. Preparing Inputs
This part of our program mostly consist of handwork preparation of the input\
We operate on heart's the **end-diastolic phase** most of the time so please keep in mind that
single images we are talking about below represent this phase.

We will show input preparation based on an exemplary unit test from test.py.
We have prepared few utilities functions that may be used to prepare these inputs, instructions below.
For respectively the first and the second view, inputs are:
1. paths to dicom files from the same exam with slightly different view angles
2. paths to vessel trees in png format, it can be either a gray scale image or a binary mask as previously
3. paths to binary masks of a vessel trees
4. path to a csv file with bifurcations' coordinates written down in this pattern:

x_position_1 | y_position_1 | x_position_2 | y_position_2
------------ | ------------ | ------------ | ------------
244 | 138 | 194 | 190
287 | 339 | 253 | 478
#### Utilities for preparing input

##### Tools for exam's data extraction
The most common data format for an exam is dicom (.dcm) so we have utility class to handle files
however we encountered few projection with no dicom file included but
JSON or plain text.
For this we prepared to classes for data extraction:_DicomFile_, _JsonTools_, _PlainTextTools_  in _dicomTools.py_,
_jsonTools_ and _plainTextTool.py_ respectively.
Obviously plain text is the least structured format so its tool is the simplest one.
The most extended is _DicomFile_.

##### Matching similiar projections
To find such pair of projections, we have prepared the function
 ```get_closest_projections(directory_path, epsilon=2)``` in _dicomTools.py_
which returns tuple of paths to dicoms with the minimal angles difference but not smaller than epsilon.

##### Generating images used to mark bifurcation points
To generate such images, in class _DicomFile_ function
```dicom_to_pngs(self, directory_path, return_pngs=None)``` allows to save gray scale pngs to a given directory.
If needed, pngs may be returned as ndarray at the end to save time it takes to load them back to work with them later.
It can be done also with ```load_cv_pngs(self, directory_path)```

##### Vessel tree generation
In _EDFinder.py_ there is _EDFinder_ class with
 ```create_vessel_masks(self, gray_scale_frames)``` function.
 If given an array with array of images it returns ndarray with binary masks (examples of masks shown down below)
 Generated masks are not enough accurate to be used as inputs and they have to be corrected manually.
 Each mask from an array must be saved and corrected manually according its original vessel tree since they are not
 enough accurate to be used as inputs.
 However we use such generated masks to estimate end-diastolic phase which will be detailed later.

##### Estimating end-diastolic frame
It's possible to automatically estimate which frame is in this phase and use this as an input for reconstruction.\
Class _EDFinder_:
 ```Python
class EDFinder:
    def __init__(self, dcm_path, cp_params=None, imgs=None, masks=None) -> None
 ````
 Each parameter is described in commented in code but it's worth to mention that the most time consuming part is loading
 images from a dicom file. So there is possibility to pass array with previously loaded gray scale pngs and their masks.
```Python
get_end_diastole_frame(self, w=26, dx_max=30, dy_max=30, frame_low=None,
 frame_up=None, all_extemas=False, optimization=False)
 ```
 This function returns index/indexes of a frame (or with their data) which
 is the most probable to be in the end-diastolic phase. Such frame may be used as an input for previously described
 unit tests or, which is more important, may be used to automatize workflow of reconstruction by decreasing manual work.

##### Partial matching of projections' frames
It commonly occurs that starting acquiring point of frames is in different part in cardiac's cycle than in other projection.
Even if starting point in time would be the same, there are reasons what may lead to dissimilarities between
corresponding frames.
Using calibrated camera, exam data and known bifurcation points for each frame, we can create a subsequence for projections
with the same length and the greatest similarity between corresponding indexes.
There is package _partial_matcher.py_ providing this functionality:
```Python
def find_matching_views(prim_views, sec_views, corresponding_points,
                       alpha_diff, beta_diff, sid1, sod1, sid2, sod2, XA)
```

Details for parameters are commented in source code.

### 2. Retrieving of the most important items
After preparing inputs for the program the next step is extraction of the necessary items for the reconstruction. These items are inter alia bifurcation points, vessel's centerline, segments of a vessel, centerline's points and correspondance of the segments from two XRA projections.

The main of our program is a function test_point_cloud() in a file test.py. This function is called in main of the test.py:
```Python
if __name__ == "__main__":
    test_point_cloud(1, False)

```
It requires two arguments:
1. Number of the test set
2. Value of the flag ***select*** which decide whether user wants to mark bifurcations or use previous ones.

#### *Configuration of a device*
The first step of the program is a calibration of our virtual device which is the object of the class DeviceConfiguration, from a file deviceConfiguration.py. <br />
This class is responsible for device calibration which minimize errors during reconstruction and for reconstruction of 2D points in 3D space. <br />
This is done in function test_dicom_bifurcations_to_3d() which is called at the beginning of the function test_point_cloud():
```Python
def test_point_cloud(test_num, select):
    ...
    # calibrate device and marking bifurcation points
    device = test_dicom_bifurcations_to_3d(png_paths, csv_path, dicom_paths, select)
```

#### *Marking bifuraction points on two projections*
First step of this function is marking bifuraction points on masks from two projections of a vessel. It happens if the value of the argument ***select***, from the beginning of the program, is equal True.
```Python
def test_dicom_bifurcations_to_3d(png_paths=None, csv_path=None, dicom_paths=None, select=False):
    # POINTS COLLECTING
    points_collector = points.PointsCollecting()

    ...

    # select points and save them to file
    if select is True:
        points_collector.save_corresponding_points_from_images(png_paths[0], png_paths[1], csv_path)
    # points_collector.save_points_from_image(png_paths[0], csv_paths[0])
    # points_collector.save_points_from_image(png_paths[1], csv_paths[1])
```
If yes, the program called method save_corresponding_points_from_images of the PointsCollecting's object. This method allow the user to mark bifurcations on two previously given projections.

<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/marking.png" alt="Image of marking bifuraction points" width="800" height="400"/>
</p>

#### *Retrieving device parameters from both projections*
For device calibration and further reconstruction essential are parameters like both positioner angles ***alpha***, ***beta*** of the device, distance beetwen X-ray source (***F***) and patient - ***SOD*** and distance beetwen X-ray source and Detector plane - ***SID***.<br />
These parameters are extracted with the help of DicomFile class from the file dicomTools.py.
```Python
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
```

#### *Device calibration*
After retrieving of required parameters program starts the calibration of our device, object of the deviceConfiguration class:
```Python
device = configuration.DeviceConfiguration(png_points[0], png_points[1], corresponding_points,
                                          sids[0], sods[0], sids[1], sods[1], alpha, beta,
                                          image_sizes[0], pixel_spacings[0])
opt_params = device.get_calibration_params()
# device.run_configured_3d_view_generation('test', False)
return device
```
This process lean on minimizing the objective function with the help of genetic algorithm.
<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/calibration.png" alt="Image of the calibration progress"/>
</p>

#### *Extraction of the items*
After calibration the next step is extraction of the items from a vessel like centerline, segments or correspondance of the branches from two projections. It happens in the function ***prepareInputs()*** from the file centerline_extraction.py, invoked in ***test_point_cloud()***.
```Python
# vessel segmentation and matching segments from two views
order, c1, c2, bifurs1, bifurs2, g1, g2 = centerline.prepareInputs(mask_paths[0], mask_paths[1], csv_path)
```
That is retrieving centerline's points, vessel's segments and graphs from both projections. It is done with ***vesselSegmentation()*** function.
```Python
def prepareInputs(mask1, mask2, csv):
    """
        Test function which propose how input data for Algorithm 4
        can be prepared.
        @param mask1: mask of the first image
        @param mask2: mask of the second image
        @param csv: name of CSV file for centerline points for both views
        @return order: order of segments from two projections
        @return c1: centerline points from the first image
        @return c2: centerline points from the second image
        @return bifurs1: bifurcations from the first image
        @return bifurs2: bifurcations from the second image, they are equal to bifurs1
        @return graph1: graph of a vessel from the first image
        @return graph2: graph of a vessel from the second image
    """

    # select points and save them to file
    # get graphs and centerline points
    graph1, c1, bifurs1 = vesselSegmentation(mask1, csv, 0)
    graph2, c2, bifurs2 = vesselSegmentation(mask2, csv, 1)
```
This function consists of three steps:
1. Extraction of the vessel's centerline and splitting it into segments.
2. Constructing the graph of a vessel where vertices are represented by segments and edges by bifurcation points.
3. Retrieving centerline's points from segments. At that moment the program extract 10% of a points from each segment.

We achieve this with the help of three functions called in ***vesselSegmentation()***: ***split_into_segments()***, ***graphSegments()*** and ***centerlineExtraction()***.
```Python
def vesselSegmentation(mask, csv_name, view):

    ...

    # split vessel into segments sets by given bifurcations
    bifurs, segments = branch.split_into_segments(prediction, bifurs)

    # graph vessel
    graph = graphSegments(segments, bifurs)

    # extract centerline points from each segment
    vessel_centerline = centerlineExtraction(segments)

    ...

    return graph, vessel_centerline, bifurs
```
#### *Centerline of a vessel*
The process of extracting the vessel's centerline take place in the function ***get_skeleton_and_inter_points()*** from the branch_detection.py. It is called in the previously mentioned function ***split_into_segments()***.
```Python
def split_into_segments(prediction, bifurs):

    ...

    # get centerline, and bifurcation points
    thinned, points_map, points = get_skeleton_and_inter_points(prediction, bifurs)
```
In the ***get_skeleton_and_inter_points()*** the program firstly extracts a centerline of a vessel. It is done by ***guo_hall*** implementation of a thinning algorithm from the ***thinning*** package.
```Python
def get_skeleton_and_inter_points(prediction, bifurs):
    """
        Function which extracts centerline from a vessel.
        @param prediction: gray OpenCV image of a vessel
        @param bifurs: bifurcation points previously marked on an
                       image of a vessel
        @return points: bifurcation points, in the future propably aligned by
                        an algorithm
        @return thinned: centerline of a vessel
        @return points_map: bifurcation points as a gray OpenCV image
        @return points: List of bifurcation points
    """
    pred = prediction.copy()
    thinned = thinning.guo_hall_thinning(pred) / 255
```
<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/centerline_2D.png" alt="Image of the centerline"/>
</p>

#### *Splitting a vessel's centerline into segments*
After extraction of the centerline program starts to split it into segments. This is done in two function: ***get_skeleton_and_inter_points()*** where it removes bifurcations from the centerline
```Python
# Split thinned vascular tree into segments
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
points_map = cv2.dilate(points_map, kernel, iterations=1)
thinned = np.clip(thinned - points_map, 0, 1)
```

and in function ***split_into_segments()*** where segments are extracted with ***OpenCV*** functions ***find*** and ***draw Contours***.

```Python
contours, hierarchy = cv2.findContours(
    thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw segments using different grayscale
bifurs = points
sum = np.zeros(prediction.shape, dtype=np.uint8)

# list of segments
segments = []

for contour in contours:
    segment = np.zeros(thinned.shape[:2], dtype=np.uint8)

    cv2.drawContours(segment, [contour], -1, 255, -1)
    # add segment to the list
    segments.append(segment)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    segment = cv2.dilate(segment, kernel, iterations=1)
```
<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/segmentation.png" alt="Image of extracted centerline points" width="800" height="400"/>
</p>

#### *Graph of a vessel*
After retrieving segments of a vessel program approach to the constructing a graph of a vessel. It happens in the function ***graphSegments()***.<br />
Graph consists of two sets: vertices and edges. Vertices are segments, edges are bifurcation points. Graph is represented as two dimensional array with ***n*** rows and ***n*** columns where ***n*** is the number of segments. At start of the algorithm the array is filled with zeros.
```Python
def graphSegments(segments, bifurs):

    ...

    # Our graph - rows and columns represents segments. If given two segments are
    # connected we give in that place at the graph index of bifurcation point
    # from variable bifurs increased by 1. If two branches are not connected we
    # leave 0.
    graph = np.zeros((len(segments), len(segments)), dtype=np.int64)
```
For each bifurcation point function iters for each segment and check if that segment is separated by this bifurcation. It is done by checking if in the region of interests of size 5x5 pixels exist any white pixels (np.argwhere(ROI == 255).shape[0] > 0). If yes it remembers this segment.
```Python
for i, bifur in enumerate(bifurs):
    # variable which contains segments indices
    nn = []
    for j, segment in enumerate(segments):
        # we check the region of range 4 around bifurcation
        # because it was extracted as 5x5 region from orginal image
        ROI = segment[bifur[1] - 4:bifur[1] + 4, bifur[0] - 4:bifur[0] + 4]
        if np.argwhere(ROI == 255).shape[0] > 0:
            # this bifurcation belongs to this segment
            nn.append(j)
```
After enumerating all segments the function checks if the list ***nn*** which remembers segments for current bifurcation and fill the graph at this place with index of bifurcation increment by one.
```Python
# Here we add edges between two segments
for r in range(0, len(nn)):
    for t in range(0, len(nn)):
        if r != t:
            graph[nn[r], nn[t]] = i + 1
```
#### *Printed graphs from both projections*
<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/graphs.png" alt="Image of printed graphs from both projections"/>
</p>

#### *Extraction of centerline's points of a vessel*
The last step of ***vesselSegmentation()*** is the extracion of smaller number of centerline's points/pixels from each segment.<br />
The rule is very simple if segment consists of more than nine pixels the program extracts 10% of all points/pixels. Otherwise it only gets first and last points.
```Python
def centerlineExtraction(segments):

    ...

    vessel_centerline = []
    for segment in segments:
        segment_centerline = []
        # get all white pixels which represents segment centerline
        tmp = np.argwhere(segment == 255)
        num_pxl = tmp.shape[0]

        # if num of pixels is greater than 9 chose 1/10 of points
        if num_pxl > 9:
            step = int(num_pxl * 0.1)
            for z in range(1, num_pxl - 1, step):
                segment_centerline.append([tmp[z, 0], tmp[z, 1]])

        # if not, only first and last pixel
        segment_centerline.insert(0, [tmp[0, 0], tmp[0, 1]])
        segment_centerline.insert(-1, [tmp[-1, 0], tmp[-1, 1]])

        vessel_centerline.append(segment_centerline)
```
#### *Extracted centerline points from two projections*
<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/two_centerline_points.png" alt="Image of extracted centerline points" width="800" height="400"/>
</p>

<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/centerline_points.png" alt="Image of extracted centerline points on a vessel" width="800" height="400"/>
</p>

#### *Finding correspondance between segments from two projections*
After extraction of items from both projections the next step is finding correspondance between their segments. It appears in function ***segmentsMatch()***.<br />
Two segments from two different projections are equal if two rows from both graphs contains the same numbers of bifurcation points.
```Python
def segmentsMatch(graph1, graph2):

    ...

    # Array which represents order - array of pairs, each pair contains indices
    # of segments from two proejctions
    order = np.empty((graph1.shape[0], 2), dtype=np.int64)

    # Array which help with proper matching
    order_bool = np.zeros((graph1.shape[0], 1), dtype=np.int64)

    print("order before initialization:")
    print(order)

    for i, segment1 in enumerate(graph1):
        for j, segment2 in enumerate(graph2):
            # check indices of bifurcation points from two rows (segments)
            tmp1 = np.sort(segment1[np.argwhere(segment1 > 0)], axis=None)
            tmp2 = np.sort(segment2[np.argwhere(segment2 > 0)], axis=None)

            if np.array_equal(tmp1, tmp2):
                if order_bool[j] == 0:
                    order[i, 0] = i
                    order[i, 1] = j
                    order_bool[j] = 1
                    break

    return order
```
#### *Complete centerline's points with bifurcation points*
After retrieving all essential items from both masks program complete centerline's points with bifurcation points and get list of bifuractions for each segment. It is done by functions ***compose_full_centerline()*** and ***getBifursOrder()*** from a file centerline_extraction.py.
```Python
# File test.py
c1, c2 = centerline.compose_full_centerline((c1, c2), (bifurs1, bifurs2), (g1, g2))
bif_orders = centerline.getBifursOrder(g1, bifurs1, bifurs2)
```
***compose_full_centerline()*** adds bifurcation to each segment which is being separated by it. It checks if given segment is separated by any bifurcation and if yes it calculates which bifurcation is connected to which end of centerline's points.
```Python
def compose_full_centerline(cents, bifs, graphs):

    ...

    for i in range(2):
        for row in range(graphs[i].shape[0]):
            # check only matrix upper triangle for performance improvement
            for col in range(row + 1, graphs[i].shape[1]):
                bif_num = graphs[i][row, col]
                if bif_num == 0:
                    continue
                bif_point = bifs[i][bif_num - 1]
                # check whether the bifurcation point is start or the end of centerline
                for centerline_num in (row, col):
                    first_diff = np.sum(np.abs(cents[i][centerline_num][0] - bif_point))
                    last_diff = np.sum(np.abs(cents[i][centerline_num][-1] - bif_point))

                    # check if bifurcation points is already on the list
                    if first_diff == 0 or last_diff == 0:
                        continue

                    if first_diff > last_diff:
                        # it is the end
                        cents[i][centerline_num].append(bif_point.tolist())
                    else:
                        # it is start
                        cents[i][centerline_num].insert(0, bif_point.tolist())

    return cents
```
#### *Centerline points with bifurcations*
<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/add_bifurs.png" alt="Image of centerline points with bifurcations"/>
</p>

At the end of this part ***getBifursOrder()*** composes for every segment a list of bifurcation points connected with given segment.
```Python
def getBifursOrder(graph, bifurs1, bifurs2):

    ...

    bif_order1 = []
    bif_order2 = []
    for segment in graph:
        idxs = np.sort(segment[np.argwhere(segment > 0)], axis=None)
        bifs1 = [bifurs1[idx - 1] for idx in idxs]
        bifs2 = [bifurs2[idx - 1] for idx in idxs]
        bif_order1.append(np.unique(bifs1, axis=0))
        bif_order2.append(np.unique(bifs2, axis=0))

    return (bif_order1, bif_order2)
```

### 3. Reconstruction of the vessel's centerline as 3D curve

Now, when we have all 2D points extracted, we can proceed to generating our 3D points cloud  
which will represent centerline of coronary arteries.  
The main problem here is that we do not know the right correspondence between points.  
Our idea on solving a problem is to first create point cloud that contains all possible points   
and then reduce it by removing points that are considered as incorrect in a given step.  
This whole functionality is implemented in this function
```Python
def minimize_reprojection_error(conf_device: DeviceConfiguration, centerlines_points,
                                bifurs_2D, masks_paths, threshold=None)
```
which vastly uses no_corr_generate_3d_centerline_from_point_cloud function.
```Python
def no_corr_generate_3d_centerline_from_point_cloud(points, errors, cp, centerlines_points, masks_paths)
```
This two functions respectively represents implementations of Algorithm 2 and Algorithm 1 from _Point-Cloud Method for Automated 3D
Coronary Tree Reconstruction_ paper.


#### *Creating an initial point cloud from centerline points of two segments*
To create point cloud with all possible points for each point along one of the 2D centerline  
we take every points on the other 2D centerline and create new 3D point.  
Hence, if there are, respectively, M and N number of points on 2D vessel centerlines in two projection planes,  
the initial point-cloud will consider M.N number of points.  
In our that functionality is available via this function.
```Python
def get_all_possible_points_from_segments(seg_points_1: np.ndarray, seg_points_2: np.ndarray,
                                          configured_device: DeviceConfiguration, known_points=None)
```

<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/initial_point_cloud.png" alt="Image of an initial point cloud" width="500" height="500"/>
</p>

#### *Ways of reducing point cloud*

##### *Reducing point cloud by setting threshold with average diameter*
We remove points with orthogonal distance between projection lines  
less than the average maximum diameter of the vessel.
To calculate diameter we use function from borders.py module.

```Python
def get_average_diameter_size(one_segment_points: np.ndarray, mask_path)
```
When the diameter is known we can delete incorrect points.
```Python
idx_to_remove = [i for i, e in zip(range(errors.shape[0]), errors) if e > avg_max_vessel_diameter]
points, errors, cp = remove_given_indexes(points, errors, cp, idx_to_remove)
```

<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/reduced_after_diameter.png" alt="Image of a reduced point cloud" width="500" height="500"/>
</p>

##### *Reducing point cloud by limiting number of one-to-many point correspondences*
For each point along the centerline from each projection plane,  
at most _k_ nearest point correspondences are retained, the rest is removed.
It is done by
```Python
def reduce_many_to_one_point_correspondence(points: np.ndarray, errors: np.ndarray, cp: np.ndarray,
                                            point_max_num_of_correspondences=1)
```
where we can set _k_ param by giving _point_max_num_of_correspondences_ value to the function.

<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/reduced_correspondance.png" alt="Image of a reduced point cloud with cp" width="500" height="500"/>
</p>

##### *Reducing point cloud by statistical outlier removal*
It is a standard way of noise and artifacts removal that is already implemented in Open3D library.  
It removes points that are further away from their neighbors compared to the average for the point cloud.
In our program to do it we just use
```Python
def statistical_outlier_removal(points, errors, cp)
```

##### *Reducing point cloud by spline interpolation*
In this step we are trying to fit a cubic B-Spline with large smoothing factor through our point cloud.  
We start with small number of breakpoints _b0_ so it covers only first _b0_ points in point cloud.  
Then we check if there are any outliers (3 times average distance from the fitted spline curve).  
If there are we remove them from point cloud.  
Now we increase number of breakpoints by some _z_ value and  
then just repeat steps while we still have uncovered points in point cloud.  
Function responsible for doing that
```Python
def remove_points_by_spline_interpolation(points, errors, cp)
```


##### *Reducing point cloud by checking reprojection error*
Lastly, we can take our 3D point cloud and project it onto each 2D plane and compare it with our 2D input points.  
Thanks to that we can obtain reprojection error.  
Now we can use it do reduce our point cloud by deleting points with reprojection error bigger then threshold.  
We can change standard threshold value by passing _threshold_ value to the _minimize_reprojection_error_ function.  
But this parameter should not be smaller then _objective_function_value_ which is calibration error.  
This functionality is implemented inside a _minimize_reprojection_error_ function.  
This is how we calculate reprojection error  
```Python
def minimize_reprojection_error(conf_device: DeviceConfiguration, centerlines_points,
                                bifurs_2D, masks_paths, threshold=None):

...

for ray_traced_point_1, ray_traced_point_2, corr, point, idx in zip(
        ray_trace_shadow_1, ray_trace_shadow_2, cp, points, range(points_num)):
    error_1 = np.sum(np.abs(ray_traced_point_1 - shadow_1[corr[0]]))
    error_2 = np.sum(np.abs(ray_traced_point_2 - shadow_2[corr[1]]))
    if error_1 > threshold or error_2 > threshold:
        it_is_correct_point()
    else:
        it_is_incorrect_point()
...

```

where _ray_trace_shadow_1_ and _ray_trace_shadow_2_ are 3D points backprojected  
onto the same planes as _shadow_1_ and _shadow_2_.  
We can aquire this points by using _get_ray_traced_shadow_ function from _DeviceConfiguration_ class.  

```Python
ray_trace_shadow_1 = conf_device.get_ray_traced_shadow(points, f[0], o[0])
ray_trace_shadow_2 = conf_device.get_ray_traced_shadow(points, f[1], o[1])
```


##### *Final reconstruction in 3D of a vessel centerline*
The final result may look like this.
<p align="center">
<img src="https://github.com/Franek18/3D_X_ray_angiography/blob/documentation/docs/images/reconstruction1.png" alt="Image of a 3D reconstruction" width="500" height="500"/>
</p>

The whole reconstruction can be tested by calling _test_point_cloud_ function from _test.py_ module.  
```Python
def test_point_cloud(test_num, select)
```
To use it you have to call it with _test_num_ param which should be equal to number of test that you want to run and  
_select_ parameter which is boolean value that decides whether you want to select bifurcation points on images or  
you already have them selected in .csv file.  
```Python
if select:
    points_collector.save_points_from_image(mask_paths[0], csv_path[0])
    points_collector.save_points_from_image(mask_paths[1], csv_path[1])
```

For now we got 5 test prepared.  
To add new test you have to add new 3 values:  
-path to mask image  
-path to csv file with points correspondence  
-path to proper dicom file  

Example  
```Python
    elif test_num == 2:
        mask_paths = ['../res/masks/0016_01.png', '../res/masks/0023_02.png']
        csv_path = '../res/corr_points_pyramid16_01_23_02.csv'
        dicom_paths = ['../res/dicoms/exam16.dcm', '../res/dicoms/exam23.dcm']
```
