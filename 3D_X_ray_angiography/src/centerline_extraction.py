import cv2
import numpy as np
import pointsCollecting as points
import branch_detection as branch

def graphSegments(segments, bifurs):
    """
        Function which aim is to represent vessel as a graph
        where segments are vertices and bifurcations are edges.
        @param segments: list of segments from an image
        @param bifurs: list of bifurcations points
        @return graph: resulted graph
    """

    # Our graph - rows and columns represents segments. If given two segments are
    # connected we give in that place at the graph index of bifurcation point
    # from variable bifurs increased by 1. If two branches are not connected we
    # leave 0.
    graph = np.zeros((len(segments), len(segments)), dtype=np.int64)

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

        # Here we add edges between two segments
        for r in range(0, len(nn)):
            for t in range(0, len(nn)):
                if r != t:
                    graph[nn[r], nn[t]] = i + 1

    return graph

def centerlineExtraction(segments):
    """
        Function which extracts 1/10 of points of each segment's centerline
        or only 2, first and last, if number of pixels in centerline is less than
        10.
        @param segments: CA vessel's segments as 2D numpy arrays
        @return vessel_centerline: list of lists, each contain extracted points
        of segment centerline
    """

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

    return vessel_centerline

def vesselSegmentation(mask, csv_name, view):
    """
        Function which split vessel's mask into segments. Segments are divided
        by bifurcation points. These points are first extracted from the image.
        After splitting mask into segments is created a graph which show
        how segments are connected with bifurcation points.
        After that from each segment are extracted 1/10 of centerline points
        for future reconstruction.
        @param mask: Mask of an image, it is required for proper
                    working of the function
        @param csv_name: CSV file which contain x,y coordinates of
                        bifurcation points from both projections
        @param view: Which view is segmented
        @return graph: Graph of a vessel. Vertices are represented by segments
                        and edges by bifurcations.
        @return vessel_centerline: List of lists, each list contain 1/10 of centerline
                                    points from each segment
        @return bifurs: TODO - new list of bifurcations which were extracted by program
    """

    # Create object of a class for bifurs coordinates loading
    points_collector = points.PointsCollecting()

    # Read marked points
    pngs_points_list = points_collector.load_corresponding_points(csv_name)

    # Read bifurs from given view
    bifurs = pngs_points_list[view]

    # Read mask of a given image
    prediction = cv2.imread(mask, 0)

    # split vessel into segments sets by given bifurcations
    bifurs, segments = branch.split_into_segments(prediction, bifurs)

    # graph vessel
    graph = graphSegments(segments, bifurs)

    # extract centerline points from each segment
    vessel_centerline = centerlineExtraction(segments)

    #for i, segment in enumerate(segments):
    #    cv2.imshow("segment " + str(i), segment)
    #cv2.waitKey(0)

    return graph, vessel_centerline, bifurs

def showSegments(c1, c2):
    """
        Test function for debugging which display image with centerline points.
        @param c1: centerline points from the first view
        @param c2: centerline points from the second view
    """
    for seg1, seg2 in zip(c1, c2):
        segment1 = np.zeros((512, 512), dtype=np.uint8)
        segment2 = np.zeros((512, 512), dtype=np.uint8)


        for point in seg1:
            segment1[point[1], point[0]] = 255

        for point in seg2:
            segment2[point[1], point[0]] = 255

        cv2.imshow("c1", segment1)
        cv2.imshow("c2", segment2)
        cv2.waitKey(0)

def segmentsMatch(graph1, graph2):
    """
        Function which orders segments from two views. We assume that number
        of segments from two view is equal, because number of bifurcation points
        is equal. So we loop over rows (segments) in a graph1 (from a first view)
        and check which row from graph2 (from a second view) match to it.
        We check this by looking at the bifurcations inserted in the row,
        because we recognize segment by it's bifurcations.
        @param graph1: Graph of a first view
        @param graph2: Graph of a second view
        @return order: Order of segments which is an array of pairs,
                        each pair contains indices of segments from first view
                        and from the second view.
    """

    if graph1.shape[0] != graph2.shape[0]:
        print('Error 1: Number of segments should be equal!')
        return 1

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

    print("Graph1:")
    print(graph1)
    print("Graph2:")
    print(graph2)

    order = segmentsMatch(graph1, graph2)

    print("Order:")
    print(order)
    print("bifurs1:")
    print(bifurs1)
    print("bifurs2:")
    print(bifurs2)
    print("c1:")
    print(c1)
    print("c2:")
    print(c2)

    for seg in c1:
        for p in seg:
            p[0], p[1] = p[1], p[0]
    for seg in c2:
        for p in seg:
            p[0], p[1] = p[1], p[0]

    bifurs1 = np.asarray(bifurs1, dtype=np.int64)
    bifurs2 = np.asarray(bifurs2, dtype=np.int64)

    return order, c1, c2, bifurs1, bifurs2, graph1, graph2

def getBifursOrder(graph, bifurs1, bifurs2):
    """
        Function which returns two lists (from both projections) of bifurcations
        indices which separate each segment in a vessel. Each segment is separated
        only by one or two bifurcations.
        @param graph: graph of a vessel
        @param bifurs1: bifurcation points from the first projection
        @param bifurs2: bifurcation points from the second projection
        @return bif_order1: list of bifurcations for each segment from primary view
        @return bif_order2: list of bifurcations for each segment from secondary view
    """
    bif_order1 = []
    bif_order2 = []
    for segment in graph:
        idxs = np.sort(segment[np.argwhere(segment > 0)], axis=None)
        bifs1 = [bifurs1[idx - 1] for idx in idxs]
        bifs2 = [bifurs2[idx - 1] for idx in idxs]
        bif_order1.append(np.unique(bifs1, axis=0))
        bif_order2.append(np.unique(bifs2, axis=0))

    #print("bif_order1:")
    #print(bif_order1)
    #print("bif_order2:")
    #print(bif_order2)

    return (bif_order1, bif_order2)

def compose_full_centerline(cents, bifs, graphs):
    """
        Function which add bifurcation points of each segment to set of centerline
        points of this segment.
        @param cents: List of centerline points for each segment of a vessel
        @param bifs: List of bifurcation points
        @param graphs: Two graphs of a vessel from both projections
        @return cents: centerline points with bifuracation points for each segment
    """
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
