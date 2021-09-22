import numpy as np
from decimal import Decimal

def OSB(t1, t2, jumpcost, type, warpwin = None, queryskip = None, targetskip = None):
    """
        This fuction is symmetric and allows the skipping of the first and last
        elements of both the target and query sequences.
        @param t1: the first (query) time series. A row vector.
        @param t2: the second (target) time series. A row vector.
        @param jumpcost: the cost of jumps in t1 or t2 used to balance
                         the similarity of elements.
                         The number of jumps is mulitplied by jumpcost.
                         If not given or equal -1 is computed automatically.
        @param type: type == 0, means OSB for corresponding frames (CF),
                         type == 1  or not present means OSB
                         for corresponding points (CP).
        @param warpwin: the warping window, same as that used in Dynamic Time Warping
        @param queryskip: sets a restriction on how many consecutive elements
                          of sequence t1 can be skipped in the matching.
        @param targetskip: sets a restriction on how many consecutive elements
                           of sequence t2 can be skiped in the matching.
        best set  warpwin = queryskip = targetskip
        @return pathcost: the cost of the cheapest matching path between t1 and t2;
                          it is a combination of distances between corresponding
                          elements and the panelties for jumping (skipping)
        @return indxrow: the index of elements in a substring of t1 that best match t2
        @return indxcol: the index of elements in a substring of t2 that best match t1
        @return d: squared Euclidean distance of correspndoning elements (no jump penalty)
    """

    m = len(t1)
    n = len(t2)
    # if warpwin is not given then
    if warpwin == None:
        warpwin = Decimal('inf')

    # if targetskip is not given then
    if targetskip == None:
        targetskip = Decimal('inf')

    # if queryskip is not given then
    if queryskip == None:
        queryskip = Decimal('inf')

    matx = np.zeros([m, n], dtype=int)
    # create a distance matrix, each entry i,j contains squared Euclidean distance of i in t1 and j in t2
    for r in range(0, m):
        for i in range(0, n):
            matx[r, i] = (t2[i] - t1[r])**2

    #print(matx)
    # computing the jumpcost, it is symmetric
    if jumpcost == None or jumpcost == -1:
        statmatx = np.min(matx.T, axis=0)
        statmatx2 = np.min(matx, axis=0)
        jumpcost = np.minimum(np.mean(statmatx) + np.std(statmatx, ddof=1), np.mean(statmatx2) + np.std(statmatx2, ddof=1))

    matxE = np.ones([m + 2, n + 2], dtype=int) * Decimal('inf')   # we add one row and one column to beginning and end of matx
                                                    # to ensure that we can skip the first and last elements
    matxE[1:m+1, 1:n+1] = matx
    matxE[0, 0] = 0
    matxE[m + 1, n + 1] = 0
    print(matxE)
    pathcost, indxrow, indxcol = findpathDAG(matxE, warpwin, queryskip, targetskip, jumpcost)
    # we normalize path cost
    pathcost = pathcost / len(indxrow)
    sum = 0
    for r in range(0, len(indxrow)):
        sum += (t1[indxrow[r]] - t2[indxcol[r]])**2

    d = np.sqrt(sum) / len(indxrow)
    #d = np.sqrt(np.sum((t1[indxrow] - t2[indxcol]) ^ 2)) / len(indxrow)

    return pathcost, indxrow, indxcol, jumpcost, d

def findpathDAG(matx, warpwin, queryskip, targetskip, jumpcost):
    """
        Function which finds path in DAG with min cost.
        @param matx: the extended difference matrix
        @param warpwin: the warping window, same as that used in Dynamic Time Warping
        @param queryskip: sets a restriction on how many consecutive elements
                          of sequence t1 can be skipped in the matching.
        @param targetskip: sets a restriction on how many consecutive elements
                           of sequence t2 can be skiped in the matching.
        @param jumpcost: the cost of jumps in t1 or t2 used to balance
                         the similarity of elements.
                         The number of jumps is mulitplied by jumpcost.
        @return pathcost: the cost of cheapest path
        @return indxrow, indxcol: indices of the cheapest path
    """
    m, n = matx.shape                                       # this matx=matxE, thus it has one row an column more than the original matx above
    weight = np.ones([m, n], dtype=np.float64) * float('inf')    # the weight of the actually cheapest path
    camefromcol = np.zeros([m, n], dtype=int)               # the index of the parent col where the cheapest path came from
    camefromrow = np.zeros([m, n], dtype=int)               # the index of the parent row where the cheapest path came from

    weight[0, :] = matx[0, :]                               # initialize first row
    weight[:, 0] = matx[:, 0]                               # initialize first column


    for i in range(0, m - 1):                               # index over rows
        for j in range(0, n - 1):                           # index over columns
            if np.abs(i - j) <= warpwin:                    # difference between i and j must be smaller than warpwin
                stoprowjump = np.min([m, i + 1 + queryskip])
                for rowjump in range(i + 1, stoprowjump):   # second index over rows
                    stopk = np.min([n, j + 1 + targetskip])
                    for k in range(j + 1, stopk):           # second index over columns
                        newweight = ( weight[i, j] +  np.float(matx[rowjump, k]) + ((rowjump - i - 1) + (k-j-1)) * jumpcost) # we favor shorter jumps by multiplying by jummpcost
                        if weight[rowjump, k] > newweight:
                            weight[rowjump, k] = newweight
                            camefromrow[rowjump, k] = i
                            camefromcol[rowjump, k] = j

    # collecting the indices of points on the cheapest path
    pathcost = weight[m - 1, n - 1]   # pathcost: minimal value
    mincol = n - 1
    minrow = m - 1
    indxcol = []
    indxrow = []
    while minrow > 0 and mincol > 0:
        indxcol.insert(0, mincol)
        indxrow.insert(0, minrow)
        mincoltemp = camefromcol[minrow, mincol]
        minrow = int(camefromrow[minrow, mincol])
        mincol = int(mincoltemp)

    indxcol = indxcol[0:len(indxcol) - 1]
    indxrow = indxrow[0:len(indxrow) - 1]
    indxcol[:] = [i - 1 for i in indxcol]
    indxrow[:] = [i - 1 for i in indxrow]

    return pathcost, indxrow, indxcol

"""
# start here
Test of OSB function
-------------------------------------------------------------------------
"""
if __name__ == "__main__":
    t1 = [1, 2, 8, 6, 8]
    t2 = [1, 2, 9, 15, 3, 5, 9]
    # t1=[ 1 2 8 12 6 8.5]; t2=[ 1 2 9  3 3  5.5 9];
    pathcost, indxrow, indxcol, jumpcost, d = OSB(t1, t2, -1, 1)
    print("Path cost: " + str(pathcost))
    print("indxrow, indxcol: " + str(indxrow) + ", " + str(indxcol))
    print("Jump cost: " + str(jumpcost))
    print("Squared Euclidean distance of correspndoning elements: " + str(d))