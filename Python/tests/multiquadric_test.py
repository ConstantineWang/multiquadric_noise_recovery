import numpy as np
from EMS.EMS_recovery import EMS_recovery
from EMS.utilities import read_ply, showPoints
from mayavi import mlab
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import argparse
import sys

def hierarchical_ems(
    point, 
    OutlierRatio=0.9,
    MaxIterationEM=20,
    ToleranceEM=1e-3,
    RelativeToleranceEM=2e-1,
    MaxOptiIterations=2,
    Sigma=0.3,
    MaxiSwitch=2,
    AdaptiveUpperBound=True,
    Rescale=False,
    MaxLayer=5,
    Eps=2.25,
    MinPoints=120,
    OutlierThreshold=0.045,
    MinOutlierRatio=0.135,
):
    point_seg = {i: [] for i in range(MaxLayer + 1)}
    point_outlier = {i: [] for i in range(MaxLayer + 1)}
    point_seg[0] = [point]
    list_quadrics = []
    
    ems_params = (OutlierRatio, MaxIterationEM, ToleranceEM, RelativeToleranceEM,
                  MaxOptiIterations, Sigma, MaxiSwitch, AdaptiveUpperBound, Rescale)
    
    for layer in range(MaxLayer):
        for segment_idx in range(len(point_seg[layer])):
            current_points = point_seg[layer][segment_idx]
            
            if len(current_points) < MinPoints:
                continue
                
            try:
                x_raw, p_raw = EMS_recovery(current_points, *ems_params)
                list_quadrics.append(x_raw)
                
                inlier_mask = p_raw > OutlierThreshold
                inliers = current_points[inlier_mask]
                outliers = current_points[~inlier_mask]
                
                if len(inliers) > 0:
                    point_seg[layer][segment_idx] = inliers
                
                outlier_ratio = len(outliers) / len(current_points)
                if len(outliers) > MinPoints and outlier_ratio > MinOutlierRatio:
                    if should_segment_outliers(outliers, Eps, MinPoints):
                        segments = cluster_points(outliers, Eps, MinPoints)
                        point_seg[layer + 1].extend(segments)
                    else:
                        point_outlier[layer].append(outliers)
                        
            except Exception as e:
                print(f"Failed to fit quadric: {e}")
                if layer < MaxLayer - 1:
                    point_seg[layer + 1].append(current_points)
    
    return point_seg, point_outlier, list_quadrics

def should_segment_outliers(outliers, eps, min_points):
    if len(outliers) < min_points * 2:
        return False
    
    pca = PCA(n_components=3)
    pca.fit(outliers)
    variance_ratio = pca.explained_variance_ratio_[0]
    
    return variance_ratio < 0.88

def cluster_points(points, eps, min_points):
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = clustering.labels_
    
    segments = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = points[labels == label]
        if len(cluster) >= min_points:
            segments.append(cluster)
    
    noise = points[labels == -1]
    if len(noise) >= min_points:
        segments.append(noise)
    
    return segments if segments else [points]

def main(argv):
    parser = argparse.ArgumentParser(
        description='Hierarchical EMS: Probabilistic Recovery of multiple superquadric surfaces from a point cloud file (*.ply).'
    )

    parser.add_argument(
        'path_to_data',
        help='Path to the point cloud file (*.ply).'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the recovered superquadrics and the input point cloud.'
    )

    parser.add_argument(
        '--runtime',
        action='store_true',
        help='Show the runtime.'
    )

    parser.add_argument(
        '--result',
        action='store_true',       
        help='Print the recovered superquadric parameters.'
    )

    args = parser.parse_args(argv)
    
    print('----------------------------------------------------')
    print('Loading point cloud from: ', args.path_to_data, '...')
    point_cloud = read_ply(args.path_to_data)
    print('Point cloud loaded.')
    print('----------------------------------------------------')

    import timeit
    start = timeit.default_timer()
    point_seg, point_outlier, list_quadrics = hierarchical_ems(
        point_cloud,
        OutlierThreshold=0.045,
        MinOutlierRatio=0.135,
        MinPoints=120,
        Eps=2.25
    )
    stop = timeit.default_timer()

    if args.runtime:
        print('Runtime: ', (stop - start) * 1000, 'ms')
    print('----------------------------------------------------')

    if args.result:
        for i, quadric in enumerate(list_quadrics):
            print(f'Quadric {i+1}:')
            print('shape =', quadric.shape)
            print('scale =', quadric.scale)
            print('euler =', quadric.euler)
            print('translation =', quadric.translation)
            print('----------------------------------------------------')

    if args.visualize:
        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        for quadric in list_quadrics:
            quadric.showSuperquadric(arclength=0.2)
        showPoints(point_cloud, scale_factor=0.1)
        mlab.show()

if __name__ == "__main__":
    main(sys.argv[1:])