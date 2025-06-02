import numpy as np
from EMS.EMS_recovery import EMS_recovery
from EMS.utilities import read_ply, showPoints
from mayavi import mlab
from sklearn.cluster import DBSCAN
from collections import defaultdict


def hierarchical_ems(
    points,
    outlier_ratio=0.9,
    max_em_iterations=20,
    em_tolerance=1e-3,
    em_relative_tolerance=2e-1,
    max_optimization_iterations=2,
    sigma=0.3,
    max_switches=2,
    adaptive_upper_bound=True,
    rescale=False,
    max_depth=5,
    dbscan_eps=1.7,
    dbscan_min_points=60,
    inlier_threshold=0.1,
    min_inlier_ratio=0.8
):
    segments = defaultdict(list)
    outliers = defaultdict(list)
    segments[0] = [points]
    quadrics = []
    
    for depth in range(max_depth):
        if not segments[depth]:
            break
            
        for segment in segments[depth]:
            quadric, probabilities = EMS_recovery(
                segment,
                outlier_ratio,
                max_em_iterations,
                em_tolerance,
                em_relative_tolerance,
                max_optimization_iterations,
                sigma,
                max_switches,
                adaptive_upper_bound,
                rescale
            )
            
            quadrics.append(quadric)
            
            inlier_mask = probabilities > inlier_threshold
            outlier_mask = ~inlier_mask
            
            segment_outliers = segment[outlier_mask]
            
            if probabilities.sum() < min_inlier_ratio * len(segment):
                clusters = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_points).fit(segment_outliers)
                unique_labels = set(clusters.labels_) - {-1}
                
                for label in unique_labels:
                    cluster_points = segment_outliers[clusters.labels_ == label]
                    segments[depth + 1].append(cluster_points)
                
                noise_points = segment_outliers[clusters.labels_ == -1]
                if len(noise_points) > 0:
                    outliers[depth].append(noise_points)
            elif len(segment_outliers) > 0:
                outliers[depth].append(segment_outliers)
    
    return dict(segments), dict(outliers), quadrics


def visualize_quadrics(quadrics, point_cloud, arclength=0.2, point_scale=0.001):
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    
    for quadric in quadrics:
        quadric.showSuperquadric(arclength=arclength)
    
    showPoints(point_cloud, scale_factor=point_scale)
    mlab.show()
    
    return fig


if __name__ == "__main__":
    point_cloud = read_ply("beer_bottle.ply")
    
    segments, outliers, quadrics = hierarchical_ems(
        point_cloud,
        dbscan_eps=1.7,
        dbscan_min_points=60
    )
    
    print(f"Generated {len(quadrics)} quadrics across {len(segments)} layers")
    
    visualize_quadrics(quadrics, point_cloud)