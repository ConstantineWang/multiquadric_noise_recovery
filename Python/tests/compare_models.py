import numpy as np
from EMS.EMS_recovery import EMS_recovery
from EMS.utilities import read_ply, showPoints, uniformSampledSuperellipse
from mayavi import mlab
import timeit
import argparse
import sys
from multiquadric_test import hierarchical_ems
from sklearn.metrics import mean_squared_error
import os
from scipy.spatial import cKDTree
from collections import defaultdict
from sklearn.decomposition import PCA

def calculate_fitting_error(points, quadric):
    """Calculate the fitting error of a point to a superquadric"""
    print("Point cloud maximum and minimum values:", np.max(points), np.min(points))
    distances = []
    for point in points:
        try:
            # Calculate the distance from the point to the superquadric
            point_rot = point @ quadric.RotM
            point_rot = point_rot - quadric.translation @ quadric.RotM
            point_rot = point_rot / quadric.scale
            
            # Calculate the superquadric equation
            x, y, z = point_rot
            r = np.sqrt(x**2 + y**2)
            
            # Calculate the distance from the point to the surface
            epsilon1 = max(quadric.shape[0], 1e-6)  # Avoid division by zero
            epsilon2 = max(quadric.shape[1], 1e-6)  # Avoid division by zero
            f = (r/epsilon1)**(2/epsilon2) + (z/epsilon1)**(2/epsilon2) - 1
            if np.isfinite(f):
                distances.append(abs(f))
        except Exception as e:
            continue
    
    if not distances:
        return float('nan'), float('nan')
    return np.mean(distances), np.std(distances)

def is_valid_quadric(q):
    return np.all(np.isfinite(q.shape)) and np.all(q.shape > 0) and \
           np.all(np.isfinite(q.scale)) and np.all(q.scale > 0)

def model_to_points(quadric, num_points=1000):
    # Sample points on the superquadric
    # Here we use the sampling logic of showSuperquadric, but without drawing, just return points
    epsilon1 = max(quadric.shape[0], 1e-6)
    epsilon2 = max(quadric.shape[1], 1e-6)
    scale = quadric.scale
    # Sampling parameters
    threshold = 1e-2
    num_limit = num_points
    arclength = 0.02
    point_eta = uniformSampledSuperellipse(epsilon1, [1, scale[2]], threshold, num_limit, arclength)
    point_omega = uniformSampledSuperellipse(epsilon2, [scale[0], scale[1]], threshold, num_limit, arclength)
    sampled_points = []
    for m in range(point_omega.shape[1]):
        for n in range(point_eta.shape[1]):
            point_temp = np.zeros(3)
            point_temp[0:2] = point_omega[:, m] * point_eta[0, n]
            point_temp[2] = point_eta[1, n]
            point_temp = quadric.RotM @ point_temp + quadric.translation
            sampled_points.append(point_temp)
    return np.array(sampled_points)

def cloud_to_model_rmse(point_cloud, quadric):
    model_points = model_to_points(quadric)
    tree = cKDTree(model_points)
    dists, _ = tree.query(point_cloud)
    return np.mean(dists), np.std(dists)

def model_coverage(point_cloud, quadric, threshold=0.1):
    model_points = model_to_points(quadric)
    tree = cKDTree(model_points)
    dists, _ = tree.query(point_cloud)
    covered = dists < threshold
    return covered, np.sum(covered) / len(point_cloud)

def pca_coverage(points):
    if len(points) < 2:
        return float('nan')
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.explained_variance_ratio_[0]

def compare_models(point_cloud, visualize=False, show='all'):
    """Compare the performance of the original and improved models"""
    results = {}
    
    # Run the original model
    print("Running the original model...")
    start = timeit.default_timer()
    sq_recovered, p = EMS_recovery(point_cloud)
    stop = timeit.default_timer()
    results['original'] = {
        'runtime': (stop - start) * 1000,  # Convert to milliseconds
        'quadric': sq_recovered,
        'inlier_ratio': np.mean(p > 0.5),  # Assume points with probability greater than 0.5 are inliers
        'error_mean': None,
        'error_std': None,
        'rmse_mean': None,
        'rmse_std': None,
        'coverage': None,
        'outlier_ratio': None,
        'pca_coverage': None
    }
    
    # Calculate the fitting error of the original model
    error_mean, error_std = calculate_fitting_error(point_cloud, sq_recovered)
    results['original']['error_mean'] = error_mean
    results['original']['error_std'] = error_std
    
    # Original model RMSE
    rmse_mean, rmse_std = cloud_to_model_rmse(point_cloud, sq_recovered)
    results['original']['rmse_mean'] = rmse_mean
    results['original']['rmse_std'] = rmse_std
    
    # Statistics for the original model: coverage and outlier ratio
    covered_mask, coverage = model_coverage(point_cloud, sq_recovered)
    results['original']['coverage'] = coverage
    results['original']['outlier_ratio'] = 1.0 - results['original']['inlier_ratio']
    results['original']['pca_coverage'] = pca_coverage(point_cloud[covered_mask])
    
    # Run the improved model
    print("Running the improved model...")
    start = timeit.default_timer()
    point_seg, point_outlier, list_quadrics = hierarchical_ems(
        point_cloud,
        OutlierThreshold=0.045,
        MinOutlierRatio=0.135,
        MinPoints=120,
        Eps=2.25
    )
    stop = timeit.default_timer()
    
    # Calculate the fitting error of the improved model
    total_error = []
    for quadric in list_quadrics:
        error_mean, _ = calculate_fitting_error(point_cloud, quadric)
        if error_mean != float('inf'):
            total_error.append(error_mean)
    
    results['improved'] = {
        'runtime': (stop - start) * 1000,  # Convert to milliseconds
        'num_quadrics': len(list_quadrics),
        'quadrics': list_quadrics,
        'error_mean': np.mean(total_error) if total_error else float('inf'),
        'error_std': np.std(total_error) if total_error else float('inf'),
        'rmse_mean': None,
        'rmse_std': None
    }
    
    # Statistics for each submodel: coverage, fitting error, RMSE
    improved_coverages = []
    improved_outlier_mask = np.ones(len(point_cloud), dtype=bool)
    improved_errors = []
    improved_rmses = []
    improved_pca_coverages = []
    for quadric in results['improved']['quadrics']:
        covered_mask, coverage = model_coverage(point_cloud, quadric)
        improved_coverages.append(coverage)
        improved_outlier_mask = improved_outlier_mask & (~covered_mask)
        error_mean, error_std = calculate_fitting_error(point_cloud[covered_mask], quadric)
        improved_errors.append((error_mean, error_std))
        rmse_mean, rmse_std = cloud_to_model_rmse(point_cloud[covered_mask], quadric)
        improved_rmses.append((rmse_mean, rmse_std))
        improved_pca_coverages.append(pca_coverage(point_cloud[covered_mask]))
    improved_outlier_ratio = np.sum(improved_outlier_mask) / len(point_cloud)
    
    # Improved model RMSE (if multiple models, take the minimum RMSE)
    if improved_rmses:
        best_rmse = min(improved_rmses, key=lambda x: x[0])
        results['improved']['rmse_mean'] = best_rmse[0]
        results['improved']['rmse_std'] = best_rmse[1]
    else:
        results['improved']['rmse_mean'] = float('nan')
        results['improved']['rmse_std'] = float('nan')
    
    # Print comparison results
    print("\n=== Model Comparison Results ===")
    print(f"Point cloud size: {len(point_cloud)} points")
    print("\nOriginal Model:")
    print(f"Runtime: {results['original']['runtime']:.2f} ms")
    print(f"Inlier ratio: {results['original']['inlier_ratio']:.2%}")
    print(f"Outlier ratio: {results['original']['outlier_ratio']:.2%}")
    print(f"Fitting error: {results['original']['error_mean']:.6f} ± {results['original']['error_std']:.6f}")
    print(f"RMSE: {results['original']['rmse_mean']:.6f} ± {results['original']['rmse_std']:.6f}")
    print(f"Point cloud coverage: {results['original']['coverage']:.2%}")
    print(f"PCA principal component coverage: {results['original']['pca_coverage']:.2%}")
    print("Shape parameters:", results['original']['quadric'].shape)
    print("Scale parameters:", results['original']['quadric'].scale)
    print("Translation parameters:", results['original']['quadric'].translation)
    print("Euler angles:", results['original']['quadric'].euler)
    
    print("\nImproved Model:")
    print(f"Runtime: {results['improved']['runtime']:.2f} ms")
    print(f"Number of detected superquadrics: {results['improved']['num_quadrics']}")
    for i, quadric in enumerate(results['improved']['quadrics']):
        print(f"Submodel {i+1}:")
        print(f"  Point cloud coverage: {improved_coverages[i]:.2%}")
        print(f"  Fitting error: {improved_errors[i][0]:.6f} ± {improved_errors[i][1]:.6f}")
        print(f"  RMSE: {improved_rmses[i][0]:.6f} ± {improved_rmses[i][1]:.6f}")
        print(f"  PCA principal component coverage: {improved_pca_coverages[i]:.2%}")
        print(f"  Shape parameters: {quadric.shape}")
        print(f"  Scale parameters: {quadric.scale}")
        print(f"  Translation parameters: {quadric.translation}")
        print(f"  Euler angles: {quadric.euler}")
    print(f"Outlier ratio: {improved_outlier_ratio:.2%}")
    
    # Visualize results
    if visualize:
        if show in ['all', 'original']:
            fig1 = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
            mlab.title('Original Model', size=0.5)
            if is_valid_quadric(results['original']['quadric']):
                results['original']['quadric'].showSuperquadric(arclength=0.2)
            else:
                print("Original model parameters are invalid, skipping visualization")
            showPoints(point_cloud, scale_factor=0.1)
            mlab.show()
        if show in ['all', 'improved']:
            fig2 = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
            mlab.title('Improved Model', size=0.5)
            for quadric in results['improved']['quadrics']:
                if is_valid_quadric(quadric):
                    quadric.showSuperquadric(arclength=0.2)
                else:
                    print("Improved model parameters are invalid, skipping visualization")
            showPoints(point_cloud, scale_factor=0.1)
            mlab.show()
    
    return results

def main(argv):
    parser = argparse.ArgumentParser(
        description='Compare the performance of the original and improved models')
    
    parser.add_argument(
        'path_to_data',
        help='Path to the point cloud file (*.ply)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the results'
    )
    parser.add_argument(
        '--show',
        choices=['all', 'original', 'improved'],
        default='all',
        help='Which model to visualize (all, original, improved)'
    )
    
    args = parser.parse_args(argv)
    
    print('----------------------------------------------------')
    print('Loading point cloud file: ', args.path_to_data, '...')
    point_cloud = read_ply(args.path_to_data)
    print('Point cloud loaded')
    print('----------------------------------------------------')
    
    results = compare_models(point_cloud, args.visualize, args.show)

if __name__ == "__main__":
    main(sys.argv[1:]) 