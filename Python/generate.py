import numpy as np

def generate_superquadric_points(a1, a2, a3, e1, e2, n_points=1000):
    """Generate points on superquadric surface"""
    u = np.random.uniform(-np.pi/2, np.pi/2, n_points)
    v = np.random.uniform(-np.pi, np.pi, n_points)
    
    x = a1 * np.sign(np.cos(u)) * (np.abs(np.cos(u)) ** e1) * np.sign(np.cos(v)) * (np.abs(np.cos(v)) ** e2)
    y = a2 * np.sign(np.cos(u)) * (np.abs(np.cos(u)) ** e1) * np.sign(np.sin(v)) * (np.abs(np.sin(v)) ** e2)
    z = a3 * np.sign(np.sin(u)) * (np.abs(np.sin(u)) ** e1)
    
    return np.column_stack([x, y, z])

def save_ply(points, filename):
    """Save points to PLY format"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

# Generate test cases
test_cases = [
    ("cube.ply", [2, 2, 2, 0.1, 0.1]),           # Cube-like
    ("sphere.ply", [2, 2, 2, 1.0, 1.0]),         # Sphere
    ("cylinder.ply", [2, 2, 3, 1.0, 0.1]),       # Cylinder
    ("rounded_box.ply", [3, 2, 1, 0.5, 0.5]),    # Rounded box
    ("pillow.ply", [2, 2, 1, 2.0, 2.0]),         # Pillow shape
]

for filename, params in test_cases:
    points = generate_superquadric_points(*params)
    
    # Add noise for realism
    noise = np.random.normal(0, 0.05, points.shape)
    points_noisy = points + noise
    
    # Add some outliers
    n_outliers = len(points) // 20
    outliers = np.random.uniform(-5, 5, (n_outliers, 3))
    final_points = np.vstack([points_noisy, outliers])
    
    save_ply(final_points, filename)
    print(f"Generated {filename}: {len(final_points)} points")
