import numpy as np

def generate_cup(radius=2, height=3, wall_thickness=0.2, n_points=2000):
    """Generate cup-like shape (hollow cylinder)"""
    points = []
    
    # Outer surface
    for _ in range(n_points // 2):
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(0, height)
        
        # Slight tapering
        r_factor = 1 + 0.1 * (z / height)
        x = radius * r_factor * np.cos(theta)
        y = radius * r_factor * np.sin(theta)
        points.append([x, y, z])
    
    # Inner surface
    inner_radius = radius - wall_thickness
    for _ in range(n_points // 4):
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(wall_thickness, height)
        
        r_factor = 1 + 0.1 * (z / height)
        x = inner_radius * r_factor * np.cos(theta)
        y = inner_radius * r_factor * np.sin(theta)
        points.append([x, y, z])
    
    # Bottom surface
    for _ in range(n_points // 4):
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, radius)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(0, wall_thickness)
        points.append([x, y, z])
    
    return np.array(points)

def generate_vase(base_radius=1.5, top_radius=1, height=4, neck_height=1, n_points=2000):
    """Generate vase with narrow neck"""
    points = []
    
    for _ in range(n_points):
        z = np.random.uniform(0, height)
        theta = np.random.uniform(0, 2*np.pi)
        
        if z < height - neck_height:
            # Body: smooth transition from base to neck
            t = z / (height - neck_height)
            radius = base_radius * (1 - 0.3 * t) + 0.3 * base_radius * np.sin(np.pi * t)
        else:
            # Neck: narrow cylinder
            radius = top_radius
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    
    return np.array(points)

def generate_bowl(radius=3, depth=2, n_points=1500):
    """Generate bowl (partial ellipsoid)"""
    points = []
    
    for _ in range(n_points):
        # Spherical coordinates, but only upper hemisphere
        phi = np.random.uniform(0, np.pi/2)  # 0 to 90 degrees
        theta = np.random.uniform(0, 2*np.pi)
        
        # Ellipsoidal scaling
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = depth * (1 - np.cos(phi))  # Invert and scale
        
        points.append([x, y, z])
    
    return np.array(points)

def generate_bottle(body_radius=1.5, neck_radius=0.5, total_height=5, neck_height=1.5, n_points=2000):
    """Generate bottle with body and neck"""
    points = []
    
    for _ in range(n_points):
        z = np.random.uniform(0, total_height)
        theta = np.random.uniform(0, 2*np.pi)
        
        if z < total_height - neck_height:
            # Body: ellipsoidal with tapering
            t = z / (total_height - neck_height)
            radius = body_radius * (1 - 0.2 * t * t)  # Quadratic tapering
        else:
            # Neck: transition + cylinder
            t = (z - (total_height - neck_height)) / neck_height
            if t < 0.3:  # Transition region
                radius = body_radius * 0.8 * (1 - t/0.3) + neck_radius * (t/0.3)
            else:  # Neck cylinder
                radius = neck_radius
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    
    return np.array(points)

def generate_mug(radius=2, height=3, handle_width=0.3, n_points=2500):
    """Generate mug with handle"""
    points = []
    
    # Main cylinder body
    for _ in range(int(n_points * 0.7)):
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(0, height)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    
    # Handle (torus section)
    handle_center_x = radius + 0.8
    handle_center_y = 0
    handle_radius = 0.6
    
    for _ in range(int(n_points * 0.2)):
        # Torus handle
        u = np.random.uniform(np.pi/4, 3*np.pi/4)  # Partial torus
        v = np.random.uniform(0, 2*np.pi)
        
        x = handle_center_x + handle_radius * np.cos(u)
        y = handle_center_y + handle_radius * np.sin(u) * np.cos(v)
        z = height/2 + handle_width * np.sin(v) + handle_radius * np.sin(u) * 0.3
        
        points.append([x, y, z])
    
    # Bottom
    for _ in range(int(n_points * 0.1)):
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, radius)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-0.1, 0.1)
        points.append([x, y, z])
    
    return np.array(points)

def generate_teapot_body(body_radius=2.5, height=2.5, n_points=2000):
    """Generate teapot main body (without spout/handle)"""
    points = []
    
    for _ in range(n_points):
        # Modified superquadric for teapot body
        u = np.random.uniform(-np.pi/2, np.pi/2)
        v = np.random.uniform(-np.pi, np.pi)
        
        # Teapot body shape (rounded but wider at middle)
        e1, e2 = 1.5, 1.2  # Shape parameters
        
        x = body_radius * np.sign(np.cos(u)) * (np.abs(np.cos(u)) ** e1) * np.sign(np.cos(v)) * (np.abs(np.cos(v)) ** e2)
        y = body_radius * np.sign(np.cos(u)) * (np.abs(np.cos(u)) ** e1) * np.sign(np.sin(v)) * (np.abs(np.sin(v)) ** e2)
        z = height * np.sign(np.sin(u)) * (np.abs(np.sin(u)) ** e1)
        
        # Flatten bottom slightly
        if z < -height * 0.7:
            z = -height * 0.7 + 0.1 * (z + height * 0.7)
            
        points.append([x, y, z])
    
    return np.array(points)

def add_noise_and_outliers(points, noise_std=0.05, outlier_ratio=0.05):
    """Add realistic noise and outliers"""
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, points.shape)
    noisy_points = points + noise
    
    # Add outliers
    n_outliers = int(len(points) * outlier_ratio)
    bbox_min, bbox_max = np.min(points, axis=0), np.max(points, axis=0)
    outliers = np.random.uniform(bbox_min - 1, bbox_max + 1, (n_outliers, 3))
    
    return np.vstack([noisy_points, outliers])

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

# Generate complex shape test cases
complex_shapes = [
    ("coffee_cup.ply", generate_cup()),
    ("flower_vase.ply", generate_vase()),
    ("cereal_bowl.ply", generate_bowl()),
    ("wine_bottle.ply", generate_bottle()),
    ("coffee_mug.ply", generate_mug()),
    ("teapot_body.ply", generate_teapot_body()),
    ("wide_vase.ply", generate_vase(base_radius=2.5, top_radius=1.8, height=3.5)),
    ("shallow_bowl.ply", generate_bowl(radius=4, depth=1.5)),
    ("beer_bottle.ply", generate_bottle(body_radius=1.2, neck_radius=0.4, total_height=6)),
    ("soup_bowl.ply", generate_bowl(radius=2.5, depth=2.5)),
]

for filename, points in complex_shapes:
    final_points = add_noise_and_outliers(points)
    save_ply(final_points, filename)
    print(f"Generated {filename}: {len(final_points)} points")

print("\nGenerated complex shapes suitable for superquadric fitting!")
print("These objects have superquadric-like base shapes with realistic variations.")
