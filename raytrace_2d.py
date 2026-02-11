import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def trace_ray(n_array, dx, dy, start_row, ds=0.1):
    """
    Traces a ray through a 2D numpy array.
    n_array: 2D numpy array [rows, cols] (y, x)
    dx, dy: physical size of each cell
    start_row: Nth element from the top (0-indexed)
    ds: step size for the integration

    Suppose that a light ray shines into a nonuniform media where
        n(x,y) is the index of refraction,
        R is the position vector,
        v is the light velocity,
        and ds is a small increment along the path of the light ray.
    What differential equation describes the path of the light ray?
    Answer: The Ray Equation: d/ds(n * dR/ds) = grad n

    This can be inverted to get: dR = ((grad n)*ds / n ) * ds

    """
    rows, cols = n_array.shape
    y_max, x_max = rows * dy, cols * dx
    
    # 1. Setup Interpolation for n and its Gradient
    y_coords = np.arange(rows) * dy
    x_coords = np.arange(cols) * dx
    interp_n = RegularGridInterpolator((y_coords, x_coords), n_array, bounds_error=False, fill_value=None)
    
    # Calculate gradients
    gy, gx = np.gradient(n_array, dy, dx)
    interp_gx = RegularGridInterpolator((y_coords, x_coords), gx, bounds_error=False, fill_value=None)
    interp_gy = RegularGridInterpolator((y_coords, x_coords), gy, bounds_error=False, fill_value=None)

    # 2. Initial Conditions
    # Start at the left edge (x=0) at the center of the specified row
    x, y = 0.0, start_row * dy + (dy / 2)
    
    # Direction: initially horizontal (angle = 0)
    # The vector v = n * dr/ds. For horizontal, dr/ds = (1, 0)
    n0 = interp_n((y, x))
    vx, vy = n0 * 1.0, n0 * 0.0
    
    path = [[x, y]]

    # 3. Integration (Euler Method)
    while 0 <= x < x_max and 0 <= y < y_max:
        # Get local gradient
        grad_x = interp_gx((y, x))
        grad_y = interp_gy((y, x))
        
        # Update momentum vector: dv/ds = grad(n)
        vx += grad_x * ds
        vy += grad_y * ds
        
        # Normalize to find the new step in dr/ds
        # Since |dr/ds| = 1, the magnitude of v must be n
        current_n = interp_n((y, x))
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Update position: dr = (v/n) * ds
        x += (vx / v_mag) * ds
        y += (vy / v_mag) * ds
        
        path.append([x, y])
        
    return np.array(path)

# --- Example Usage ---
# Create a sample index gradient
# Experiment with this

step = 20
grid_n = np.ones((100, 200))
for iy in range(100):
    for ix in range(200):
        grid_n[iy, ix] = 1.0
        for center_y in range(0,100,step):
            for center_x in range(50,150,step):
                if center_y != 0: #45:
                    grid_n[iy, ix]  += 0.3 * np.exp(-((iy - center_y)**2 + (ix -center_x)**2) / 40)



paths_list = []
for row_num in range(5,100,5):
    q = trace_ray(grid_n, dx=1.0, dy=1.0, start_row=row_num, ds=0.5)
    paths_list.append(q)

# Visualization
plt.imshow(grid_n, extent=[0, 200, 100, 0], cmap='viridis')

for path_ix in paths_list:
    plt.plot(path_ix[:, 0], path_ix[:, 1], 'r-')

plt.colorbar(label='Refractive Index n')
plt.legend()
plt.show()
