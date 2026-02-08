# halftone stitch

# import an image, convert to numpy array, and use the grayscale brightness
# to determine the stitch density

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import pandas as pd
import pyembroidery

def get_zigzag_points(cell_width, cell_height, excursion_fraction, stitch_distance, offset_x = 0.0, offset_y = 0.0):

    # A series of line segments connects the five points
    # [(0, 0), (0.25*cell_width, excursion_fraction*cell_height),
     #(0.5*cell_width, 0), (0.75*cell_width, -excursion_fraction*cell_height),
     # (cell_width, 0)]
    # where a and b are in inches, and c is a pure constant between 0 and 1.
    # Starting from (0,0), add one point to a Python list every s inches along the line segments.
    # Compute distance along the line segments with the Euclidean distance.

    points = [(0, 0),
              (0.25*cell_width, excursion_fraction*cell_height),
              (0.5*cell_width, 0),
              (0.75*cell_width, -excursion_fraction*cell_height),
              (cell_width, 0)]
    
    # Calculate length of a single segment
    seg_len = math.sqrt((0.25 * cell_width)**2 + (excursion_fraction * cell_height)**2)
    total_len = 4 * seg_len
    
    zigzag_points = []
    current_dist = 0.0
    
    while current_dist <= total_len:
        # Determine which segment [idx, idx+1] the distance falls into
        idx = min(int(current_dist // seg_len), 3)
        
        # Local distance within the current segment
        d_local = current_dist % seg_len if current_dist < total_len else seg_len
        if current_dist == total_len and len(zigzag_points) > 0 and zigzag_points[-1] == points[-1]:
             break # Avoid duplicate end point if s divides total_len perfectly

        # Linear interpolation
        p_start = points[idx]
        p_end = points[idx + 1]
        
        ratio = d_local / seg_len
        xp = p_start[0] + ratio * (p_end[0] - p_start[0])
        yp = p_start[1] + ratio * (p_end[1] - p_start[1])
        
        zigzag_points.append((xp + offset_y, yp + offset_x))
        current_dist += stitch_distance
    
    x = 2
        
    return zigzag_points

def halftone_stitch(image_path, fit_pattern_inside_H_W_inches, row_height_in, col_width_in,
                    stitch_len_in, dark_thread_on_light_background, stitch_height_scale):
    # Load the image and convert to grayscale
    image = Image.open(image_path).convert('L') # L = luminance -- converts image to grayscale upon import
    image_width_pixels, image_height_pixels = image.size

    pixels_per_inch = max(image_height_pixels/fit_pattern_inside_H_W_inches[0],
                          image_width_pixels/fit_pattern_inside_H_W_inches[1])
    pixels_per_row = row_height_in * pixels_per_inch
    pixels_per_col = col_width_in * pixels_per_inch

    new_image_width = int(image_width_pixels / pixels_per_col)
    new_image_height = int(image_height_pixels / pixels_per_row)

    resized_image = image.resize((new_image_width, new_image_height))

    # each pixel is col_width_in wide  and row_height_in tall in the resized_image.
    # Convert the grayscale PIL image to a NumPy array
    numpy_img = np.array(resized_image) / 255.0

    x_y_coords = []

    for ix in range(numpy_img.shape[0]):
        row_x_y_coords = []
        for iy in range(numpy_img.shape[1]):
            x0 = row_height_in * (ix + 0.5)
            y0 = col_width_in * (iy)
            if dark_thread_on_light_background:
                excursion = stitch_height_scale*(1.0 - numpy_img[ix][iy])
            else:
                excursion = stitch_height_scale*(numpy_img[ix][iy])
            zigzag_points = get_zigzag_points(cell_width = col_width_in,
                          cell_height = row_height_in,
                          excursion_fraction = excursion,
                          stitch_distance = stitch_len_in,
                          offset_x = x0, offset_y = y0)
            row_x_y_coords.extend(zigzag_points)
        if ix % 2 == 0:
            x_y_coords.extend(row_x_y_coords)
        else:
            row_x_y_coords.reverse()
            x_y_coords.extend(row_x_y_coords)
    # by default x_y_coords is actually sideways because of the way PIL reads in images.  We need to rotate it.
    h = fit_pattern_inside_H_W_inches[0]
    x_y_coords = [(c[0], h-c[1]) for c in x_y_coords]
    return x_y_coords

# Example usage

fit_pattern_inside_H_W_inches = [4.75, 6.75]
filename_input = 'crane.jpg'
xy_c = halftone_stitch(image_path = filename_input,
                fit_pattern_inside_H_W_inches = fit_pattern_inside_H_W_inches,
                row_height_in = 0.12,
                col_width_in = 0.1,
                stitch_len_in = 0.08,
                dark_thread_on_light_background = True,
                stitch_height_scale = 1.2)

df = pd.DataFrame(xy_c, columns=['x', 'y'])
df.to_csv(filename_input + '.csv')

stitch_count = len(xy_c)
print('Stitch count: ' + str(stitch_count))

# Create a new embroidery pattern
pattern = pyembroidery.EmbPattern()

pattern.add_thread({"rgb": (255, 0, 0), "description": "Red Thread", "catalog": "1000"})

# Add coordinates as stitches
scale_factor_vp3 = 254 # convert to units of 1/10 mm

for x, y in xy_c:
    pattern.add_stitch_absolute(pyembroidery.STITCH, scale_factor_vp3*x, -scale_factor_vp3*y)

# Export to VP3 format
pyembroidery.write(pattern, filename_input + ".vp3")

xL, yL = zip(*xy_c)
plt.plot(xL, yL, linestyle='-', color='b')
plt.xlabel('X (inches)')
plt.ylabel('Y (inches)')
plt.axis('square')
plt.title('Stitch Pattern for ' + filename_input)
plt.grid(True)

plt.show()
print('Done!')