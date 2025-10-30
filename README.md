# Analysing a Solar Flare with SUIT Data: Light Curve Tutorial

This tutorial guides you through the process of analysing a solar flare using multi-wavelength data from the Solar Ultraviolet Imaging Telescope (SUIT). We will start with raw FITS files, perform crucial pre-processing steps like exposure normalization and image co-alignment, and finally generate and plot light curves for the flare from a defined region of interest.

The primary steps in this workflow are:
1.  **Setup**: Importing necessary libraries.
2.  **Helper Functions**: Defining reusable functions for key tasks.
3.  **Data Loading**: Loading and pre-processing SUIT data for 8 different narrow-band filters.
4.  **Image Co-alignment**: Spatially aligning all images across different filters and over time.
5.  **Region of Interest (ROI) Selection**: Defining the flaring region using an intensity contour.
6.  **Light Curve Generation**: Creating time-series data from the ROI.
7.  **Visualisation & Export**: Plotting the light curves against GOES data and saving the results.

> **Important Note on Memory Usage**:
Beware that the peek function, although it provides a nice interactive way to view your maps, clogs your memory.
As of now, I still haven't found a way to clear the figures from the memory. The more you use the peek function, the slower the code will become. You can avoid it by using plot instead of peek, but it will be a lot less interactive. If the code gets very slow, just restart the kernel and start again with your new insights.

## 1. Setup and Imports

First, import the required libraries. We'll use `sunpy` and `sunkit-image` for solar data analysis, `astropy` for units and coordinates, and `matplotlib` for plotting.

```python
# Set up interactive plotting in a separate window
%matplotlib qt

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
from sunpy.map import Map
from datetime import datetime
import csv
from itertools import zip_longest

import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy import timeseries as ts
from tqdm import tqdm

# Sunkit-image for co-alignment
from sunkit_image.coalignment import mapsequence_coalign_by_match_template as coalign
from sunkit_image.coalignment import calculate_match_template_shift as cal_shift
from sunkit_image.coalignment import apply_shifts 

# Matplotlib utilities for plotting
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm
```

## 2. Helper Functions

To keep our main workflow clean and organised, we'll define a few helper functions. These functions encapsulate repetitive tasks like exposure normalisation, co-alignment, and light curve generation.

### Exposure Normalisation
This function normalises the image data by its exposure time. This is crucial for comparing images taken with different exposure settings. The function divides each image's data by its exposure time and updates the metadata to reflect a standard 1-second exposure.

```python
def exp_norm(map_sequence):
    """
    Normalizes a map sequence by its exposure time.
    """
    mod_rois = []
    for smap in tqdm(map_sequence):
        esf = (smap.meta['cmd_expt']) / 1e3 # Exposure scaling factor from ms to s
        data = smap.data / esf   
        data[data < 0] = 0
        smap.meta['cmd_expt'] = 1000 # Set metadata to a standard 1-second exposure
        mod_rois.append(Map(data, smap.meta))
    mod_rois = Map(mod_rois, sequence=True)
    return mod_rois
```

### Interactive Submap Selection
This function allows you to interactively select a rectangular region of interest (ROI) from an image by clicking on two opposite corners. It's useful for creating templates for co-alignment.

```python
def submap_with_ginput(sunpy_map):
    """
    Prompts the user to select two corners on a map plot and returns the
    corresponding submap.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=sunpy_map)
    sunpy_map.plot(axes=ax)
    ax.set_title("Click two corners of the ROI (e.g., bottom-left and top-right)")

    # Wait for 2 clicks from the user
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(pts) != 2:
        raise RuntimeError("ROI selection cancelled or failed.")

    (x1, y1), (x2, y2) = pts
    bottom_left = (min(x1, x2), min(y1, y2)) * u.pixel
    top_right = (max(x1, x2), max(y1, y2)) * u.pixel

    submap = sunpy_map.submap(bottom_left=bottom_left, top_right=top_right)
    return submap
```

### Co-alignment Wrapper
This function simplifies the process of co-aligning a sequence of maps. It uses `sunkit-image` to calculate the shifts required to align each map to a reference template and then applies those shifts. Checkthe  Coalignment module for more info.
Visit https://docs.sunpy.org/projects/sunkit-image/en/latest/code_ref/coalignment.html to know more about the functions used

```python
def co_align(map_sequence, template, layer_index=0, clip=False):
    """
    Co-aligns a map sequence to a given template.
    """
    shifts = cal_shift(map_sequence, layer_index=layer_index, template=template)
    plate_scale = map_sequence[0].scale[0]
    
    coaligned_maps = apply_shifts(map_sequence, xshift=-shifts['x']/plate_scale, yshift=-shifts['y']/plate_scale, clip=clip)
    return coaligned_maps
```

### Cropping a Map Sequence
This function crops all maps in a sequence to the dimensions of a sample map, ensuring that all images have a consistent size and similar FOV.

```python
def crop_maps(mc, sample):
    """
    Crops all maps in a map sequence to the dimensions of a sample map.
    """
    z = []
    b_left = sample.bottom_left_coord
    t_right = sample.top_right_coord
    x1, y1 = sample.world_to_pixel(b_left)
    x2, y2 = sample.world_to_pixel(t_right)
    for temp in mc:
        z.append(temp.submap(bottom_left=[x1, y1]*u.pix, top_right=[x2, y2]*u.pix))  
    z = Map(z, sequence=True)
    return z
```

### Interactive Alignment Workflow
This function combines submap selection and co-alignment into a single interactive workflow. It prompts you to select a template from the first image and then aligns the entire sequence to it.

```python
def align(mc, index=0):
    """
    An interactive function to select a template and co-align a map sequence.
    """
    templ = submap_with_ginput(mc[index])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=templ)
    templ.plot(axes=ax)
    
    ax.set_title("Click anywhere to close the template and begin coalignment", fontweight="bold")
    plt.waitforbuttonpress()
    plt.close()
    
    aligned = co_align(map_sequence=mc, layer_index=0, template=templ, clip=False)
    aligned.peek(); plt.show(); plt.colorbar()
    return aligned
```

### Light Curve Generation
This function generates a light curve by summing the pixel values within a given mask for each map in a sequence.

```python
def make_lc(mask, map_sequence):
    """
    Creates a light curve by summing pixel values within a mask for each map.
    """
    lc = []; time = []
    for smap in map_sequence:
        data = smap.data
        lc.append(np.sum(data[mask]))
        # Parse the timestamp from the FITS header
        time.append(datetime.strptime(smap.meta['DHOBT_DT'].split('.')[0], "%Y-%m-%dT%H:%M:%S"))
    return time, lc
```

## 3. Loading and Pre-processing the Data

Now, let's load the FITS files for each of the 8 SUIT narrow-band filters. We will perform exposure normalisation and crop them all to the same size. We start with the NB03 filter, which we will use as our reference.

```python
basepath = 'data/'

# Load, normalize, and crop the NB03 files
files3 = sorted(glob(basepath + 'nb03/*NB03.fits'))
smaps3 = exp_norm(Map(files3, sequence=True))
smaps3 = crop_maps(smaps3, smaps3[0]) # Ensure all maps have the same dimensions
```
Now, we repeat the process for the remaining 7 filters, ensuring each is cropped to match the dimensions of our NB03 reference maps.
```python
# Loading rest of the dataset
files1 = sorted(glob(basepath+'nb01/*NB01.fits'))
smaps1 = crop_maps(exp_norm(Map(files1,sequence=True)),smaps3[0])

files2 = sorted(glob(basepath+'nb02/*NB02.fits'))
smaps2 = crop_maps(exp_norm(Map(files2,sequence=True)),smaps3[0])

files4 = sorted(glob(basepath+'nb04/*NB04.fits'))
smaps4 = crop_maps(exp_norm(Map(files4,sequence=True)),smaps3[0])

files5 = sorted(glob(basepath+'nb05/*NB05.fits'))
smaps5 = crop_maps(exp_norm(Map(files5,sequence=True)),smaps3[0])

files6 = sorted(glob(basepath+'nb06/*NB06.fits'))
smaps6 = crop_maps(exp_norm(Map(files6,sequence=True)),smaps3[0])

files7 = sorted(glob(basepath+'nb07/*NB07.fits'))
smaps7 = crop_maps(exp_norm(Map(files7,sequence=True)),smaps3[0])

files8 = sorted(glob(basepath+'nb08/*NB08.fits'))
smaps8 = crop_maps(exp_norm(Map(files8,sequence=True)),smaps3[0])
```
You can inspect any of the loaded datasets using `peek()`.
```python
# Have a look at any map you like.
smaps6.peek()
```

## 4. Co-alignment: The Key to Multi-Wavelength Analysis

Co-alignment is a critical step. It ensures that a specific solar feature is located at the same pixel position in all images. We will perform this in two stages:

1.  **Inter-Filter Alignment**: Align the first image of each filter to a common reference.
2.  **Intra-Filter Alignment**: Align all subsequent images within a filter's time series to its (now aligned) first image.

### Step 4.1: Inter-Filter Alignment (Spatial Registration)

We start by selecting a template from the first NB03 image. This template should contain distinct, high-contrast features.

```python
# Create a template from the first NB03 image for alignment.
template = submap_with_ginput(smaps3[0])
template.plot()
```

Next, we create a temporary map sequence containing only the first frame from each of the 8 filters and align them to the template.

```python
# Create a map sequence of the first images from all filters.

temp = Map(smaps1[0], smaps2[0], smaps3[0], smaps4[0], smaps5[0], smaps6[0], smaps7[0], smaps8[0], sequence=True)
for i,tmap in enumerate(temp):
    print(i,'    ',tmap.meta['ftr_name']) #this just shows index of a particular filter. Use this index to align 

# Align the images using the template

temp_aligned = co_align(temp, layer_index=2, template=template)
#index=2 corresponds to NB03 in the 'temp' sequence. Map() sorts maps based on observation time. With the sample data provided, index=2 should relate to the index of the NB03 filter. You are free to choose whatever filter you want; all other filters will be aligned to the layer_index provided.

temp_aligned.peek(); plt.show()

filter_index = {}
for i,tmap in enumerate(temp):
    filter_index[f"{tmap.meta['ftr_name']}"] = i
#filter_index is a dictionary to store the index of different filters from temp
(format---> 'filter':index). This will be useful in the next step. 
```

If the coalignment does not work, try using a different template. The template provided below will only work with the sample data. Selecting a template is a trail and error process. Don't be discouraged. Have a look at the coalignment module for tips to make a template if you havent already.
```python
bottom_left = SkyCoord(-407*u.arcsec, 397*u.arcsec, frame=smaps3[0].coordinate_frame)
top_right = SkyCoord(-348*u.arcsec, 442*u.arcsec, frame=smaps3[0].coordinate_frame)
template1 = smaps3[0].submap(bottom_left,top_right=top_right)
template1.plot()
```

Finally, we replace the original first frame of each filter's map sequence with these newly aligned frames. This ensures our reference frame for the next step is correctly registered across all wavelengths.

```python
# Replace the first map of each sequence with the aligned one.
# We use the dictionary created earlier to extract the aligned image of a particular filter correspondin to that map_sequence.
smaps1.maps[0]  = temp[filter_index[f"{smaps1[1].meta['ftr_name']}"]]
smaps2.maps[0]  = temp[filter_index[f"{smaps2[1].meta['ftr_name']}"]]
smaps3.maps[0]  = temp[filter_index[f"{smaps3[1].meta['ftr_name']}"]]
smaps4.maps[0]  = temp[filter_index[f"{smaps4[1].meta['ftr_name']}"]]
smaps5.maps[0]  = temp[filter_index[f"{smaps5[1].meta['ftr_name']}"]]
smaps6.maps[0]  = temp[filter_index[f"{smaps6[1].meta['ftr_name']}"]]
smaps7.maps[0]  = temp[filter_index[f"{smaps7[1].meta['ftr_name']}"]]
smaps8.maps[0]  = temp[filter_index[f"{smaps8[1].meta['ftr_name']}"]]
```

### Step 4.2: Intra-Filter Alignment

Now we align all images within each filter's time series to its aligned first image. We use our interactive `align` function for this. This step corrects for satellite pointing drift over time.

**This process is interactive and may take a few tries.** When prompted, select a small sub-region with clear, stable features to serve as the alignment template. A good template is key to successful co-alignment.

```python
coaligned_maps1 = align(smaps1)
coaligned_maps2 = align(smaps2)
coaligned_maps3 = align(smaps3)
coaligned_maps4 = align(smaps4)
coaligned_maps5 = align(smaps5)
coaligned_maps6 = align(smaps6)
coaligned_maps7 = align(smaps7)
coaligned_maps8 = align(smaps8)
```
After alignment, you can check the result again with `peek()`. The solar features should now appear stable throughout the animation.
```python
coaligned_maps3.peek(); plt.colorbar()
```

## 5. Defining the Flare Region (ROI)

To create a light curve, we need to define the region of the flare. A common method is to use an intensity contour from the flare's peak emission. Here, we'll use the 60% contour of the peak brightness in the NB03 filter.

First, let's identify the peak frame and create the contour. For this dataset, the peak is around the 24th frame.

```python
j = 24 # Index of the flare peak in the NB03 sequence
plt.figure()
contour_level = np.max(coaligned_maps3[j].data) * 0.6
contours = plt.contour(coaligned_maps3[j].data, levels=[contour_level], colors='red')
plt.imshow(coaligned_maps3[j].data, origin='lower', cmap='gray')
plt.title("60% Contour at Flare Peak")
plt.show()
```
Now, we convert this contour into a boolean mask. The light curve will be generated by summing all pixel values where the mask is `True`.
```python
# Create a boolean mask from the contour
path = contours.get_paths()[0]
data_shape = coaligned_maps3[j].data.shape
x, y = np.meshgrid(np.arange(data_shape[1]), np.arange(data_shape[0]))
points = np.vstack((x.flatten(), y.flatten())).T 
mask = path.contains_points(points).reshape(data_shape)

plt.figure()
plt.imshow(mask, origin='lower')
plt.title("Flare Region Mask")
plt.show()
```

## 6. Generating and Plotting Light Curves

With our aligned data and flare mask, we can now generate the light curves for all 8 filters using the `make_lc` function.

```python
time1, lc1 = make_lc(mask, coaligned_maps1)
time2, lc2 = make_lc(mask, coaligned_maps2)
time3, lc3 = make_lc(mask, coaligned_maps3)
time4, lc4 = make_lc(mask, coaligned_maps4)
time5, lc5 = make_lc(mask, coaligned_maps5)
time6, lc6 = make_lc(mask, coaligned_maps6)
time7, lc7 = make_lc(mask, coaligned_maps7)
time8, lc8 = make_lc(mask, coaligned_maps8)
```

For context, we'll also load GOES X-ray flare data.

```python
# Load the GOES data file
file_goes = glob('*.nc')[0]
goes_18 = ts.TimeSeries(file_goes)
xrs_short = goes_18.data['xrsb'] # 1-8 Angstrom channel
```

Finally, let's plot our SUIT light curves, normalized to their peak values, and overlay the GOES data.

```python
fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
ax = ax.flatten()

# Plotting each light curve on different subplots for clarity
ax[2].plot(time1, lc1/max(lc1), '.-', label='NB01')
ax[1].plot(time2, lc2/max(lc2), 'm.-', label='NB02')
ax[0].plot(time3, lc3/max(lc3), '.-.', label='NB03')
ax[0].plot(time4, lc4/max(lc4), 'yh-', label='NB04')
ax[1].plot(time5, lc5/max(lc5), 'c^-', label='NB05')
ax[2].plot(time6, lc6/max(lc6), 's-', label='NB06')
ax[2].plot(time7, lc7/max(lc7), '.-', label='NB07')
ax[0].plot(time8, lc8/max(lc8), 'h--', label='NB08')

ax[0].set_xlim(time3[0], time3[-1])

# Formatting the plots
for i in range(3):
    ax[i].legend(frameon=False)
    ax[i].axvline(x=time3[np.argmax(lc3)], color='black', linestyle='dotted', label='NB03 Peak')
    ax[i].axvline(x=xrs_short.idxmax(), color='green', linestyle='dashed', label='GOES Peak')
    
    # Add GOES data on a secondary y-axis
    ax1 = ax[i].twinx()
    ax1.plot(xrs_short, 'r', label='GOES')
    ax1.set_ylim(1e-6, 9e-4)
    ax1.legend(frameon=False, loc=2)
    ax1.set_yscale('log')

# Adding labels and titles
fig.subplots_adjust(hspace=0.0)
fig.text(0.02, 0.5, 'SUIT Normalized Counts', va='center', rotation='vertical', fontsize=16)
fig.text(0.96, 0.5, 'Flare Class W/m$^{2}$', va='center', rotation=270, fontsize=16)
fig.subplots_adjust(right=0.9) 
plt.suptitle('60% Contour LightCurves', fontsize=18, y=0.92)
fig.supxlabel('TIME (UT)', fontsize=16, y=0.04)

# Create proxy handles for the vertical line legends
vline_black = Line2D([0], [0], color='black', linestyle='dotted', label='NB03 Peak')
vline_green = Line2D([0], [0], color='green', linestyle='dashed', label='GOES Peak')
fig.legend(handles=[vline_black, vline_green], loc='upper right', bbox_to_anchor=(0.85, 0.9), frameon=False)

plt.show()
# plt.savefig("lightcurves_60percent.pdf", dpi=400)
```

## 7. Saving the Results

As a final step, it's good practice to save your processed data. Let's save the generated light curves to a CSV file for future analysis.

```python
# Save the light curve data to a CSV file
with open('nblc_60contour.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['time1', 'lc1', 'time2', 'lc2', 'time3', 'lc3', 'time4', 'lc4', 
                     'time5', 'lc5', 'time6', 'lc6', 'time7', 'lc7', 'time8', 'lc8'])   
    # Write data rows
    writer.writerows(zip_longest(time1, lc1, time2, lc2, time3, lc3, time4, lc4,
                                 time5, lc5, time6, lc6, time7, lc7, time8, lc8, 
                                 fillvalue=''))
```

Congratulations! You have successfully processed multi-wavelength SUIT data to produce co-aligned image sequences and scientifically valuable light curves of a solar flare.


