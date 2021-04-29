import skimage.measure, skimage.morphology
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2

def flood_fill(test_array,h_max=255):
    """
    source: https://stackoverflow.com/questions/36294025/python-equivalent-to-matlab-funciton-imfill-for-grayscale
    """
    input_array = np.copy(test_array) 
    el = scipy.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = scipy.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = scipy.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,scipy.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

def mask2points(array, extents):
    """
    Turns a mask into points.
    
    extents: [[minimum_i, maximum_i], [minimum_j, maximum_j], [minimum_k, maximum_k]]
    """
    extents = np.asarray(extents)
    
    # If extents are from edge-to-edge like MATLAB
    # Pixel Spacing
    pixel_i = (extents[0][1]-extents[0][0])/(array.shape[0])
    pixel_j = (extents[1][1]-extents[1][0])/(array.shape[1])
    
    #print(pixel_i)
    # origin_pixel
    dict1 = skimage.measure.regionprops_table(array.astype(np.uint8), properties=['coords'])

    origin_pixel_center_i = extents[0][0]+0.5*pixel_i
    origin_pixel_center_j = extents[1][0]+0.5*pixel_j

    world_i = dict1['coords'][0][:,0:1]*pixel_i + origin_pixel_center_i
    world_j = dict1['coords'][0][:,1:2]*pixel_j + origin_pixel_center_j
    
    world_coords = np.concatenate((world_i, world_j), axis=1)
    
    if len(array.shape) ==3:
        #print("3D array")
        pixel_k = (extents[2][1]-extents[2][0])/(array.shape[2])
        origin_pixel_center_k = extents[2][0]+0.5*pixel_k
        world_k = dict1['coords'][0][:,2:3]*pixel_k + origin_pixel_center_k
        world_coords = np.concatenate((world_i, world_j, world_k), axis=1)
        
    return world_coords

def imageComplement(image):
    image = image.astype(np.float)
    max_image = np.amax(image)
    min_image = np.amin(image)
    
    if max_image == min_image:
        print("No difference between max and min intensity.")
        return image
    
    scaled_image = (image - min_image)/(max_image - min_image)
    scaled_complement = 1. - scaled_image
    complement = scaled_complement*(max_image-min_image) + min_image
    return complement

def connectedComponentsLabelling(mask, connectivity=1, k=1):
    """
    output_mask , label = connectedComponentsLabelling(input_mask, connectivity, k)
    
    OUTPUT:
    output_mask: the array containing the CC-labelled item that is the 'kth' largest in area.  E.g. k=1 means that the largest CC has been extracted and output.
    label: the label associated with this CC.
    INPUT:
    input_mask: the array to be CC-labelled
    connectivity = 1 --> see skimage.measure.label for definition
    k: integer for selecting which CC to extract.  k=1 means that the largest-area CC is extracted.
    """
    labels = skimage.measure.label(mask, background=0, connectivity=connectivity)
    props = skimage.measure.regionprops(labels)
    region_area = []
    for region in range(len(props)):
        region_area.append(props[region]['area'])
    
    sorted_area = np.sort(region_area)
    argsorted = np.argsort(region_area)
    idx = argsorted[-k]
    max_area = sorted_area[-1]
    area = sorted_area[-k]
    if max_area == 0:
        print('No connected components found')
        return
    else:
        mask = labels == props[idx]['label']
        label = props[idx]['label']
        return mask, area, label
    
def plotImage(image_array, PixelSize, origin):
    """
    plotImage(image_array, PixelSize, origin)
    
    """
    # extents is [left, right, top, bottom]
    extents = [origin[1], origin[1]+PixelSize[1]*(image_array.shape[1]-1), origin[0]+PixelSize[0]*(image_array.shape[0]-1), origin[0] ]
    plt.imshow(image_array, extent=extents, origin='upper', cmap='Greys')
    plt.gca().invert_yaxis()
    plt.bone()
    plt.plot
    return extents


def BoneSegmentation(nda_nobed):
    """
    Segment high-intensity structures (i.e. bone) from CT image.
    Input: 
    nda_nobed: 3D array of body, masked to remove non-body structures e.g. bed.
    """
    bone = nda_nobed.copy() > 250 # Adams et al. 2012. Chapter 12 - Radiology (Pediatric Bone Second Edition).
    
    #bone, _, _ = connectedComponentsLabelling(bone, connectivity=1, k=1)
    
    strel=skimage.morphology.disk(3)    
    for i in range(bone.shape[0]):
        # Dilate
        bone[i,:,:] = skimage.morphology.dilation(bone[i,:,:], selem=strel)
        # Hole-fill
        bone[i,:,:] = scipy.ndimage.binary_fill_holes(bone[i,:,:])
        # Erode
        bone[i,:,:] = skimage.morphology.erosion(bone[i,:,:], selem=strel)
    
    
    return bone

def LungSegmentation(nda_nobed, mask):
    """
    Input: 
    nda_nobed:  3D array of body, masked to remove non-body structures e.g. bed.
    mask:  body mask (3D array, same size as nda_nobed
    """
    
    #th = skimage.filters.threshold_otsu(nda_nobed, nbins=256)
    lung = nda_nobed.copy() < -300 # Automatic Detection and Quantification of Ground-Glass Opacities on High-Resolution CT Using Multiple Neural Networks. 
#                                    Hans-Ulrich Kauczor, Kjell Heitmann, Claus Peter Heussel, Dirk Marwede, Thomas Uthmann, and Manfred Thelen
#                                    American Journal of Roentgenology 2000 175:5, 1329-1334

    lung = lung * mask  # remove non-lung air structures outside body.
    
    lung_original_mask = lung.copy()
    
    # Erode low-HU structures to separate lung & bronchi
    
    """strel=skimage.morphology.disk(3)
    for i in range(lung.shape[0]):
        lung[i,:,:] = skimage.morphology.erosion(lung[i,:,:], selem=strel)"""
    lung = skimage.morphology.erosion(lung, selem=skimage.morphology.ball(1))
    # CC-labelling -- find the lungs
    lung1, maxarea1, _ = connectedComponentsLabelling(lung, connectivity=1, k=1)
    lung2, maxarea2, _ = connectedComponentsLabelling(lung, connectivity=1, k=2)
    if maxarea2/maxarea1 > 0.25:
        lung = (lung1 + lung2)>0
    else:
        lung = lung1 > 0
    
    # Dilate to return to previous size & shape
    lung = skimage.morphology.dilation(lung, selem=skimage.morphology.ball(1))
    """for i in range(lung.shape[0]):
        lung[i,:,:] = skimage.morphology.dilation(lung[i,:,:], selem=strel)"""
    
    # Multiply with original mask to get the correct edges
    lung = lung_original_mask * lung
    
    return lung

def LungMaskSegmentation_Gozes2018(nda_nobed , body_mask ):
    """
    Extract a 2D lung mask from a 3D CT image.
    Paper: 
    
    
    Inputs:
        nda_nobed: [IS x AP x LR] CT of patients
        body_mask: [IS x AP x LR] body mask of patients
    """
    nda_nobed_masked = (nda_nobed < -500) * body_mask # exclude extracorporeal air
    lung_segment = nda_nobed_masked.copy()
    
    lung_segment , _, _ = connectedComponentsLabelling(lung_segment,1,1)
    
    # Create the 2D mask
    lung_segment_2D = np.sum(lung_segment, axis=1) > 0
    
    return lung_segment_2D
    