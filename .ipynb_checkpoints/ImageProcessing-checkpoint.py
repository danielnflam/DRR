import skimage.measure, skimage.morphology
import numpy as np
import matplotlib.pyplot as plt
import scipy

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
    extents = [origin[1], origin[1]+PixelSize[1]*(image_array.shape[1]-1), origin[0]+PixelSize[0]*(image_array.shape[0]-1), origin[0] ]
    plt.imshow(image_array, extent=extents, origin='upper')
    plt.gca().invert_yaxis()
    plt.bone()
    plt.plot
    return extents


def BoneSegmentation(nda_nobed):
    """
    Input: 
    nda_nobed: 3D array of body, masked to remove non-body structures e.g. bed.
    """
    bone = nda_nobed.copy() > 250 # Adams et al. 2012. Chapter 12 - Radiology (Pediatric Bone Second Edition).
    
    # Dilate
    strel=skimage.morphology.ball(3)
    bone = skimage.morphology.dilation(bone, selem=strel)
    # Hole-fill
    for i in range(bone.shape[0]):
        bone[i,:,:] = scipy.ndimage.binary_fill_holes(bone[i,:,:])
    
    # Erode
    bone = skimage.morphology.erosion(bone, selem=strel)
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
    
    # CC-labelling
    lung1, maxarea1, _ = connectedComponentsLabelling(lung, connectivity=1, k=1)
    lung2, maxarea2, _ = connectedComponentsLabelling(lung, connectivity=1, k=2)
    if maxarea2/maxarea1 > 0.5:
        lung = (lung1 + lung2)>0
    else:
        lung = lung1 > 0
    
    return lung