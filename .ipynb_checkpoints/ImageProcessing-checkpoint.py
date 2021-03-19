import skimage
import skimage.measure
import numpy as np

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
    if max_area == 0:
        print('No connected components found')
        return
    else:
        mask = labels == props[idx]['label']
        label = props[idx]['label']
        return mask, label