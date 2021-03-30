import numpy as np
import ImageProcessing
import scipy as scp

def DiceSimilarityCoefficient(axial_auto, axial_validation):
    """
    Calculates the Dice Similarity Coefficient between 2 numpy ndarrays. DSC = 2*(A and B) / (|A| + |B|)

    Input ndarrays must be boolean.

    From Sorenson's paper:  Sørensen, T. (1948). "A method of establishing groups of equal amplitude in plant sociology based on similarity of species and its application to analyses of the 
    vegetation on Danish commons". Kongelige Danske Videnskabernes Selskab. 5 (4): 1–34

    """
    denom = (axial_auto.astype('bool').sum() + axial_validation.astype('bool').sum())*0.5
    num = (axial_auto.astype('bool') & axial_validation.astype('bool')).sum()
    return num/denom

def JaccardIndex(axial_auto, axial_validation):
    """
    Calculates the Jaccard Index / Tanimoto Index between 2 numpy ndarrays. DSC = (A and B) / (A or B)

    Input ndarrays must be boolean.

    From Jaccard's paper: Jaccard, Paul (February 1912). "THE DISTRIBUTION OF THE FLORA IN THE ALPINE ZONE.1". New Phytologist. 11 (2): 37–50. doi:10.1111/j.1469-8137.1912.tb05611.x

    """
    denom = (axial_auto.astype('bool') | axial_validation.astype('bool')).sum()
    num = (axial_auto.astype('bool') & axial_validation.astype('bool')).sum()
    return num/denom

def MCCD(axial_auto, axial_validation, extents_2D):
    """
    Calculate MCCD.
    Extents_2D is [[minimum_i, maximum_i], [minimum_j, maximum_j]]
    """
    
    # Distance transform to get contour mask voxels
    axial_auto = scp.ndimage.distance_transform_edt(axial_auto)
    axial_validation = scp.ndimage.distance_transform_edt(axial_validation)
    
    # Mask to points
    points_auto = ImageProcessing.mask2points(axial_auto, extents_2D)
    points_val = ImageProcessing.mask2points(axial_validation, extents_2D)
    
    # Contour points compared to each other
    d12 = []
    d21 = []
    for point in range(points_auto.shape[0]):
        a_point = points_auto[point]
        dist = np.linalg.norm(a_point - points_val , axis=1)
        d12.append(min(dist))
        
        
    for point in range(points_val.shape[0]):
        a_point = points_val[point]
        dist = np.linalg.norm(a_point - points_auto , axis=1)
        d21.append(min(dist))
    
    d = (1/points_auto.shape[0])*np.sum(d12) + (1/points_val.shape[0])*np.sum(d21)
    return d

def SSIM(reference, testImage):
    return 0