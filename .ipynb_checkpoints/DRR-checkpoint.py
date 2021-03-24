import numpy as np

def createDRR(array, k, VoxelSize, ImOrigin):
    """
    Campo's algorithm for generating DRR images
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6239425/
    
    Output:
    out: 2D array
    PixelSize: 2D array pixel size.  For k=0, PixelSize is [HxW]; for k=1, PixelSize is [CxW]; for k=2, PixelSize is [HxC].
    origin: uppermost and leftmost corner of the array
    
    Input: 
    array: [CxHxW] where C is planes (IS), H is rows (RL), W is columns (AP).
    k: denotes the axis direction which is summed over.
    VoxelSize: [C, H, W] size of voxel in array
    ImOrigin: [C, H, W] the data coordinates of the Rightmost, Anteriormost, Inferiormost corner of the array
    """
    if k==1:
        array = np.moveaxis(array, 0, 1)
        PixelSize = [VoxelSize[0], VoxelSize[2]]
        origin = [ImOrigin[0], ImOrigin[2]]
    elif k==2:
        array = np.moveaxis(array, 0, 2)
        array = np.swapaxes(array,0,1)
        PixelSize = [VoxelSize[1], VoxelSize[0]]
        origin = [ImOrigin[1], ImOrigin[0]]
    elif k==0:
        array = array
        PixelSize = [VoxelSize[1], VoxelSize[2]]
        origin = [ImOrigin[1], ImOrigin[2]]
    else:
        raise RuntimeError('k can be 0, 1, 2')
    out = DRR_algorithm_Campo(array, beta=0.85)
    return out, PixelSize, origin

def DRR_algorithm_Campo(array, beta):
    """
    Campo's algorithm for generating DRR images
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6239425/
    
    Input: 
    array: [CxHxW] where C is planes (IS), H is rows (RL), W is columns (AP).
    beta: 0.85 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6239425/
    
    Output:
    out: [HxW] with values increasing from 0
    """
    
    a = (np.maximum(array,-1024) + 1024)/1000
    out = (1/array.shape[0])*np.sum((np.exp(beta*a)-1), axis=0)
    return out