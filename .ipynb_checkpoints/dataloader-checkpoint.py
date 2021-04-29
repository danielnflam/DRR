import pathlib
import os
import re
import SimpleITK as sitk
import numpy as np
import pandas as pd
"""
This file holds the functions required to extract numpy ndarrays, VoxelSize, ImOrigin, patient_code from the directories. 
"""

def CTCovid19August2020_generator(directory_path="D:/data/CT-Covid-19-August2020/"):
    directoryPath = pathlib.Path(directory_path)
    print(directoryPath)

    path_to_file = []
    for root, dirs, files in os.walk(directoryPath):
        for name in files:
            if name.endswith(".nii"):
                path_to_file.append(os.path.join(root, name))
                # For each .nii file:
                a_filepath = os.path.join(root, name)
                print(a_filepath)
                head, tail = os.path.split(a_filepath)
                patient_code_file = os.path.splitext(tail)[0]

                regex = re.compile(r'(?<=-)[0-9_]+')
                patient_code = regex.findall(patient_code_file)[0]
                print(patient_code)

                reader = sitk.ImageFileReader()
                reader.SetFileName(a_filepath)
                reader.LoadPrivateTagsOn()
                reader.ReadImageInformation()
                # Read Metadata

                for k in reader.GetMetaDataKeys():
                    v = reader.GetMetaData(k)
                    #print(f"({k}) = = \"{v}\"")

                image = reader.Execute();

                # Properties are [CxHxW]
                VoxelSize = list( map( float, [reader.GetMetaData('pixdim[3]'), reader.GetMetaData('pixdim[1]'), reader.GetMetaData('pixdim[2]')]) )
                ImOrigin = list( map( float, [reader.GetMetaData('qoffset_z'), reader.GetMetaData('qoffset_y'), reader.GetMetaData('qoffset_x')]) )
                ImOrigin[1],ImOrigin[2] = -ImOrigin[1], -ImOrigin[2]
                print("Origin: "+ str(ImOrigin))
                print("Voxel Size: " + str(VoxelSize))

                # Generate DRR based on image
                nda = sitk.GetArrayFromImage(image)
                
                yield nda, VoxelSize, ImOrigin, patient_code

def RIDER_CT_generator(directoryPath="D:\data\RIDER-CT", metadata_filename="metadata.csv"):
    df = pd.read_csv(os.path.join(directoryPath,metadata_filename))
    # If file is marked as CT
    CT_indices = df["SOP Class UID"]=="1.2.840.10008.5.1.4.1.1.2"
    df_CT = df[CT_indices]

    # For each study
    for index, row in df_CT.iterrows():
        patient_code = row["Series UID"]

        file_subpath = row["File Location"]
        file_subpath = file_subpath[2:]
        file_directory_path = os.path.join(directoryPath, file_subpath)

        # Read the file path
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(file_directory_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        spacing = image.GetSpacing()
        #print("Image spacing:", spacing[0], spacing[1], spacing[2])
        size = image.GetSize()
        #print("Image size:", size[0], size[1], size[2])
        origin = image.GetOrigin()
        #print("Origin:", origin[0], origin[1], origin[2])

        # Turn into Numpy NDARRAY
        # Note that array is shaped like [D x H x W]
        nda = sitk.GetArrayFromImage(image)
        print(nda.shape)
        VoxelSize = (spacing[2], spacing[0], spacing[1] )
        ImOrigin = (origin[2], origin[0], origin[1])
        yield nda, VoxelSize, ImOrigin, patient_code

class LIDC_IDRI:
    def __init__(self, directoryPath="D:\data\LIDC-IDRI", metadata_filename="metadata.csv"):
        self.dataFrame = pd.read_csv(os.path.join(directoryPath,metadata_filename))
        self.directoryPath = directoryPath
        self.metadata_filename = metadata_filename
    def generateImage(self, imageType="CT"):
        # If file is marked as CT
        if imageType == "CT":
            image_indices = self.dataFrame["SOP Class UID"].str.contains('1.2.840.10008.5.1.4.1.1.2')
        if imageType == "DX":
            image_indices = self.dataFrame["SOP Class UID"]=="1.2.840.10008.5.1.4.1.1.1"
            
        #print(image_indices)
        df_CT = self.dataFrame[image_indices]

        # For each study
        for index, row in df_CT.iterrows():
            patient_code = row["Series UID"]

            file_subpath = row["File Location"]
            file_subpath = file_subpath[2:]
            file_directory_path = os.path.join(self.directoryPath, file_subpath)

            # Read the file path
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(file_directory_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            spacing = image.GetSpacing()
            #print("Image spacing:", spacing[0], spacing[1], spacing[2])
            size = image.GetSize()
            #print("Image size:", size[0], size[1], size[2])
            origin = image.GetOrigin()
            #print("Origin:", origin[0], origin[1], origin[2])

            # Turn into Numpy NDARRAY
            # Note that array is shaped like [D x H x W]
            nda = sitk.GetArrayFromImage(image)
            print(nda.shape)
            VoxelSize = (spacing[2], spacing[0], spacing[1] )
            ImOrigin = (origin[2], origin[0], origin[1])
            yield nda, VoxelSize, ImOrigin, patient_code