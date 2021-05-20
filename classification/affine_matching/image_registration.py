'''
register images from source to target
'''

from align_transform import Align


# Load source image and target image
source_path = '/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/JPEGImages/DSC_2410.JPG'
target_path = '/media/ck/B6DAFDC2DAFD7F45/program/pyTuft/tiny-instance-segmentation/dataset/JPEGImages/DSC_2496.JPG'

# Create instance
al = Align(source_path, target_path, threshold=1)

# Image transformation
al.align_image()
