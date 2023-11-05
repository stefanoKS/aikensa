import cv2
import os
import glob
import sys

# Check if an argument has been provided
if len(sys.argv) != 2:
    raise ValueError("Please provide a directory path as an argument.")
    
# Define the directory path that contains images from the first argument.
img_dir = sys.argv[1]
images = glob.glob(os.path.join(img_dir, '*.png'))

# Specify the output video file name and its FPS.
output = 'output.mp4'
fps = 4

# Ensure there is at least one image to read.
if not images:
    raise FileNotFoundError("No images found in the specified directory.")

# Retrieve the dimensions of the first image.
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define the codec and create VideoWriter object.
# Note: You may need to install additional codecs or use a different fourcc if 'H265' is not available.
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

out = cv2.VideoWriter(output, fourcc, fps, (width, height))

# Read each image file and write it to the video.
for img in sorted(images):
    frame = cv2.imread(img)
    out.write(frame)

# Release the VideoWriter object.
out.release()
