import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
def main(argv):
	image = Image.open(argv[0]) 
	image_rotate = image.rotate(180)
	image_rotate.save("ans2.png")

if __name__ == '__main__':
	main(sys.argv[1:])