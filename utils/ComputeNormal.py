import numpy as np 
import math

def ComputeNormal(edge, image, x, y):
# edge--edge image in numpy array
# image-origin image in numpy array
	NORMAL_SIZE = 5
	if x-NORMAL_SIZE<0 or x+NORMAL_SIZE>image.shape[0] or y-NORMAL_SIZE<0 or y+NORMAL_SIZE>image.shape[1]:
		return 0
	sum_N, sum_x, sum_y, sum_xy, sum_x2 = 0, 0, 0, 0, 0
	for i in range(x-NORMAL_SIZE, x+NORMAL_SIZE):
		for j  in range(y-NORMAL_SIZE, y+NORMAL_SIZE):
			if edge[i][j] == 255:
				sum_N  += 1
				sum_x  += i
				sum_y  += j
				sum_xy += i*j
				sum_x2 += i*i
	normal = math.atan2(sum_x2 - sum_x*sum_x/sum_N, sum_xy - sum_x*sum_y/sum_N) + 1.571
	offset_x = int(5*math.sin(normal))
	offset_y = int(5*math.cos(normal))
	if image[x+offset_x][y+offset_y] > image[x-offset_x][y-offset_y]:
		return normal + 3.1416
	else:
		return normal 