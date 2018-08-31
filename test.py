import numpy as np
import cv2
from PIL import Image,ImageDraw
import math
import random
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from utils.ComputeNormal import ComputeNormal
from utils.CropRectangle import CropRectangle
from utils.ShowGrasp     import ShowGrasp



file_name = 'test/a/0.png'
MAX_EDGE_POINTS_NUM = 1000
TARGET_POINT_PAIR_DISTANCE = [165] # 11~200, if unknown, set it to [20, 50, 80, 110, 140, 170, 200]
image_width, image_height = 20, 20 # don't change


# import neural network
if K.image_data_format() == 'channels_first':
    input_shape = (1, image_width, image_height)
else:
    input_shape = (image_width, image_height, 1)
model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape = input_shape,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(16, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))
model.load_weights('weights/weights.h5', by_name = True)
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['binary_accuracy'])


# read images and compute edge
image_origin = Image.open(file_name).convert('L')
image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
t1 = time.clock()
image = cv2.GaussianBlur(image, (3,3), 0)
edge = cv2.Canny(image, 10, 50)
#cv2.imwrite("egde.jpg", edge)
edge = np.array(edge)

# find edge points
all_edge_points = np.nonzero(edge)
index_list = random.sample(range(len(all_edge_points[0])), min(len(all_edge_points[0]), MAX_EDGE_POINTS_NUM))
edge_points = [[all_edge_points[0][i], all_edge_points[1][i]] for i in index_list]


t2 = time.clock()
# generate grasp candidates(point pairs)
point_pairs = []
image = np.array(image)
DISTANCE_RANGE = 15 
for point in edge_points:
	normal = ComputeNormal(edge, image, point[0], point[1])
	sine   = math.sin(normal)
	cosine = math.cos(normal)
	for open_width in TARGET_POINT_PAIR_DISTANCE:
		for i in range(open_width - DISTANCE_RANGE, open_width + DISTANCE_RANGE):
			x = int(point[0] + i*sine)
			y = int(point[1] + i*cosine)
			if x>=0 and y>=0 and x<image.shape[0] and y<image.shape[1]:
				if edge[x][y] == 255:
					point_pairs.append([point[0], point[1], x, y])
					break


t3 = time.clock()
# pre-processing of grasp candidates
data = np.empty((len(point_pairs), image_width, image_height, 1), dtype='float32')
for i in range(len(point_pairs)):
	x1 = point_pairs[i][0]
	y1 = point_pairs[i][1]
	x2 = point_pairs[i][2]
	y2 = point_pairs[i][3]
	image_rectangle = CropRectangle(x1, y1, x2, y2, image_origin)

	if image_rectangle.size[0] <= 51:
		image_net = image_rectangle
	else:
		im_left = image_rectangle.crop((0,0,20,20))
		im_right = image_rectangle.crop((image_rectangle.size[0]-20,0,image_rectangle.size[0],20))
		im_mid = image_rectangle.crop((20,0,image_rectangle.size[0]-20,20))
		im_mid = im_mid.resize((10,20))
		image_net = Image.new('L',(50,20))
		image_net.paste(im_left,(0,0))
		image_net.paste(im_mid,(20,0))
		image_net.paste(im_right,(30,0))

	#image_net = image_rectangle.resize((image_width, image_height)) # simply resizing
	image_net = image_net.resize((image_width, image_height))
	image_net = np.asarray(np.array(image_net.convert('L')), dtype='float32')
	data[i,:,:,0] = np.transpose(image_net) * (1.0/255)


t4 = time.clock()
# network inference
grasp_scores = model.predict(data)


t5 = time.clock()
# rank the grasp candidates
ranked_grasps = []
for i in range(len(grasp_scores)):
	ranked_grasps.append([point_pairs[i][0], point_pairs[i][1], point_pairs[i][2], point_pairs[i][3], grasp_scores[i][0]])
ranked_grasps.sort(key = lambda x: x[4] , reverse = True)
best_grasp = ranked_grasps[0]

t6 = time.clock()



print('Total %d edge points!' %(len(edge_points)))
print('Total %d grasp candidates!' %(len(point_pairs)))

print('Best grasp in image coordinate x1, y1, x2, y2, grasp scores: ', best_grasp)

print('Edge & edge points generation: %.3f s' %(t2 - t1))
print('Grasp candidates generation:   %.3f s' %(t3 - t2))
print('Learning data generation:      %.3f s' %(t4 - t3))
print('Network inference:             %.3f s' %(t5 - t4))
print('Post processing:               %.3f s' %(t6 - t5))
print('Total computation time:        %.3f s' %(t6 - t1))


grasp = ShowGrasp(image_origin, best_grasp[0], best_grasp[1], best_grasp[2], best_grasp[3])
grasp.save('grasp.png')
