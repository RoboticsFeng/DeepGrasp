from PIL import Image
import math



def CropRectangle(x1, y1, x2, y2, image):
	CONTACT_DISTANCE = 10
	GRIPPER_FINGER_WIDTH = 20

	x0 = int((x1 + x2)*0.5)
	y0 = int((y1 + y2)*0.5)
	# crop a small area and rotate the rectangle to horizontal
	offset = int((abs(x1-x2) + abs(y1-y2) + CONTACT_DISTANCE*2 + GRIPPER_FINGER_WIDTH)*0.5)
	region1 = (y0-offset, x0-offset, y0+offset, x0+offset)
	image_crop = image.crop(region1)
	image_rotate = image_crop.rotate(math.atan2(x2-x1, y2-y1)*57.3)

	# crop the rectangle
	gripper_open_width_half = math.sqrt((x1-x2)**2 + (y1-y2)**2)*0.5 + CONTACT_DISTANCE
	region2 = (int(image_rotate.size[0]*0.5-gripper_open_width_half), int(image_rotate.size[1]*0.5-GRIPPER_FINGER_WIDTH*0.5), 
		int(image_rotate.size[0]*0.5+gripper_open_width_half), int(image_rotate.size[1]*0.5+GRIPPER_FINGER_WIDTH*0.5))
	image_rectangle = image_rotate.crop(region2)

	return image_rectangle