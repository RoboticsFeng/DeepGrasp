from PIL import Image, ImageDraw
import math

def ShowGrasp(image, x1, y1, x2, y2):
	CONTACT_DISTANCE = 10
	GRIPPER_FINGER_WIDTH = 20

	image = image.convert('RGB')
	draw = ImageDraw.Draw(image)
	a = CONTACT_DISTANCE
	b = GRIPPER_FINGER_WIDTH
	x0 = (x1 + x2)/2
	y0 = (y1 + y2)/2
	alpha1 = math.atan2(y1-y0, x1-x0)
	alpha2 = math.atan2(y2-y0, x2-x0)
	rect0 = [int(x1+a*math.cos(alpha1) - b*0.5*math.sin(alpha1)), int(y1+a*math.sin(alpha1) + b*0.5*math.cos(alpha1))]
	rect1 = [int(x1+a*math.cos(alpha1) + b*0.5*math.sin(alpha1)), int(y1+a*math.sin(alpha1) - b*0.5*math.cos(alpha1))]
	rect2 = [int(x2+a*math.cos(alpha2) - b*0.5*math.sin(alpha2)), int(y2+a*math.sin(alpha2) + b*0.5*math.cos(alpha2))]
	rect3 = [int(x2+a*math.cos(alpha2) + b*0.5*math.sin(alpha2)), int(y2+a*math.sin(alpha2) - b*0.5*math.cos(alpha2))]
	draw.line((rect1[1], rect1[0], rect0[1], rect0[0]), fill = (0,0,255))#blue
	draw.line((rect2[1], rect2[0], rect1[1], rect1[0]), fill = (255,0,0))#red
	draw.line((rect3[1], rect3[0], rect2[1], rect2[0]), fill = (0,0,255))    
	draw.line((rect0[1], rect0[0], rect3[1], rect3[0]), fill = (255,0,0))

	return image	
