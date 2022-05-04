import mss
import cv2 as cv
import pyautogui
import pytesseract as pts
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def take_screenshot():
	with mss.mss() as sct:
		# grab screenshot
		sct_image = sct.grab(sct.monitors[1])
		
		# raw data to PIL image
		image = Image.frombytes("RGB", sct_image.size, sct_image.bgra, "raw", "BGRX")
		
		# save debug image
		output = "monitor.png"
		image.save(output)
		
		return image


def get_state(needle_path):
	
	# get images for matchTemplate
	if debug:
		take_screenshot()
		needle_path = "images/equipment.png"
		screenshot = cv.imread("monitor.png")
		needle = cv.imread(needle_path)
	else:
		take_screenshot()
		screenshot = cv.imread("monitor.png")
		needle = cv.imread(needle_path)
		
	# TODO: debug flag
	if debug:
		# generate result for debug
		res = cv.matchTemplate(screenshot, needle, cv.TM_CCOEFF_NORMED)
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
		print(min_val, max_val, min_loc, max_loc)
		top_left = max_loc
		bottom_right = (top_left[0] + needle.shape[1], top_left[1] + needle.shape[0])
		cv.rectangle(screenshot, top_left, bottom_right, 255, 2)
		cv.imwrite('res.png', screenshot)
	
		# plot result for debug
		canvas = np.zeros((screenshot.shape[0], screenshot.shape[1]), np.uint8)
		plt.figure(figsize=(16, 12))
		img = mpimg.imread("res.png")
		plt.imshow(img)
		plt.axis('on')
		plt.title("result")
		plt.imshow(cv.cvtColor(img, cv.IMREAD_GRAYSCALE))
		plt.show()
		
		
if __name__ == '__main__':
	debug = True
	get_state("images/equipment.png")
	# print(pts.image_to_string(Image.open('monitor.png')))
	