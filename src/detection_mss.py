import collections
import mss
import cv2 as cv
import pyautogui
import pytesseract as pts
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


class MatchResult(object):
    
    def __init__(self, match_results, needle):
        """
        :param match_results: results from matchTemplate()
        :param needle: needle image
        """
        self.min_val = match_results[0]
        self.max_val = match_results[1]
        self.min_loc = match_results[2]
        self.top_left = match_results[3]
        self.bottom_right = (self.top_left[0] + needle.shape[1], self.top_left[1] + needle.shape[0])
    
    def percent_string(self) -> str:
        """
        returns string for less clutter in plots
        :return: percent as string with no float
        """
        return str(int(self.max_val * 100))
    
    def match(self) -> bool:
        """
        returns true if match is over set %
        :return: match bool
        """
        return True if self.max_val > 0.99 else False


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


def draw_result(haystack, result, needle_path):
    """
    draws rectangles on haystack image with the name and percentage of given needle image
    :param haystack: original image
    :param result: result of type MatchResult
    :param needle_path: name of file
    """
    cv.rectangle(
        haystack,
        result.top_left, result.bottom_right,
        (0, 255, 0), 2
    )
    cv.putText(
        haystack,
        (needle_path + " " + result.percent_string() + "%"),
        (result.top_left[0], result.top_left[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )


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
    
    # generate result
    res = cv.matchTemplate(screenshot, needle, cv.TM_CCOEFF_NORMED)
    result = MatchResult(cv.minMaxLoc(res), needle)
    draw_result(screenshot, result, needle_path)
    if debug:
        print("comparing screenshot with %s" % needle_path)
        print(cv.minMaxLoc(res))
        cv.imwrite('result.png', screenshot)
    
    # plot result for debug
    canvas = np.zeros((screenshot.shape[0], screenshot.shape[1]), np.uint8)
    plt.figure(figsize=(16, 12))
    img = mpimg.imread("result.png")
    plt.imshow(img)
    plt.axis('on')
    plt.title("result")
    plt.imshow(cv.cvtColor(img, cv.IMREAD_GRAYSCALE))
    plt.show()
    
    plt.figure(figsize=(7, 5))
    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]
    # plt.bar(needle_path, str(int(max_val * 100)))
    plt.barh(names, values)
    plt.show()


if __name__ == '__main__':
    debug = True
    get_state("images/equipment.png")
# print(pts.image_to_string(Image.open('monitor.png')))
