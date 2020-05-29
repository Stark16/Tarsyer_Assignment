import os
from xml.etree import ElementTree
import random
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math

def plot_diff(Char_data):
    difference = []
    x = []
    y = []
    mean = 0
    for i in range(len(Char_data)):
        diff = abs(Char_data[0][1][1] - Char_data[i][1][1])
        difference.append([Char_data[i][0], diff])
        mean += diff
        x.append(i)
        y.append(diff)
    print(difference)
    mean = mean/len(Char_data)
    nm = 0
    for i in range(len(difference)):
        nm += (int(difference[i][1]) - mean) ** 2
    sd = math.sqrt(nm/len(difference))
    print("Mean: {}, SD: {}".format(mean, sd))

    plt.plot(x, y)
    ax = plt.gca()
    ax.set_ylim([0, 100])
    plt.show()


def arrange_nums(nums, final_sequence):

    # Sorting the sequence of numbers based on y values in ascending order:
    nums.sort(key = lambda x: x[1][1], reverse=False)

    if nums[0][1][0] < nums[1][1][0]:
        final_sequence[2] = nums[0]
        final_sequence[3] = nums[1]
    else:
        final_sequence[2] = nums[1]
        final_sequence[3] = nums[0]
    #print(final_sequence)



def Arrange(filename, Char_data, final_sequence, width, height):
    chars = []
    nums = []
    numberplate = []

    for i in range(len(Char_data)):

        if Char_data[i][0].isnumeric() == False:
            chars.append(Char_data[i])
        else:
            nums.append(Char_data[i])

    #arrange_char(chars, final_sequence)

    arrange_nums(nums, final_sequence)
    
    # print(chars)




def read_from_xml(filepath, filename):
    xml_obj = ElementTree.parse(os.path.join(filepath, filename))
    objects = xml_obj.findall('object')

    dim_obj = xml_obj.find('size')
    width = int(dim_obj.find('width').text)
    height = int( dim_obj.find('height').text)

    Char_data = []
    final_sequence = [None] * 10
    for ob in objects:
        char = ob.find('name').text


        Bounding_Box = ob.find('bndbox')
        xmin = int(Bounding_Box.find('xmin').text)
        xmax = int(Bounding_Box.find('xmax').text)
        ymin = int(Bounding_Box.find('ymin').text)
        ymax = int(Bounding_Box.find('ymax').text)

        x_center = int(xmin + abs(xmax - xmin)/2)
        y_centre = int(ymin + abs(ymax - ymin)/2)

        Centre = [x_center, y_centre]

        Bounding_Box = [xmin, xmax, ymin, ymax]
        Char_data.append([char, Centre])

        #print("* [{}], Charactor: '{}', Bounding_Box: [{}]".format(
            #       filename, char, Bounding_Box
            #    )
            #   )
    random.shuffle(Char_data)
    #print(Char_data)
    plot_diff(Char_data)
    Arrange(filename, Char_data, final_sequence, width, height)






for file in os.listdir("./Char-detection"):
    if file[-3:] == 'xml':
        read_from_xml("./Char-detection", file)


'''  img = np.zeros((height, width, 3), dtype=np.uint8) 
    cv2.imshow("img", img)
    #cv2.waitKey(0)'''