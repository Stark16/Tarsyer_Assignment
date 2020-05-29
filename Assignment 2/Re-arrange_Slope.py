import os
from xml.etree import ElementTree
import random
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math


def check_shape(Char_data, width, height):
    jumps = 0
    x = []
    y = []

    for i in range(len(Char_data)):
        x.append(Char_data[i][1][0])
        y.append(Char_data[i][1][1])

    x.sort(reverse=True)
    y.sort(reverse=True)

    for i in range(len(x)):
        try:
            x1 = (x[i])
            y1 = (y[i])
            x2 = (x[i+1])
            y2 = (y[i+1])
            if abs(x2 - x1) <= abs(y2 - y1):
                jumps += 1
        except:
            continue
    #print(x, y)
    #print(jumps)

    '''for i in range(len(Char_data)):
        try:
            x1 = (Char_data[i][1][0])
            x.append(x1)
            y1 = (Char_data[i][1][1])
            y.append(y1)
            x2 = (Char_data[i+1][1][0])
            y2 = (Char_data[i+1][1][1])
            if abs(x2 - x1) <= abs(y2 - y1):
                    jumps += 1
        except:
            continue'''
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_ylim([0, height])
    ax.set_xlim([0, width])
    plt.show()

    return 1 if jumps >= 1 else 0


'''def arrange_nums(nums, final_sequence):
    # Sorting the sequence of numbers based on y values in ascending order:
    nums.sort(key=lambda x: x[1][1], reverse=False)

    if nums[0][1][0] < nums[1][1][0]:
        final_sequence[2] = nums[0]
        final_sequence[3] = nums[1]
    else:
        final_sequence[2] = nums[1]
        final_sequence[3] = nums[0]
    # print(final_sequence)
'''

def Arrange(filename, Char_data, width, height, shape_code):
    final_sequence = []

    if shape_code != 1:
        Char_data.sort(key=lambda x: x[:][1][0], reverse=False)
        for i in range(len(Char_data)):
            final_sequence.append(Char_data[i][0])
        final_sequence[:] = [''.join(final_sequence[:])]
        print(final_sequence)
    else:
        chars = []
        nums = []
        for i in range(len(Char_data)):

            if Char_data[i][0].isnumeric() == False:
                chars.append(Char_data[i])
            else:
                nums.append(Char_data[i])

        # arrange_char(chars, final_sequence)

        # arrange_nums(nums, final_sequence)

        # print(chars)


def read_from_xml(filepath, filename):
    xml_obj = ElementTree.parse(os.path.join(filepath, filename))
    objects = xml_obj.findall('object')

    dim_obj = xml_obj.find('size')
    width = int(dim_obj.find('width').text)
    height = int(dim_obj.find('height').text)

    Char_data = []
    for ob in objects:
        char = ob.find('name').text

        Bounding_Box = ob.find('bndbox')
        xmin = int(Bounding_Box.find('xmin').text)
        xmax = int(Bounding_Box.find('xmax').text)
        ymin = int(Bounding_Box.find('ymin').text)
        ymax = int(Bounding_Box.find('ymax').text)

        x_center = int(xmin + abs(xmax - xmin) / 2)
        y_centre = int(ymin + abs(ymax - ymin) / 2)

        Centre = [x_center, y_centre]

        Bounding_Box = [xmin, xmax, ymin, ymax]
        Char_data.append([char, Centre])

        # print("* [{}], Charactor: '{}', Bounding_Box: [{}]".format(
        #       filename, char, Bounding_Box
        #    )
        #   )
    random.shuffle(Char_data)
    final_sequence = []


    # print(Char_data)
    shape_code = check_shape(Char_data, width, height)
    print(shape_code)
    Arrange(filename, Char_data, width, height, shape_code)


for file in os.listdir("./Char-detection"):
    if file[-3:] == 'xml':
        read_from_xml("./Char-detection", file)

'''  img = np.zeros((height, width, 3), dtype=np.uint8) 
    cv2.imshow("img", img)
    #cv2.waitKey(0)'''