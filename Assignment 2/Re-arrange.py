import os
from xml.etree import ElementTree
import random

def arrange_char(filename, Char_data):
    chars = []
    nums = []
    numberplate = []
    for i in range(len(Char_data)):
       if Char_data[i][0].isnumeric() == False:
           chars.append(Char_data[i])
       else:
           nums.append(Char_data[i])

    for i in range(len(chars)):
        # alph, centre = chars[i]

        if chars[i][1][1] < chars[i+1][1][1]:
            if abs([i][1][0] - chars[i+1][1][0]) < abs([i][1][1] - chars[i+1][1][1]):
                numberplate.append(chars[i])


    # print(chars)




def read_from_xml(filepath, filename):
    xml_obj = ElementTree.parse(os.path.join(filepath, filename))
    objects = xml_obj.findall('object')
    Char_data = []
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
    arrange_char(filename, Char_data)






for file in os.listdir("./Char-detection"):
    if file[-3:] == 'xml':
        read_from_xml("./Char-detection", file)