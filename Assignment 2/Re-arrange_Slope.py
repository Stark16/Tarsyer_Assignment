import os
from xml.etree import ElementTree
import random
from matplotlib import pyplot as plt


# check_shape method:
# 1. Takes Char_data and dimensions of image as input.
# 2. Returns 0 if license plate has 1 line or returns non zero value of it has 2:
def check_shape(Char_data, width, height):
    jumps = 0 # Jumps is incremented if there is a massive change between 2 successive y values of characters
    x = []
    y = []

    # Separating ordinates x and y:
    for i in range(len(Char_data)):
        x.append(Char_data[i][1][0])
        y.append(Char_data[i][1][1])

    #sorting x and y ordinates from low to high to get a uni-directional curve.
    x.sort(reverse=True)
    y.sort(reverse=True)

    # This loop checks if at any point the line joining 2 points has more y slope than x slope
    # if yes then it is considered as a jump variable is incremented.
    for i in range(len(x)):

        try:
            x1 = x[i]
            x2 = x[i + 1]
            y1 = y[i]
            y2 = y[i + 1]
            if abs(x2 - x1) <= abs(y2 - y1):
                jumps += 1
        except:
            continue

    # Plotting the curves for visual representation:
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_ylim([0, height])
    ax.set_xlim([0, width])
    plt.title('Curve of X vs Y of Characters centers')
    plt.show()

    return 1 if jumps >= 1 else 0



# Arrange Method:
# 1. Takes filename, filename, Char_data, width, height, shape_code as parameters
# 2. Arranges and stores the final_sequence of the lincense plate
# 3. returns the final sequence value

def Arrange(filename, Char_data, width, height, shape_code):
    final_sequence = []

    # if shape_code is 0 that means the plot between x and y was almost linear.
    # This indicates that all the characters in license plate are in 1 line and
    # we can sort the characters in increasing order of their x-ordinate of the centers.
    if shape_code == 0:
        Char_data.sort(key=lambda x: x[:][1][0], reverse=False)
        for i in range(len(Char_data)):
            final_sequence.append(Char_data[i][0])
        final_sequence[:] = [''.join(final_sequence[:])]
        return final_sequence


    # However, if shape_code is non zero this indicates that there is a big value shift
    # in y-axis indicating that there are 2 lines in the license plate.
    else:
        line1 = []      # Stores the characters from top line.
        line2 = []      # Stores the characters from bottom line.

        # calculating Mean of y values to determine if a character belongs to top line or bottom line:
        mean = 0
        for i in range(len(Char_data)):
            mean += Char_data[i][1][1]
        mean /= len(Char_data)
        print(mean)

        # sorting the characters between line1 and line2 by comparing their y-values with mean
        for i in range(len(Char_data)):
            line1.append(Char_data[i]) if Char_data[i][1][1] < mean else line2.append(Char_data[i])
        line1.sort(key=lambda x: x[:][1][0], reverse=False)
        line2.sort(key=lambda x: x[:][1][0], reverse=False)

        # Appending and joining everything to final_sequence list and returning the same
        for i in range(len(line1)):
            final_sequence.append(line1[i][0])
        for i in range(len(line2)):
            final_sequence.append((line2[i][0]))
        final_sequence[:] = ["".join(final_sequence[:])]
        return final_sequence


# read_from_xml function:
# Responsible for following:
# 1. Read the data from the xml file and store it in 'Char_data' array the instructed format
# 2. The Char_data array is passed through 2 more functions to determine the correct sequence
#    of the license plate characters.

def read_from_xml(filepath, filename):
    xml_obj = ElementTree.parse(os.path.join(filepath, filename))  # created the xml object
    objects = xml_obj.findall('object')

    dim_obj = xml_obj.find('size')
    width = int(dim_obj.find('width').text)
    height = int(dim_obj.find('height').text)

    Char_data = []  # initializing the Char_data array to store:
    # 1. Characters, 2. Centre of the bounding
    # box w.r.t origin
    for ob in objects:
        char = ob.find('name').text

        Bounding_Box = ob.find('bndbox')
        xmin = int(Bounding_Box.find('xmin').text)  # Extracting bounding box coordinates
        xmax = int(Bounding_Box.find('xmax').text)
        ymin = int(Bounding_Box.find('ymin').text)
        ymax = int(Bounding_Box.find('ymax').text)

        x_center = int(xmin + abs(xmax - xmin) / 2)  # Calculating x and y centres
        y_centre = int(ymin + abs(ymax - ymin) / 2)

        Centre = [x_center, y_centre]

        Char_data.append([char, Centre])  # Storing the character data in array

        # print("* [{}], Charactor: '{}', Bounding_Box: [{}]".format(
        #       filename, char, Bounding_Box
        #    )
        #   )
    random.shuffle(Char_data)  # shuffling the final data array for emulating
                               # the expected behaviour with an OCR

    # Final sequence array that will be used to store the final sequence of license plate:
    final_sequence = []


    # Shape Code is used to determine the shape of the number plate, I.e 1 line or 2 lined plate:

    shape_code = check_shape(Char_data, width, height)    # check_shape function determines No. of lines in plate.

    print(shape_code)

    # Arrange arranges and returns the characters by choosing the method suggested by the shape_code value:

    final_sequence = Arrange(filename, Char_data, width, height, shape_code)

    print("The Correct sequence for license plate for file: {}, is: {}".format(filename, final_sequence))


# Main function has a for Loop to loop through all the xml files in the specified directory:
def main():
    for file in os.listdir("./Char-detection"):
        if file[-3:] == 'xml':
            read_from_xml("./Char-detection", file)  # File path and Filename are sent to read_from_xml to be
            # further processed on.


main()
