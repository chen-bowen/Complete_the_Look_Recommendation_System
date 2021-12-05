# import libraries
import cv2
from matplotlib import pyplot as plt


def print_image(im1, im2, im3, im4, im5, im6):

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 2  # 2
    columns = 5  # 2

    # reading images
    Image1 = cv2.cvtColor(cv2.imread(im1), cv2.COLOR_RGB2BGR)
    Image2 = cv2.cvtColor(cv2.imread(im2), cv2.COLOR_RGB2BGR)
    Image3 = cv2.cvtColor(cv2.imread(im3), cv2.COLOR_RGB2BGR)
    Image4 = cv2.cvtColor(cv2.imread(im4), cv2.COLOR_RGB2BGR)
    Image5 = cv2.cvtColor(cv2.imread(im5), cv2.COLOR_RGB2BGR)
    Image6 = cv2.cvtColor(cv2.imread(im6), cv2.COLOR_RGB2BGR)
    # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # fig = plt.subplots(rows, columns, gridspec_kw={'height_ratios': [4, 1]})

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(Image1)
    plt.axis("off")
    plt.title("Selected Product")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 6)

    # showing image
    plt.imshow(Image2)
    plt.axis("off")
    plt.title("Suggestion 1")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 7)

    # showing image
    plt.imshow(Image3)
    plt.axis("off")
    plt.title("Suggestion 2")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 8)

    # showing image
    plt.imshow(Image4)
    plt.axis("off")
    plt.title("Suggestion 3")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 9)

    # showing image
    plt.imshow(Image5)
    plt.axis("off")
    plt.title("Suggestion 4")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 10)

    # showing image
    plt.imshow(Image6)
    plt.axis("off")
    plt.title("Suggestion 5")

    plt.savefig("2by5.png")
    plt.show()


if __name__ == "__main__":
    Image1 = "Data/1.png"
    Image2 = "Data/1.png"
    Image3 = "Data/1.png"
    Image4 = "Data/1.png"
    Image5 = "Data/1.png"
    Image6 = "Data/1.png"
    print_image(Image1, Image2, Image3, Image4, Image5, Image6)
