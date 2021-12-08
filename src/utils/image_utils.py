# import libraries
from config import config as cfg
from matplotlib import pyplot as plt


def bounding_box_process(img, bbox):
    """
    takes in the image and bounding box information,
    and returns bounding boxes in x_min, y_min, x_max, y_max
    """
    # get image shape and bounding box information
    img_height, img_width = img.shape()
    x, y, w, h = bbox

    # get bouding box coordinates
    x_min = img_width * x
    y_min = img_height * y

    x_max = x_min + img_width * w
    y_max = y_min + img_height * h

    return x_min, y_min, x_max, y_max


def convert_to_url(signature):
    """convert image"""
    prefix = "http://i.pinimg.com/400x/%s/%s/%s/%s.jpg"
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


def display_recommended_products(im1, im2, im3, im4, im5, im6, simlarity_scores):

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 2  # 2
    columns = 5  # 2

    # reading images
    Image1 = plt.imread(f"{cfg.DATASET_DIR}/{im1}")
    Image2 = plt.imread(f"{cfg.DATASET_DIR}/{im2}")
    Image3 = plt.imread(f"{cfg.DATASET_DIR}/{im3}")
    Image4 = plt.imread(f"{cfg.DATASET_DIR}/{im4}")
    Image5 = plt.imread(f"{cfg.DATASET_DIR}/{im5}")
    Image6 = plt.imread(f"{cfg.DATASET_DIR}/{im6}")

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
    plt.title(f"Option #1 \n Score: {round(simlarity_scores[0],2)}")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 7)

    # showing image
    plt.imshow(Image3)
    plt.axis("off")
    plt.title(f"Option #2 \n Score: {round(simlarity_scores[1],2)}")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 8)

    # showing image
    plt.imshow(Image4)
    plt.axis("off")
    plt.title(f"Option #3 \n Score: {round(simlarity_scores[2],2)}")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 9)

    # showing image
    plt.imshow(Image5)
    plt.axis("off")
    plt.title(f"Option #4 \n Score: {round(simlarity_scores[3],2)}")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 10)

    # showing image
    plt.imshow(Image6)
    plt.axis("off")
    plt.title(f"Option #5 \n Score: {round(simlarity_scores[4],2)}")

    plt.savefig("2by5.png")
    plt.show()
