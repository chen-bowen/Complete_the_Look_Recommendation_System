# import libraries
from config import config as cfg
from matplotlib import pyplot as plt


def print_image(im1, im2, im3, im4, im5, im6, simlarity_scores):

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
    plt.title(f"1 - {simlarity_scores[0]}")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 7)

    # showing image
    plt.imshow(Image3)
    plt.axis("off")
    plt.title(f"2 - {simlarity_scores[1]}")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 8)

    # showing image
    plt.imshow(Image4)
    plt.axis("off")
    plt.title(f"3 - {simlarity_scores[2]}")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 9)

    # showing image
    plt.imshow(Image5)
    plt.axis("off")
    plt.title(f"4 - {simlarity_scores[3]}")

    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 10)

    # showing image
    plt.imshow(Image6)
    plt.axis("off")
    plt.title(f"5 - {simlarity_scores[4]}")

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
