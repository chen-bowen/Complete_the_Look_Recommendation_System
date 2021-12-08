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
