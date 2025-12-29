import cv2

def crop_center(img, crop_tamanho):
    """
        Crop the center of the image.
        Parameters:
            - img: The input image to be cropped.
            - crop_tamanho: The size of the square crop.
        Returns:
            - imagem_cropped: The cropped image as a NumPy array.
    """

    # get the dimensions of the image
    altura = img.shape[0]
    largura = img.shape[1]

    # calculate the starting and ending coordinates for the crop
    start_x = largura // 2 - crop_tamanho // 2
    start_y = altura // 2 - crop_tamanho // 2
    end_x = start_x + crop_tamanho
    end_y = start_y + crop_tamanho

    # crop the image
    imagem_cropped = img[start_y:end_y, start_x:end_x]
    return imagem_cropped


def zoom_center(img, zoom_factor=1.5):
    """
        Zoom in on the center of the image.
        Parameters:
            - img: The input image to be zoomed.
            - zoom_factor: The factor by which to zoom in.
        Returns:
            - img_zoomed: The zoomed image as a NumPy array.
    """

    y_size = img.shape[0]
    x_size = img.shape[1]

    # define new boundaries
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

    # first crop image then scale
    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)