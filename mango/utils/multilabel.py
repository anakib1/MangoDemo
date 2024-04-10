import numpy as np
from typing import List
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


def render_one_matrix(matrix: np.array):
    matrix = matrix / np.sum(matrix)
    ConfusionMatrixDisplay(matrix).plot()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    image = Image.open(buffer)
    arr = np.array(image)

    buffer.close()
    return arr


def render_confusion_matrix(y_true: np.array, y_pred: np.array) -> List[np.array]:
    """
    :param y_true: (B, C) one-hot encoded labels
    :param y_pred: (B, C) one-hot encoded predictions
    :return: list of (H, W, 4) images - confusion matrix per class
    """

    conf = multilabel_confusion_matrix(y_true, y_pred)
    ret = []
    for matrix in conf:
        ret.append(render_one_matrix(matrix))

    return ret


def render_confusion_matrix_sololabel(y_true: np.array, y_pred: np.array) -> np.array:
    """
    :param y_true: (B, C) one-hot encoded labels
    :param y_pred: (B, C) one-hot encoded predictions
    :return: list of (H, W, 4) images - confusion matrix per class
    """

    matrix = confusion_matrix(y_true, y_pred)
    return render_one_matrix(matrix)
