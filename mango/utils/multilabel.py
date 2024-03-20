import numpy as np
from typing import List
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


def render_confusion_matrix(y_true: np.array, y_pred: np.array) -> List[np.array]:
    """
    :param y_true: (B, C) one-hot encoded labels
    :param y_pred: (B, C) one-hot encoded predictions
    :return: list of (H, W, 4) images - confusion matrix per class
    """

    conf = multilabel_confusion_matrix(np.random.randint(0, 2, (16, 10)), np.random.randint(0, 2, (16, 10)))
    ret = []
    for matrix in conf:
        matrix = matrix / np.sum(matrix)
        ConfusionMatrixDisplay(matrix).plot()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        image = Image.open(buffer)
        arr = np.array(image)
        ret.append(arr)

        buffer.close()

    return ret
