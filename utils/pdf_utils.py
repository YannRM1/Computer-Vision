"""
Conversion de fichiers PDF en images numpy via PyMuPDF (fitz).
"""

import fitz
import numpy as np
import cv2


def pdf_to_images(pdf_path: str, dpi: int = 150) -> list[np.ndarray]:
    """
    Convertit chaque page d'un PDF en image BGR (numpy).

    Args:
        pdf_path: chemin vers le fichier PDF
        dpi: résolution cible en dots per inch

    Returns:
        Liste d'images BGR, une par page
    """
    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img)
    doc.close()
    return images


def count_pdf_pages(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n
