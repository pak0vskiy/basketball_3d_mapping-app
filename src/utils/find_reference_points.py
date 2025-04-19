import cv2
import numpy as np

def find_points(img):

    # Load binary mask
    mask = img
    # mask = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # if mask is None:
    #     raise ValueError("Mask Image not Found")

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (assumed to be the court)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    court_contour = contours[0]


    # Get the convex hull to remove noise
    hull = cv2.convexHull(court_contour)

    # Approximate the contour with fewer points
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)


    # Convert to a list of (x, y) points
    points = np.squeeze(approx)

    # Sort points: leftmost & rightmost (baseline), and top of key/3PT line
    points = sorted(points, key=lambda p: (p[1], p[0]))  # Sort by y first, then x

    # Assuming a half-court view, take four points:
    top_two = sorted(points[:2], key=lambda p: p[0])  # Left & right top
    bottom_two = sorted(points[-2:], key=lambda p: p[0])  # Left & right bottom

    # Final four reference points (sorted: TR, BR, BL, TL)
    reference_points = np.array([top_two[1], bottom_two[1], bottom_two[0], top_two[0]], dtype=np.float32)

    
    return reference_points

