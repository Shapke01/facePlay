from imutils import face_utils
from scipy.spatial import Delaunay
from itertools import combinations
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def getRectangles(img, detector):
    rect = detector(img, 0)
    return rect

def getCoordinates(img, rect, predictor):
    x = predictor(img, rect)
    coords = face_utils.shape_to_np(x)
    return coords

def drawCoords(img):
    tmp = img.copy()
    rects = getRectangles(tmp, detector)
    for rect in rects:
        coords = getCoordinates(tmp, rect, predictor)
        for x,y in coords:
            cv2.circle(tmp, (x,y), radius=2, color=(255, 0, 255), thickness=-1)

    return tmp

def drawLines(img):
    tmp = img.copy()
    rects = getRectangles(tmp, detector)
    for rect in rects:
        coords = getCoordinates(tmp, rect, predictor)
        trianglesIndices = Delaunay(coords).simplices
        trianglesCoords = coords[trianglesIndices]
        for tri in trianglesCoords:
            cv2.line(tmp, tri[0], tri[1], color=(255, 0, 255), thickness=1)
            cv2.line(tmp, tri[1], tri[2], color=(255, 0, 255), thickness=1)
            cv2.line(tmp, tri[0], tri[2], color=(255, 0, 255), thickness=1)

    return tmp

def getTriIndicesForAllFaces(img):
    triIndicesForAll = []
    rects = getRectangles(img, detector)

    for rect in rects:
        coords = getCoordinates(img, rect, predictor)
        trianglesIndices = Delaunay(coords).simplices
        triIndicesForAll.append(trianglesIndices)
    
    return np.array(triIndicesForAll)

def getTrianglesCoords(img, trianglesIndices=None, detector=detector, predictor=predictor):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = getRectangles(gray, detector=detector)
    if len(rects) != 1:
        return None, None
    rect = rects[0]
    keypointsCoords = getCoordinates(gray, rect, predictor=predictor)
    
    if type(trianglesIndices) == type(None):
        trianglesIndices = Delaunay(keypointsCoords).simplices
        
    trianglesCoords = keypointsCoords[trianglesIndices].astype(np.float32)
    
    return trianglesIndices, trianglesCoords

def mapT2T(src_img, dest_img, src_vertices, dest_vertices):
    dest_x, dest_y, dest_w, dest_h = cv2.boundingRect(dest_vertices)
    src_x, src_y, src_w, src_h = cv2.boundingRect(src_vertices)

    dest_vertices_tmp = dest_vertices.copy()
    src_vertices_tmp = src_vertices.copy()
    dest_vertices_tmp[:,0] -= np.min(dest_vertices[:,0])
    src_vertices_tmp[:,0] -= np.min(src_vertices[:,0])
    dest_vertices_tmp[:,1] -= np.min(dest_vertices[:,1])
    src_vertices_tmp[:,1] -= np.min(src_vertices[:,1])

    dest_tmp = dest_img.copy()
    src_cut = src_img[src_y:src_y+src_h,
                      src_x:src_x+src_w]

    # transformation Matrix
    M = cv2.getAffineTransform(src_vertices_tmp, dest_vertices_tmp)

    # transform source image
    transformed = cv2.warpAffine(src_cut, M, (dest_w, dest_h))
    #plt.imshow(img_tri)
    # use mask to copy transformed image part to
    # destination image
    mask = np.zeros((dest_h, dest_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dest_vertices_tmp.astype(int), (255))
    dest_tmp[dest_y:dest_y+dest_h,
             dest_x:dest_x+dest_w][mask != 0] = transformed[mask != 0]
    
    return dest_tmp

def swapAll(src_image, dest_image, src_coords, dest_coords):
    tmp = dest_image.copy()
    for vertices, new_vertices in zip(src_coords, dest_coords):
        tmp = mapT2T(src_image, tmp, vertices, new_vertices)
    return tmp

def numpyToFile(array, filename):
    np.savetxt(filename, array, delimiter=',', fmt='%d')

def fileToNumpy(filename):
    return np.genfromtxt(filename, delimiter=",", dtype=np.int32)