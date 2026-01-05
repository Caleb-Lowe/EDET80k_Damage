import perspectivemap as pm
import numpy as np
from abc import ABC, abstractmethod


class ROI(ABC):

    xmin = None
    xmax = None
    ymin = None
    ymax = None

    def getBounds(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @abstractmethod
    def onBorder(self, x, y):
        pass

    @abstractmethod
    def inROI(self, x, y):
        pass
    
    @abstractmethod
    def area(self):
        pass


class RectangleROI(ROI):
    # Initialized with the top-left and bottom-right corners
    def __init__(self, corner1, corner2):
        if (len(corner1) != 2) or (len(corner2) != 2):
            raise Exception("Points must have the format [x,y]")
        self.xmin = np.min((corner1[0],corner2[0]))
        self.xmax = np.max((corner1[0],corner2[0]))
        self.ymin = np.min((corner1[1],corner2[1]))
        self.ymax = np.max((corner1[1],corner2[1]))

    def onBorder(self, x, y):
        if (x == self.xmin or x == self.xmax) and y >= self.ymin and y <= self.ymax:
            return True
        elif (y == self.ymin or y == self.ymax) and x >= self.xmin and x <= self.xmax:
            return True
        return False
    
    def inROI(self, x, y):
        return (x > self.xmin and x < self.xmax) and (y > self.ymin and y < self.ymax)
    
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
    
CHIP = RectangleROI(np.flip(pm.get_chip_corners()[0]), np.flip(pm.get_chip_corners()[2]))
BRICK = RectangleROI(np.flip(pm.get_target_corners()[0]), np.flip(pm.get_target_corners()[2]))
FBRICK = RectangleROI(np.flip(pm.get_fbrick_corners()[0]), np.flip(pm.get_fbrick_corners()[2]))



class PolyROI(ROI):
    # Initialized with a list of vertices (points)
    def __init__(self, points):
        self.vertices = np.array(points)
        self.ymin = np.min(self.vertices[:,0])
        self.ymax = np.max(self.vertices[:,0])
        self.xmin = np.min(self.vertices[:,1])
        self.xmax = np.max(self.vertices[:,1])
        self.degree = len(self.vertices) # Number of sides of the polygon
    
    def onBorder(self, x, y):
        for i in range(self.degree):
            next_i = (i + 1) % self.degree
            # Vertical line case
            if (self.vertices[i][1] == self.vertices[next_i][1]):
                if (x == self.vertices[i][1]) and (y >= min(self.vertices[i][0], self.vertices[next_i][0])) and (y <= max(self.vertices[i][0], self.vertices[next_i][0])):
                    return True
            # Horizontal line case
            elif (self.vertices[i][0] == self.vertices[next_i][0]):
                if (y == self.vertices[i][0]) and (x >= min(self.vertices[i][1], self.vertices[next_i][1])) and (x <= max(self.vertices[i][1], self.vertices[next_i][1])):
                    return True
            # Diagonal line case
            else:
                # Define a line between two points and check if (x,y) is on that line segment
                linex = lambda x: (self.vertices[next_i][0] - self.vertices[i][0]) / (self.vertices[next_i][1] - self.vertices[i][1]) * (x - self.vertices[i][1]) + self.vertices[i][0]
                liney = lambda y: (self.vertices[next_i][1] - self.vertices[i][1]) / (self.vertices[next_i][0] - self.vertices[i][0]) * (y - self.vertices[i][0]) + self.vertices[i][1]
                if (x >= min(self.vertices[i][1], self.vertices[next_i][1])) and (x <= max(self.vertices[i][1], self.vertices[next_i][1])):
                    if np.isclose(y, linex(x), atol=0.5):
                        return True
                if (y >= min(self.vertices[i][0], self.vertices[next_i][0])) and (y <= max(self.vertices[i][0], self.vertices[next_i][0])):
                    if np.isclose(x, liney(y), atol=0.5):
                        return True
        return False
    
    def inROI(self, x, y):
        # Ray-casting algorithm to determine if point is in polygon
        inside = False
        p1y, p1x = self.vertices[0]
        for i in range(self.degree + 1):
            p2y, p2x = self.vertices[i % self.degree]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def area(self):
        # Calculate area using the shoelace formula
        area = 0
        for i in range(self.degree):
            j = (i + 1) % self.degree
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        area = abs(area) / 2.0
        return area
    



class EllipseROI(ROI):
    # Initialized with the center, and the length of the semi-major/minor axes. Vertex and co-vertix must be located on the x/y axes
    def __init__(self, center, x_axis, y_axis):
        self.y, self.x = center
        self.x_axis = x_axis
        self.y_axis = y_axis        
        self.xmax = center[1] + x_axis
        self.xmin = center[1] - x_axis
        self.ymax = center[0] + y_axis
        self.ymin = center[0] - y_axis

    def onBorder(self, x, y):
        return np.isclose(((x - self.x) ** 2 / (self.x_axis **2)) + ((y - self.y) ** 2 / (self.y_axis **2)), 1,
                          atol=0.02)
    
    def inROI(self, x, y):
        return (((x - self.x) ** 2 / (self.x_axis **2)) + ((y - self.y) ** 2 / (self.y_axis **2))) < 1
    
    def area(self):
        return np.pi * self.x_axis * self.y_axis
    



class superROI(ROI):
    # initialized with a list of ROIs to combine
    def __init__(self, rois, mode='union'):
        self.rois = rois
        self.mode = mode  # 'union', 'subtract', or 'intersect'
        if mode == 'union':
            self.xmin = min([roi.xmin for roi in rois])
            self.xmax = max([roi.xmax for roi in rois])
            self.ymin = min([roi.ymin for roi in rois])
            self.ymax = max([roi.ymax for roi in rois])
        elif mode == 'subtract':
            self.xmin = rois[0].xmin
            self.xmax = rois[0].xmax
            self.ymin = rois[0].ymin
            self.ymax = rois[0].ymax
        elif mode == 'intersect':
            self.xmin = max([roi.xmin for roi in rois])
            self.xmax = min([roi.xmax for roi in rois])
            self.ymin = max([roi.ymin for roi in rois])
            self.ymax = min([roi.ymax for roi in rois])
        else:
            raise Exception("Invalid mode for superROI. Use 'union', 'subtract', or 'intersect'.")

    def onBorder(self, x, y):
        if self.mode == 'union':
            # A point is on the border if it is on the border of any ROI and not inside any other ROI
            for roi in self.rois:
                if roi.onBorder(x, y):
                    for other_roi in self.rois:
                        if other_roi != roi and other_roi.inROI(x, y):
                            return False
                    return True
                else:
                    return False
            
        if self.mode == 'subtract':
            # A point is on the border if it is on the border of the first ROI and not inside any of the other ROIs,
            # # or if it is on the border of any of the other ROIs and inside the first ROI
            if self.rois[0].onBorder(x, y):
                for roi in self.rois[1:]:
                    if roi.inROI(x, y):
                        return False
                return True
            else:
                for roi in self.rois[1:]:
                    if roi.onBorder(x, y) and self.rois[0].inROI(x, y):
                        return True
                return False
            
        if self.mode == 'intersect':
            # A point is on the border if it is on the border of any ROI and inside all other ROIs
            for roi in self.rois:
                if roi.onBorder(x, y):
                    inside_all = True
                    for other_roi in self.rois:
                        if other_roi != roi and not other_roi.inROI(x, y):
                            inside_all = False
                            break
                    if inside_all:
                        return True
            return False
        
    def inROI(self, x, y):
        if self.mode == 'union':
            for roi in self.rois:
                if roi.inROI(x, y):
                    return True
            return False
        elif self.mode == 'subtract':
            if self.rois[0].inROI(x, y):
                for roi in self.rois[1:]:
                    if roi.inROI(x, y):
                        return False
                return True
            return False
        elif self.mode == 'intersect':
            for roi in self.rois:
                if not roi.inROI(x, y):
                    return False
            return True
        
    def area(self):
        print("Warning: Area calculation for superROI is approximate and may be inaccurate for complex shapes.")
        if self.mode == 'union':
            # Approximate area by summing individual areas (may overcount overlaps)
            total_area = 0
            for roi in self.rois:
                total_area += roi.area()
            return total_area
        elif self.mode == 'subtract':
            total_area = self.rois[0].area()
            for roi in self.rois[1:]:
                total_area -= roi.area()
            return total_area
        elif self.mode == 'intersect':
            # Approximate area by taking the minimum area (may overcount)
            min_area = float('inf')
            for roi in self.rois:
                area = roi.area()
                if area < min_area:
                    min_area = area
            return min_area