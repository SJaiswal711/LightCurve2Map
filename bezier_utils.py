# Code from: https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib
# This file is required for generating bezier curve in other files.
import numpy as np
from scipy.special import binom
import random
from skimage.draw import polygon

random.seed(10)

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0, seed=None):
    """
    Given an array of points *a*, create a Bezier-like curve through those points.
    
    Parameters
    ----------
    a : ndarray
        Input 2D points of shape (N, 2).
    rad : float
        Steering parameter for control point distance (0 to 1).
    edgy : float
        How 'edgy' the curve should be. 0 is the smoothest.
    seed : int or None
        Optional seed for reproducibility if randomness is involved.
        
    Returns
    -------
    x, y : ndarray
        X and Y coordinates of the generated curve.
    a : ndarray
        Extended input array with angle information.
    """
    if seed is not None:
        np.random.seed(seed)

    p = np.arctan(edgy) / np.pi + 0.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])

    # Normalize angles to [0, 2pi]
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)

    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)

    # Now call get_curve (assumes it supports randomness or noise)
    s, c = get_curve(a, r=rad, method="var", numpoints=30)
    x, y = c.T
    return x, y, a



def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


def shape_to_mask(x, y, size=38):
    rr, cc = polygon(y * (size - 1), x * (size - 1), shape=(size, size))
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask

def normalize_to_unit_square(x, y):
    """Normalize x and y to fit in [0,1] x [0,1] while preserving aspect ratio."""
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    scale = 1.0 / max(x_range, y_range)

    x = (x - np.min(x)) * scale
    y = (y - np.min(y)) * scale

    # Center padding
    x_pad = (1 - (np.max(x) - np.min(x))) / 2
    y_pad = (1 - (np.max(y) - np.min(y))) / 2
    x += x_pad - np.min(x)
    y += y_pad - np.min(y)

    return x, y
