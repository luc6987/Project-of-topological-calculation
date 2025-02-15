import random
import numpy as np
from itertools import combinations
import time
from concurrent.futures import ThreadPoolExecutor

## QUESTION 1:

class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    


    def contains(self, point, tol=1e-9):
        """Check if the sphere contains the given point with tolerance for float imprecision."""
        return np.linalg.norm(self.center - np.array(point)) <= self.radius + tol

    def contains_strict(self, point, tol=1e-9):
        """Check if the sphere strictly contains the given point with tolerance for float imprecision."""
        norme = np.linalg.norm(self.center - np.array(point))
        return not (np.isclose(norme, self.radius, atol=tol) or norme > self.radius + tol)
    
    
    def onradius(self,point):
        """Check if the point is on the sphere"""
        return np.isclose(np.linalg.norm(self.center - np.array(point)),self.radius)
    

#################################################################### TASK 1 ####################################################################


def make_sphere_n_points(points):
    """
     Finds the minimal circumcircle in the d-dimensional space for n points (n <= d+1).
    
    When utilising this method, we assume that the points are not colinear and that the dimension is at least 2.

    Params :
        points: np.ndarray, shape (n, d)
    Returns :
        Sphere
    """

    nppoints = np.array(points, dtype=float)

    if len(points) > nppoints.shape[1] + 1:
        raise ValueError("Number of points must be less than or equal to the dimension of the space plus one.")

    if len(points) == 0:
        print("No points given in circumcircle calculation")
        raise None;
    if len(points) == 1:
        return Sphere(center=points[0], radius=0) 
    if len(points) == 2:
        return Sphere(center=(nppoints[0]+nppoints[1])/2, radius=np.linalg.norm(nppoints[0]-nppoints[1])/2)
    
    #print("nppoints shape:", nppoints.shape)

    # Calcul de la matrice A et du vecteur b
    diffs = nppoints[1:] - nppoints[0]  # Différences (P^n - P^0) pour n = 1, ..., N-1
    #print("diffs shape:", diffs.shape)

    A = 2 * np.dot(diffs, diffs.T)  # Matrice A
    b = np.sum(diffs ** 2, axis=1)  # Vecteur b

    # Résolution du système linéaire pour trouver les coefficients k
    k = np.linalg.solve(A, b)

    # Calculer le centre
    center = nppoints[0] + np.dot(k, diffs)

    # Calculer le rayon
    radius = np.linalg.norm(center - nppoints[0])

    # Retourner une instance de Sphere
    return Sphere(center=center, radius=radius)
    
def trivial(R,d=3):
    """Find the minimal sphere for 0, 1 or mores points."""
    if not R:
        return Sphere([0 for i in range(d)], 0)
    elif len(R) == 1:
        return Sphere(R[0], 0)
    elif len(R) >= 1:
        return make_sphere_n_points(R)

def welzl(P, R, d):
    """Recursive implementation of Welzl's algorithm for 3D."""

    if not P or len(R) == len(P[0])+   1 :
        return trivial(R,d)

    p = P.pop(random.randint(0, len(P) - 1))
    D = welzl(P, R,d)

    if D.contains(p):
        P.append(p)
        return D

    result = welzl(P, R + [p],d)
    P.append(p)
    return result

def minimal_enclosing_sphere(points):
    """Compute the minimal enclosing sphere for a set of points."""
    points = points[:]
    random.shuffle(points)
    d=len(points[0])
    return welzl(points, [],d)
   
## Question 2:

#Ici, on déduit que la filtration value de chaque simplexe est le MEB (à vérifier)

#################################################################### TASK 2 ####################################################################



def task2(points,emu):
    """Compute the filtration value for the points in emu"""
    points_chosen = [points[i] for i in emu]
    filtration = minimal_enclosing_sphere(points_chosen).radius

    return filtration



#################################################################### TASK 3 ####################################################################



    
def enum3(points,d):
    """
    Génère un tableau où chaque ligne correspond aux sous-ensembles d'une certaine taille
    """
    n = len(points)
    return [[list(comb) for comb in combinations(range(n), k)] for k in range(1, d + 2)]
  
 

def task3(points, l, printit=False):
    #input: points: list of tuples, the points in Rd
    # l: float, the maximal radius of the sphere
    # printit: bool, print the result or not
    
    #output: simplex: dict, the simplexes of dimension at most d and filtration value at most l

    d = len(points[0])
    enum = enum3(points, d)
    simplex = {tuple([i]): Sphere(points[i], 0) for i in range(len(points))}

    for i in range(1, len(enum)):
        for current_simplex in enum[i - 1]:
            for k in range(len(points)):
                if k in current_simplex:
                    continue

                pn = tuple(current_simplex + [k])
                if pn in simplex:
                    continue

                new_points = [points[idx] for idx in current_simplex] + [points[k]]
                MEB = minimal_enclosing_sphere(new_points)

                if MEB.radius < l:
                    simplex[pn] = MEB

    if printit:
        for key, value in simplex.items():
            print(f"{key} -> {value.radius}")
    return simplex




#################################################################### TASK 4 ####################################################################


def Is_in_alpha_complex(P,R):
    """Check if the simplex R is in the alpha complex of P"""
    d = len(P[0])

    #On se borne aux d+1 premiers points qui suffisent à déterminer la sphère.
    d = len(P[0])
    if len(R) > d+1:
        R_used = R[:d+1]
    else:
        R_used = R

    circum=make_sphere_n_points(R_used)
    
    for p in P:
       if circum.contains_strict(p):
           return False, circum

    return True, circum

def filtration_value(R):
    """Compute the filtration value of the simplex R"""
    return make_sphere_n_points(R)

def task4(points,R):
    """"Reuse the LP-type algorithm with new parameters in order to determine
if a simplex is in the α-complex and its filtration value. Note that this is less
standard than for the MEB, you need to explain how this new problem fits in
the framework."""
    result, circum= Is_in_alpha_complex(points,R)
    if result:
        print(f"Simplex {R} is in the alpha complex")
        return True, circum
    else:
        print(f"Simplex {R} is not in the alpha complex")
        return False, circum





#################################################################### TASK 5 ####################################################################

def enum5(points):
    """
    Génère un tableau où chaque ligne correspond aux sous-ensembles d'une certaine taille
    """
    n = len(points)
    return [[list(comb) for comb in combinations(range(n), k)] for k in range(2, len(points[0]) + 2)]
  

def task5(points, K, l):
    """ Given a set P of n points in Rd, implement an algorithm that enumerates the simplexes of dimension at most k and filtration value at most l of the α-complex and their filtration values.""" 
    #input: points: list of tuples,the points in Rd
    # K: int, the maximal dimension of the simplexes
    # l: float, the maximal filtration value

    #output: simplex: dict, the simplexes of dimension at most k and filtration value at most l of the α-complex and their filtration values
    # filtration_value: float, the maximal filtration value
    # IsSimplex: dict, the simplexes of dimension at most k and filtration value at most l of the α-complex and their filtration values
    
    enum = enum5(points)
    filtration_value = 0
    IsSimplex = {}
    simplex = {}  # on initialise le premier simplexe

    #for all simplexes possible
    for i in range(0, K):
        for j in range(len(enum[i])):
            #construction of the simplex
            pn = tuple(enum[i][j])
            simplex_tmp = [points[idx] for idx in pn]
            
            #check if the simplex is in the alpha complex
            result, _ = Is_in_alpha_complex(points, simplex_tmp)
            if not result:
                continue
            
            #check if the simplex is already in the list
            if pn in simplex:
                continue

            #check if the number of points is greater than the dimension
            if len(pn) > len(points[0])+1:
                continue
            
            #compute the filtration value of the simplex
            Circoncircle = make_sphere_n_points(simplex_tmp)

            #check if the filtration value is less than l
            if Circoncircle.radius < l:
                IsSimplex[pn] = 1
                simplex[pn] = Circoncircle
                if Circoncircle.radius > filtration_value:
                    filtration_value = Circoncircle.radius
            else:
                IsSimplex[pn] = 0

    return simplex, filtration_value, IsSimplex


