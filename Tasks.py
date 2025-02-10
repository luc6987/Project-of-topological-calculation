import random
import numpy as np
from itertools import combinations

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



    
def enum3(points):
    """
    Génère un tableau où chaque ligne correspond aux sous-ensembles d'une certaine taille.
    """
    n = len(points)
    return [[list(comb) for comb in combinations(range(n), k)] for k in range(1, n + 1)]
  
 

def task3(points,l, printit=False):
    enum = enum3(points)
    IsSimplex = {tuple([i]): 1 for i in range(len(points))}

    simplex = {tuple([i]): Sphere(points[i], 0) for i in range(len(points))} #on initialise le premier simplexe

    
    for i in range(1,len(enum)):
        for j in range(len(enum[i-1])):
            current_simplex = enum[i-1][j]

            for k in range(len(points)):

                pn =tuple(current_simplex + [k])

                if k in current_simplex:
                    IsSimplex[pn] = 0
                    break

                if pn in simplex:
                    break

                new_points = [points[idx] for idx in current_simplex] + [points[k]]

                MEB = minimal_enclosing_sphere(new_points)

                if MEB.radius < l:
                    simplex[pn] = MEB
                    IsSimplex[pn] = 1
                else:
                    IsSimplex[pn] = 0

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

def task5(points,K,l):
      """ Given a set P of n points in Rd, implement an algorithm that enu-
merates the simplexes of dimension at most k and filtration value at most l of
the α-complex and their filtration values.""" 
      enum = enum3(points)
      #ca me parait très lourd de print enum
      print(f"enum={enum}")
      filtration_value=0
      IsSimplex = {tuple([i]): 1 for i in range(len(points))}

      simplex = {tuple([i]): Sphere(points[i], 0) for i in range(len(points))} #on initialise le premier simplexe

      for i in range(1,K):
          for j in range(len(enum[i-1])):
              
              current_simplex = enum[i-1][j]

              for k in range(len(points)):
                  
                  if k in current_simplex:
                      break
                  
                  pn =tuple(current_simplex + [k])

                  if not Is_in_alpha_complex(points,pn):
                    IsSimplex[pn] = 0
                    break

                  if pn in simplex:
                    break

                  new_simplex = [points[idx] for idx in current_simplex] + [points[k]]

                  #ici, on utilise trivial et pas make_sphere... ? 
                  MEB = trivial(new_simplex)

                  if MEB.radius < l:
                      if MEB.radius > filtration_value:
                        filtration_value=MEB.radius
                      simplex[pn] = MEB
                      IsSimplex[pn] = 1
                  else:
                      IsSimplex[pn] = 0
                      break

                  print(f"new simplex: {pn} -> {MEB.radius} with filtration value {filtration_value}")

# Test cases
def test_task1():
    """Test cases for minimal enclosing sphere."""
    # Test 1: Single point
    points = [(0, 0, 0)]
    sphere = minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 0)
    print("Test 1.1 passed!")

    # Test 2: Two points
    points = [(0, 0, 0), (2, 0, 0)]
    sphere = minimal_enclosing_sphere(points)
    print(sphere.center)
    assert np.allclose(sphere.center, [1, 0, 0])
    assert np.isclose(sphere.radius, 1)
    print("Test 1.2 passed!")

    # Test 3: Three points
    points = [(-10, 0, 0), (10, 0, 0), (0, 1, 0)]
    sphere = minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 10)
    print("Test 1.3 passed!")

    # Test 4: Four points
    points = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    sphere =  minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, np.sqrt(26))
    print("Test 1.4 passed!")

    print("All test cases passed!")


# Test cases
def test_task2():
        P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]

        enu=[0]
        assert np.allclose(task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[1]
        assert np.allclose( task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[2]
        assert np.allclose( task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[3]
        assert np.allclose( task2(P,enu), 0)
        print(f"Test({enu})passed!")

        enu=[2,1]
        assert np.allclose( task2(P,enu), 3.53553)   
        print(f"Test({enu})passed!")

        enu=[1,0]
        assert np.allclose( task2(P,enu), 3.67425)   
        print(f"Test({enu})passed!")

        enu=[3,2]
        assert np.allclose( task2(P,enu), 4)   
        print(f"Test({enu})passed!")

        enu=[2,0]
        assert np.allclose( task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[3,0]
        assert np.allclose( task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[2,1,0]
        assert np.allclose( task2(P,enu), 4.39525)   
        print(f"Test({enu})passed!")

        enu=[3,2,0]
        assert np.allclose( task2(P,enu), 4.71495)   
        print(f"Test({enu})passed!")

        enu=[3,1]
        assert np.allclose( task2(P,enu), 4.94975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1]
        assert np.allclose( task2(P,enu), 5)   
        print(f"Test({enu})passed!")

        enu=[3,1,0]
        assert np.allclose( task2(P,enu), 5.04975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1,0]
        assert np.allclose( task2(P,enu), 5.09902)   
        print(f"Test({enu})passed!")

        print("Test 2 all passed! ")
        
def test_task3():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    task3(P,1000)

def test_task4():
    P=[(0,5,0),(3,4,0),(-3,4,0)]
    R=P
    print(f"---- Test for {P}")
    a= task4(P,R)
    print(f"Complex ? {a[0]}")
    print(f"filtration value: {a[1]}")

    P.append((0,0,4))
    R=P
    print(f"---- Test for {P}")
    a= task4(P,R)
    print(f"Complex ? {a[0]}")
    print(f"filtration value: {a[1]}")

    P.append((0,0,-4))
    print(f"---- Test for {P}")
    a= task4(P,R)
    print(f"Complex ? {a[0]}")
    print(f"filtration value: {a[1]}")

def task5(points,K,l):
      """ Given a set P of n points in Rd, implement an algorithm that enu-
      merates the simplexes of dimension at most k and filtration value at most l of
      the α-complex and their filtration values.""" 
      enum = enum3(points)
      print(f"enum={enum}")
      filtration_value=0
      IsSimplex = {tuple([i]): 1 for i in range(len(points))}

      simplex = {tuple([i]): Sphere(points[i], 0) for i in range(len(points))} #on initialise le premier simplexe

      for i in range(1,K):
        for j in range(len(enum[i-1])):
              
          current_simplex = enum[i][j]

          for k in range(len(points)):
                  
            if k in current_simplex:
              break
                        
            pn = current_simplex.append(k)

            is_alpha, circum = Is_in_alpha_complex(points,[points[idx] for idx in current_simplex])

            if not is_alpha:
              IsSimplex[pn] = 0
              break

            if pn in simplex:
              break

            new_simplex = [points[idx] for idx in current_simplex] + [points[k]]

            MEB = trivial(new_simplex)

            if MEB.radius < l:
              if MEB.radius > filtration_value:
                        filtration_value=MEB.radius
              simplex[pn] = MEB
              IsSimplex[pn] = 1
            else:
              IsSimplex[pn] = 0
              break

            print(f"new alpha-simplex: {pn} -> {MEB.radius} ")
      print(f"filtration value: {filtration_value}")
      return None


