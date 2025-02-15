from Tasks import *
from scipy.spatial import distance_matrix

# The goal here is to implement extensions of the algorithms developed in Tasks.py

def task3_extended(points,l, printit=False):
    #adds range searching : if 2 points are farther away that 2xl, they cannot be part of the same simplex

    d=len(points[0])
    enum = enum3(points,d)
    IsSimplex = {tuple([i]): 1 for i in range(len(points))}

    dist = distance_matrix(points, points)

    simplex = {tuple([i]): Sphere(points[i], 0) for i in range(len(points))} #on initialise le premier simplexe

    
    for i in range(1,len(enum)):
        for j in range(len(enum[i-1])):
            current_simplex = enum[i-1][j]
            
            for k in range(len(points)):

                pn =tuple(current_simplex + [k])

                if k in current_simplex:
                    #utile de le préciser ? 
                    IsSimplex[pn] = 0
                    break
                
                #check, mais en soi redondant non ?
                if pn in simplex:
                    break
                
                #pré-check. Efficace si peu de points sont proches
                sub_matrix = dist[np.ix_(current_simplex, current_simplex)]
                if sub_matrix.max() > 2*l:
                    IsSimplex[tuple(current_simplex)] = 0
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

class Vcell:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.hyperpl= #matrice des hyperplans 

        self.radius = radius


def plot_voronoi(points):
    "plot voronoi cells. Tries to do it efficiently"

    d=len(points[0])
    enum = enum3(points,d)
    IsSimplex = {tuple([i]): 1 for i in range(len(points))}

    dist = distance_matrix(points, points)
    #récupérer une matrice avec les points dans l'ordre de distance
    #créer une classe de voronoi cell ? 

    simplex = {tuple([i]): Sphere(points[i], 0) for i in range(len(points))} #on initialise le premier simplexe


