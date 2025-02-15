from Tasks import *
from scipy.spatial import distance_matrix
from plots import *

# The goal here is to implement extensions of the algorithms developed in Tasks.py

def task3_extended(points,l, printit=False):
    #adds range searching : if 2 points are farther away than 2xl, they cannot be part of the same simplex

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



class Simplex:
    def __init__(self,points,sphere):
        self.filtration=sphere.radius
        self.points = points

class Alpha_complex:
    def __init__(self,points,filtration):
        D=len(points[0]) #dimension
        n=len(points)

        dist = distance_matrix(points, points)
        sorted_distances = []
        for i in range(len(points)):
            sorted_distances.append([j for j in np.argsort(dist[i]) if dist[i][j] <= 2 * filtration])

        self.points = points
        self.filtration = filtration
        self.sorted_distances=sorted_distances
        self.neighbors = {i: [{j for j in np.argsort(dist[i]) if dist[i][j] <= 2 * filtration}]
+[set() for j in range(3,D+2)] for i in range(n)} #structure: clé est l'indice du point, la valeur est une liste d'ensemble des points voisins dans le d-simplexe
        self.filtration_values = {(i):0 for i in range(n)} #On n'utilise pas ici l'objet Simplex pour éviter des problèmes de moficiations des ensembles via al fonction de hashage
        


def task5_extended(points,l, printit=False):
    #Pour le moment, ne fonctionne qu'en dimension 2
    D=len(points[0])

    complex = Alpha_complex(points,l)


    #On commence pour la dimension 2:
    d=0

    for i_p, point in enumerate(points):
        P=complex.sorted_distances[i_p]
        iteration_list = P.copy() #on prend une copie de P pour itérer sur P tout en le modifiant
        for i_neighb in reversed(iteration_list): #On prend le point le plus éloigné, car il a beaucoup de chances de ne pas être un voisin
            
            if i_neighb not in complex.neighbors[i_p][0]: #teste si on ne l'a pas déjà enlevé (à optimiser)
                continue
            
            simp = [point,points[i_neighb]]
            circum=make_sphere_n_points(simp)

            for i_insider in P:
                insider = points[i_insider]
                if circum.contains_strict(insider):
                    complex.neighbors[i_p][d].remove(i_neighb) #ce n'est pas un voisin
                    complex.neighbors[i_neighb][d].remove(i_p) #ce n'est pas un voisin

                    break
            complex.filtration_values[(i_p,i_neighb)]=circum.radius #on ajoute la filtration value
            
            P = P[:-1] #On enlève le dernier point de P: il ne peut plus être contenu dans un autre circumcercle de i_p puisque c'était le plus loin

    if printit:
        print(complex.neighbors)
    
    simplex = {(i,neighbor):Sphere((0,0),complex.filtration_values[(i,neighbor)]) for i in range(len(points)) for neighbor in complex.neighbors[i][0]}
    plot_alpha_complex(simplex,points,l)




class Vcell:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.hyperpl=None # matrice des hyperplans
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



def test_task5_extended():
    points = [(np.random.randint(-10, 10), np.random.randint(-10, 10)) for _ in range(15)]
    print(f"Points: {points}")
    l=3
    task5_extended(points,l, printit=True)

test_task5_extended()

