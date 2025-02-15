#test en dim 2

from Tasks import *
from test import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points_sphere(points, sphere, ax, color_black=False):
    """Plot the points and the minimal enclosing sphere."""
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], label='Points')
    if color_black:
         col='b'
    else:
         col='r'
    
    
    circle = plt.Circle(sphere.center[:2], sphere.radius, color=col, fill=False, label='Sphere')
    ax.add_artist(circle)
    
    ax.set_xlim(min(points[:, 0]) - sphere.radius - 1, max(points[:, 0]) + sphere.radius + 1)
    ax.set_ylim(min(points[:, 1]) - sphere.radius - 1, max(points[:, 1]) + sphere.radius + 1)
    ax.set_aspect('equal', 'box')
    ax.legend()

def plot_points_sphere_3d(points, sphere, ax):
    """Plot the points and the minimal enclosing sphere in 3D."""
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Points')
    
    # Create a sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = sphere.center[0] + sphere.radius * np.cos(u) * np.sin(v)
    y = sphere.center[1] + sphere.radius * np.sin(u) * np.sin(v)
    z = sphere.center[2] + sphere.radius * np.cos(v)
    
    ax.plot_wireframe(x, y, z, color='r', alpha=0.1, label='Sphere')
    
    ax.set_xlim(min(points[:, 0]) - sphere.radius - 1, max(points[:, 0]) + sphere.radius + 1)
    ax.set_ylim(min(points[:, 1]) - sphere.radius - 1, max(points[:, 1]) + sphere.radius + 1)
    ax.set_zlim(min(points[:, 2]) - sphere.radius - 1, max(points[:, 2]) + sphere.radius + 1)
    ax.set_aspect('auto')
    
    for point in points:
        if sphere.contains(point):
            if sphere.onradius(point):
                ax.scatter(point[0], point[1], point[2], color='g', label='On Sphere')
            else:
                ax.scatter(point[0], point[1], point[2], color='b', label='Inside Sphere')
        else:
            ax.scatter(point[0], point[1], point[2], color='r', label='Outside Sphere')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())



def test_and_plot_minimal_enclosing_sphere_2D():
    """Generate 5 tests with different points in 2 dimensions and plot the results."""
    #here, we use points in 3 dimensions and set the 3rd dimension to 0 to be able to run our circumsphere algorithm
    test_cases = [
        [(0, 0, 0), (1, 1, 0)],
        [(0, 0, 0), (2, 0, 0), (1, 1, 0)],
        [(0, 0, 0), (2, 0, 0), (1, 1, 0), (1, -1, 0)],
        [(0, 0, 0), (2, 0, 0), (1, 1, 0), (1, -1, 0), (0, 2, 0)],
        [(0, 0, 0), (2, 0, 0), (1, 1, 0), (1, -1, 0), (0, 2, 0), (2, 2, 0)]
    ]
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, points in enumerate(test_cases):
        sphere = minimal_enclosing_sphere(points)
        plot_points_sphere(points, sphere, axs[i])
        axs[i].set_title(f'Test {i+1}')
    
    plt.show()


def test_and_plot_minimal_enclosing_sphere_3d():
    """Generate 5 tests with different points in 3 dimensions and plot the results."""
    test_cases = [
        [(0, 0, 0), (1, 1, 0), (1, 0, 1)],
        [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)],
        [(0, 0, 0), (1, 3, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 4, 0), (1, 1, 0), (0, 1, 1), (2, 0, 1), (1, 1, 1), (0, 0, 4)],
        [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 1), (1, 1, 2)]
    ]
    
    fig = plt.figure(figsize=(20, 10))
    
    for i, points in enumerate(test_cases):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        sphere = minimal_enclosing_sphere(points)
        plot_points_sphere_3d(points, sphere, ax)
        ax.set_title(f'Test {i+1}')
        
        for point in points:
            if sphere.contains(point):
                if sphere.onradius(point):
                    ax.scatter(point[0], point[1], point[2], color='g', label='On Sphere')
                else:
                    ax.scatter(point[0], point[1], point[2], color='b', label='Inside Sphere')
            else:
                ax.scatter(point[0], point[1], point[2], color='r', label='Outside Sphere')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    if any(not sphere.contains(point) for point in points):
        print("Warning: There are points outside the minimal enclosing sphere.")
    plt.show()


def test_circumsphere():
    #test circumsphere:
    test_cases = [
        [(0, 0, 0), (2, 0, 0), (1, 1, 0)]]
       
    fig, axs = plt.subplots(1, len(test_cases), figsize=(20, 4))

    for i, points in enumerate(test_cases):
            sphere = make_sphere_n_points(points)
            plot_points_sphere(points, sphere, axs[i])
            axs[i].set_title(f'Test {i+1}')
            for ax in axs:
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                ax.set_aspect('equal', 'box')
    plt.show()





def test_task4_plots():
    test_cases=[[(0,5,0),(3,4,0),(-3,4,0)],[(0,5,0),(3,4,0),(-3,4,0),(0,0,4)],[(0,5,0),(3,4,0),(0,0,4),(0,0,-4)]]

    fig, axs = plt.subplots(1, len(test_cases), figsize=(20, 4), subplot_kw={'projection': '3d'})

    for i, points in enumerate(test_cases):
        ax = axs[i]
        simplex, circum = task4(points, points)
        plot_points_sphere_3d(points, circum, ax)
        if simplex:
            ax.set_title(f'Test {i+1} - In alpha complex')
        else:
            ax.set_title(f'Test {i+1} - Not in alpha complex')

    # Rotate the plot for better visibility
    for ax in axs.flat:
        ax.view_init(elev=20., azim=180)

    plt.show()



#######################     Fonction pour plotter en pdf les complexes, créée les fichiers pour sc.py ############

import os
import sys
import numpy as np
import argparse
from scipy.spatial import distance_matrix
from itertools import combinations
from scipy.spatial import Voronoi, voronoi_plot_2d

#On se propose ici d'utiliser la bibliotheque sc.py pour ploter les complexes simpliciaux

#on enregistre d'abord le complexe simplicial dans un fichier texte:

def write_simplex_and_coords_files(simplex, filtration, points,i):
    os.makedirs("complex_plot", exist_ok=True)
    with open(f"complex_plot/coor_{i}.txt", "w") as f_coords, open(f"complex_plot/cplx_{i}.txt", "w") as f_simplex:
        for point in points:
            f_coords.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        for key in simplex:
            if simplex[key].radius < filtration:
                f_simplex.write(f"{' '.join(map(str, key))}\n")


def test_plot_complexes():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    j=1
    for i in [0, 4, 5, 100]:
        j+=1
        print(f"-------------------filtration={i}-------------------")
        write_simplex_and_coords_files(task3(P, i), i, P,j)
        #os.system("python3 sc.py --complex complex_plot/cplx.txt --coordinates complex_plot/coor.txt plot3d")
        #display(Image(filename="complex_plot/plot.png"))


def plot_cech_complex(simplex, points, radius):
    """Représente les simplexes et les points dans le plan."""

    #prendre une copie du dictionnaire simplex:
    simplex_copy = simplex.copy()

    points = np.array(points)
    
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], label='Points')
     # Draw circles around each point
    for point in points:
        circle = plt.Circle(point, radius, color='skyblue', alpha=0.3)
        ax.add_patch(circle)
        
    combin = None

    while True:
        if not simplex_copy:
            break
        
        combin, s = simplex_copy.popitem()
        if combin == None or len(combin) > 3:
            print("invalid combin value")
            break

        # Plot vertexes
        if len(combin) == 1:
            i = combin[0]
            ax.plot(points[i][0], points[i][1], 'ro')

        # Plot edges
        if len(combin) == 2:
            if len(combin) > 2:
                break
            i, j = combin
            ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'b-')

        # Plot triangles
        if len(combin) == 3:
            i, j, k = combin
            triang = np.array([points[i], points[j], points[k]])
            tri = plt.Polygon(triang, alpha=0.3, color='g')
            ax.add_patch(tri)

    ax.set_title('Cech Complex')
    ax.set_aspect('equal', 'box')
    plt.show()


def plot_alpha_complex(simplex, points,radius):
    """Représente les simplexes de l'alpha complexe dans le plan"""

    #prendre une copie du dictionnaire simplex:
    simplex_copy = simplex.copy()

    points = np.array(points)
    
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], label='Points')
     # Draw circles around each point
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2)
        
    combin = None

    while True:
        if not simplex_copy:
            break
        
        combin, s = simplex_copy.popitem()
        if combin == None or len(combin) > 3:
            print("invalid combin value")
            break

        # Plot vertexes
        if len(combin) == 1:
            i = combin[0]
            ax.plot(points[i][0], points[i][1], 'ro')

        # Plot edges
        if len(combin) == 2:
            if len(combin) > 2:
                break
            i, j = combin
            ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'b-')

        # Plot triangles
        if len(combin) == 3:
            i, j, k = combin
            triang = np.array([points[i], points[j], points[k]])
            tri = plt.Polygon(triang, alpha=0.3, color='g')
            ax.add_patch(tri)

    ax.set_title('Cech Complex')
    ax.set_aspect('equal', 'box')
    plt.show()


   

def test_plot_cech_complex_2d():
    points = [(np.random.randint(-10, 10), np.random.randint(-10, 10)) for _ in range(15)]

    #radius = 0.5
    l=3

    simplex = task3(points, l, printit=True)
    plot_cech_complex(simplex, points,l)

def test_plot_alpha_complex_2d():
    points = [(np.random.randint(-10, 10), np.random.randint(-10, 10)) for _ in range(15)]

    #radius = 0.5
    l=3

    simplex = task3(points, l, printit=True)
    plot_alpha_complex(simplex, points,l)

def compare_cech_alpha(points,l,printit=False):
    cech = task3(points, l, printit)
    alpha = task5(points, l, printit)
    plot_cech_complex(cech, points,l)
    plot_alpha_complex(alpha, points,l)






######################################## RUN THE PLOTS ########################################




if __name__ == "__main__":
    #test_and_plot_minimal_enclosing_sphere_2D()
    #test_and_plot_minimal_enclosing_sphere_3d()
    #test_circumsphere()
    #test_task4_plots()
    #test_plot_complexes()
    #test_plot_cech_complex_2d()
    test_plot_alpha_complex_2d()
