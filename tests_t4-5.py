from Tasks import *
import matplotlib.pyplot as plt
from plots import *



def test_circumcenter_alpha():
    test_cases =[[(1,0),(3,0)],[(1,0),(1.5,2.3)],[(3,0),(1.5,2.3)],[(1,0),(3,0),(1.5,2.3)]]
    
    

    fig, ax = plt.subplots(1, 1, figsize=(20, 4))

    for i, points in enumerate(test_cases):
            sphere = make_sphere_n_points(points)
            if i==3:
                plot_points_sphere(points, sphere, ax,color_black=True)
                break
            plot_points_sphere(points, sphere, ax)
            ax.set_aspect('equal', 'box')
    plt.show()

test_circumcenter_alpha()