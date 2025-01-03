import  Tasks as t
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# Test cases
def test_task1():
    """Test cases for minimal enclosing sphere."""
    # Test 1: Single point
    points = [(0, 0, 0)]
    sphere = t.minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 0)
    print("Test 1 passed!")

    # Test 2: Two points
    points = [(0, 0, 0), (2, 0, 0)]
    sphere = t.minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [1, 0, 0])
    assert np.isclose(sphere.radius, 1)
    print("Test 2 passed!")

    # Test 3: Three points
    points = [(-10, 0, 0), (10, 0, 0), (0, 1, 0)]
    sphere = t.minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, 10)
    print("Test 3 passed!")

    # Test 4: Four points
    points = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    sphere = t.minimal_enclosing_sphere(points)
    assert np.allclose(sphere.center, [0, 0, 0])
    assert np.isclose(sphere.radius, np.sqrt(26))
    print("Test 4 passed!")

    print("All test cases passed!")


# Test cases
def test_task2():
        P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]

        enu=[0]
        assert np.allclose(t.task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[1]
        assert np.allclose(t.task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[2]
        assert np.allclose(t.task2(P,enu), 0)  
        print(f"Test({enu})passed!")

        enu=[3]
        assert np.allclose(t.task2(P,enu), 0)
        print(f"Test({enu})passed!")

        enu=[2,1]
        assert np.allclose(t.task2(P,enu), 3.53553)   
        print(f"Test({enu})passed!")

        enu=[1,0]
        assert np.allclose(t.task2(P,enu), 3.67425)   
        print(f"Test({enu})passed!")

        enu=[3,2]
        assert np.allclose(t.task2(P,enu), 4)   
        print(f"Test({enu})passed!")

        enu=[2,0]
        assert np.allclose(t.task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[3,0]
        assert np.allclose(t.task2(P,enu), 4.12311)   
        print(f"Test({enu})passed!")

        enu=[2,1,0]
        assert np.allclose(t.task2(P,enu), 4.39525)   
        print(f"Test({enu})passed!")

        enu=[3,2,0]
        assert np.allclose(t.task2(P,enu), 4.71495)   
        print(f"Test({enu})passed!")

        enu=[3,1]
        assert np.allclose(t.task2(P,enu), 4.94975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1]
        assert np.allclose(t.task2(P,enu), 5)   
        print(f"Test({enu})passed!")

        enu=[3,1,0]
        assert np.allclose(t.task2(P,enu), 5.04975)   
        print(f"Test({enu})passed!")

        enu=[3,2,1,0]
        assert np.allclose(t.task2(P,enu), 5.09902)   
        print(f"Test({enu})passed!")

        print("Test 2 all passed! ")
def test_task3():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    L=[0,3,4,5,10]
    i=1
    for l in L:
        print(f"---- Test {i} pour l={l}--------(luc)")
        i+=1
        t.task3(P,l)
def test_task3_mathias():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    L=[0,1,3,5,10]
    i=1
    for l in L:
        print(f"---- Test {i} pour l={l}--------(mathias)")
        i+=1
        t.task3_mathias(P,l)


def test_task4():
    P=[(0,-5,0),(3,4,0),(-3,4,0)]
    P1=P
    print(f"---- Test for {P1}")
    a=t.task4(P1,P1)
    print(f"Complex ? {a[0]} ; Radius: {a[1]}")
    P1.append((0,0,4))
    print(f"---- Test for {P1}")
    a=t.task4(P1,P1)
    print(f"Complex ? {a[0]} ; Radius: {a[1]}")
    P1.append((0,0,-4))
    print(f"---- Test for {P1}")
    a=t.task4(P1,P1)
    print(f"Complex ? {a[0]} ; Radius: {a[1]}")


def plot_circle(P,cMEB=True):
    if cMEB:
        MEB= t.minimal_enclosing_sphere(P)
    else: 
        MEB=t.alpha_sphere(P)
    
    Pbis=[p[0:2] for p in P]
    center = MEB.center[0:2]
    rayon = MEB.radius
    plt.figure(figsize=(6, 6))

    for p in Pbis:
        plt.scatter(p[0],p[1])
        plt.scatter(center[0],center[1])

    theta = np.linspace(0, 2 * np.pi, 100)  # 100 points entre 0 et 2π
    x = center[0] + rayon * np.cos(theta)    # Coordonnées x
    y = center[1] + rayon * np.sin(theta)    # Coordonnées y

    # Tracer le cercle
    plt.plot(x, y, label="Cercle")
    plt.scatter(center[0], center[1], color='red', label="Centre du cercle")  # Point central
    plt.gca().set_aspect('equal', adjustable='box')  # Maintenir le cercle rond
    plt.axhline(0, color='gray', linestyle='--')  # Axe x
    plt.axvline(0, color='gray', linestyle='--')  # Axe y  
    if cMEB:
        plt.title("MEB")
    else:
        plt.title("alpha circle")
    plt.show()

    return None

def plot_sphere(P,cMEB=True):
    if cMEB:
        MEB= t.minimal_enclosing_sphere(P)
    else: 
        #MEB=t.alpha_sphere(P)
        #MEB= t.make_sphere_four_points(P[0],P[1],P[2],P[3])
        MEB = t.circumcenter_mathias(P)
    
    center = MEB.center
    rayon = MEB.radius

    x_center, y_center, z_center = center  # Center of the ball  # Radius of the ball

    # Generate a grid for the sphere
    u = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
    v = np.linspace(0, np.pi, 100)      # Polar angle
    x = x_center + rayon * np.outer(np.cos(u), np.sin(v))
    y = y_center + rayon * np.outer(np.sin(u), np.sin(v))
    z = z_center + rayon * np.outer(np.ones_like(u), np.cos(v))

    # Plot the sphere
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='cyan', alpha=0.2, edgecolor='gray')

    # Add the center point
    ax.scatter(x_center, y_center, z_center, color="red", label="Center", s=100)

    #plot the points
    for p in P:
            test = MEB.onbound(p)
            if test:
                 ax.scatter(p[0],p[1],p[2], color="black", label="Point on bound", s=100)
            else:
                ax.scatter(p[0],p[1],p[2], color="yellow", label="outside point", s=100)
    #Add other points

    # Set labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    plt.legend()
    
    if cMEB:
        plt.title(f"MEB, cercle: {MEB.center}, {MEB.radius}")
    else:
        plt.title(f"alpha circle, cercle: {MEB.center}, {MEB.radius}")
    plt.show()

    return None

def test_sphere4points():
    #ce test montre que la sphère pour 4 points ne convient pas
    P=[(2,0,0),(0,1,0),(1/np.sqrt(2),1/np.sqrt(2),0),(1,1,2)]
    plot_sphere(P, False)




def main_test_all():
    print("---------Question 1------------")
    test_task1()
    print("---------Question 2------------")
    test_task2()
    print("---------Question 3------------")
    test_task3()
    print("fonction mathias:")
    test_task3_mathias()
    print("---------Question 4------------")
    test_task4()


test_sphere4points()