from Tasks import *
from plots import *
import numpy as np
import random
import matplotlib.pyplot as plt


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

    test_cases = [
        ([0], 0),
        ([1], 0),
        ([2], 0),
        ([3], 0),
        ([2, 1], 3.53553),
        ([1, 0], 3.67425),
        ([3, 2], 4),
        ([2, 0], 4.12311),
        ([3, 0], 4.12311),
        ([2, 1, 0], 4.39525),
        ([3, 2, 0], 4.71495),
        ([3, 1], 4.94975),
        ([3, 2, 1], 5),
        ([3, 1, 0], 5.04975),
        ([3, 2, 1, 0], 5.09902),
    ]

    for enu, expected in test_cases:
        assert np.allclose(task2(P, enu), expected), f"Test({enu}) failed!"
        print(f"Test({enu}) passed!")



        
def test_task3():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    for i in [0,4,5,100]:
        print(f"-------------------filtration={i}-------------------")
        task3(P, i, printit = True)


def test_task4():
    P=[(0,5,0),(3,4,0),(-3,4,0)]
    R=[(0,5,0),(3,4,0),(-3,4,0)]
    print(f"---- Test for {P}")
    a= task4(P,R)
    if a[0]:
        print(f"{R} is in the  $\alpha$-complex")
        print(f"filtration value: {a[1].radius}")
        print(f"center {a[1].center}")
    else:
        print(f"{R} is not in the  $\alpha$-complex")

    P.append((0,0,4))
    R.append((0,0,4))
    print(f"---- Test for {P}")
    a= task4(P,R)
    if a[0]:
        print(f"{R} is in the alpha-complex")
        print(f"filtration value: {a[1].radius}")
        print(f"center {a[1].center}")
    else:
        print(f"{R} is not in the  alpha-complex")


    P.append((0,0,-4))
    R.append((0,0,-4))
    print(f"---- Test for {P}")
    a= task4(P,R)
    if a[0]:
        print(f"{R} is in the  alpha-complex")
        print(f"filtration value: {a[1].radius}")
        print(f"center {a[1].center}")
    else:
        print(f"{R} is not in the  alpha-complex")


def test_task5():
    #generate random n points in R^d:
    n=random.randint(5,10)
    print(f"n={n}")
    d=random.randint(2,5)
    print(f"d={d}")
    points=[tuple(np.random.rand(d)) for i in range(n)]
    print(f"Points: {points}")
    k=random.randint(2,d)
    print(f"k={k}")
    l=np.random.rand(1)
    print(f"l={l}")
    task5(points,k,l)


def test_4d_task2():
    P = [(5, 0, 1, 0), (-1, -3, 4, 0), (-1, -4, -3, 0), (-1, 4, -3, 0)]

    test_cases = [
        ([0], 0),
        ([1], 0),
        ([2], 0),
        ([3], 0),
        ([2, 1], 3.53553),
        ([1, 0], 3.67425),
        ([3, 2], 4),
        ([2, 0], 4.12311),
        ([3, 0], 4.12311),
        ([2, 1, 0], 4.39525),
        ([3, 2, 0], 4.71495),
        ([3, 1], 4.94975),
        ([3, 2, 1], 5),
        ([3, 1, 0], 5.04975),
        ([3, 2, 1, 0], 5.09902),
    ]

    for enu, expected in test_cases:
        assert np.allclose(task2(P, enu), expected), f"Test({enu}) failed!"
        print(f"Test({enu}) passed!")

def test_4D_circum():
    P = [(5, 0, 1, 0), (-1, -3, 4, 0), (-1, -4, -3, 0), (-1, 4, -3, 0),(0,0,0,19)]
    sphere = make_sphere_n_points(P)
    print(f"Center: {sphere.center}")
    print(f"Radius: {sphere.radius}")
    for point in P:
        if sphere.contains(point):
            if sphere.onradius(point):
                print(f"Point {point} is on the sphere")
            else:
                print(f"Point {point} is inside the sphere")
        else:
            print(f"Point {point} is outside the sphere")

def test_4D_MEB():
    P = [(5, 0, 1, 0), (-1, -3, 4, 0), (-1, -4, -3, 0), (-1, 4, -3, 0), (0, 0, 0, 19),
         (2, 2, 2, 2), (-2, -2, -2, -2), (3, 3, 3, 3), (-3, -3, -3, -3), (4, 4, 4, 4),
         (-4, -4, -4, -4), (1, 1, 1, 1), (-1, -1, -1, -1), (6, 6, 6, 6), (-6, -6, -6, -6),
         (7, 7, 7, 7), (-7, -7, -7, -7), (8, 8, 8, 8), (-8, -8, -8, -8), (9, 9, 9, 9),
         (-9, -9, -9, -9), (10, 10, 10, 10), (-10, -10, -10, -10)]
    sphere = minimal_enclosing_sphere(P)
    print(f"Center: {sphere.center}")
    print(f"Radius: {sphere.radius}")
    for point in P:
        if sphere.contains(point):
            if sphere.onradius(point):
                print(f"Point {point} is on the sphere")
            else:
                print(f"Point {point} is inside the sphere")
        else:
            print(f"Point {point} is outside the sphere")






if __name__ == "__main__":

    print("---------Question 1------------")
    test_task1()
    print("---------Question 2------------")
    test_task2()
    print("---------Question 3------------")
    test_task3()

    print("---------Question 4------------")
    test_task4()
    print("---------Question 5------------")
    #test_task5()


    print("---------Robustness checks------------")
    #test_4d_task2()
    test_4D_MEB()

