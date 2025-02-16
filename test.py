from Tasks import *
from plots import *
import numpy as np
import random
import time
import pandas as pd
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
        
def test_task3_simple():
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3)]
    task3(P,1000)

def test_task3():
    #generate random n points in R^d:
    #n=random.randint(5,10)
    n=100
    print(f"n={n}")
    d=random.randint(2,5)
    print(f"d={d}")
    points=[tuple(np.random.rand(d)) for i in range(n)]
    print(f"Points: {points}")
    k=random.randint(2,d)
    print(f"k={k}")
    l=np.random.rand(1)
    print(f"l={l}")
    start_time = time.time()
    task3(points,l,printit=True)
    print("--- %s seconds ---" % (time.time() - start_time))

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
    P = [(5, 0, 1), (-1, -3, 4), (-1, -4, -3), (-1, 4, -3), (8,8,9),(1,4,1),(5,10,-10)]
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


def test_task5_random(in2D = False):
    #generate random n points in R^d:
    n=random.randint(5,10)
    d=random.randint(2,5)

    k=random.randint(2,d)
    if in2D:
        d=2
        k=2

    print(f"n={n}")
    print(f"d={d}")
    print(f"k={k}")


    points=[tuple(np.random.rand(d)) for i in range(n)]
    print(f"Points: {points}")
    l=np.random.rand(1)
    print(f"l={l}")
    simplex,_,_ = task5(points,k,l)
    if in2D:
        plot_alpha_complex(simplex,points,l)






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


def test_compare_efficiency():
    # Generate random points and filtration value

    results = []
    l=1/3

    for n in range(5, 20, 5):  # Increasing sample size n
        for d in range(2, 5):  # Increasing dimension d
            points = [tuple(np.random.rand(d)) for _ in range(n)]

            # Measure time for task3
            start_time = time.time()
            task3(points, l)
            task3_time = time.time() - start_time

            # Measure time for task5
            start_time = time.time()
            task5(points, d, l)
            task5_time = time.time() - start_time

            results.append({
                'Sample Size (n)': n,
                'Dimension (d)': d,
                'Function': 'task3',
                'Time (seconds)': task3_time
            })
            results.append({
                'Sample Size (n)': n,
                'Dimension (d)': d,
                'Function': 'task5',
                'Time (seconds)': task5_time
            })

    # Print results in a table
    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))

    # Plot the evolution of computing time for task3 and task5

    plt.figure(figsize=(14, 8))
    for function in results_df['Function'].unique():
        for dimension in results_df['Dimension (d)'].unique():
            subset = results_df[(results_df['Function'] == function) & (results_df['Dimension (d)'] == dimension)]
            plt.plot(subset['Sample Size (n)'], subset['Time (seconds)'], marker='o', label=f'{function} (d={dimension})')
    plt.title('Evolution of Computing Time for task3 and task5')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Time (seconds)')
    plt.legend(title='Function and Dimension')
    plt.grid(True)
    plt.show()


def test_compare_plots():
    cech = task3(points, l)
    alpha = task5(points, l)
    plot_cech_complex(cech, points,l)
    plot_alpha_complex(alpha, points,l)






if __name__ == "__main__":

    # print("---------Question 1------------")
    # test_task1()
    # print("---------Question 2------------")
    # test_task2()
    # print("---------Question 3------------")
    # test_task3()
    # test_task3_simple()

    # print("---------Question 4------------")
    # test_task4()
    # print("---------Question 5------------")
    # test_task5_random(in2D = True)



    # print("---------Robustness checks------------")
    # test_4d_task2()
    # test_4D_MEB()

    print("---------Efficiency comparison------------")
    test_compare_efficiency()

