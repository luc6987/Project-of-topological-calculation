#!/usr/bin/env python3

import sys
import numpy as np
import argparse

def compute_diagram(action, file_complex):
    import gudhi
    st=gudhi.SimplexTree()
    for line in file_complex:
        fields = line.split()
        st.insert([int(i) for i in fields])
    st.compute_persistence()
    for i, n in enumerate(st.betti_numbers()):
        if n != 0:
            print("dimension", i, ":", n)

def plot3d(action, file_complex, file_coordinates):
    points = np.loadtxt(file_coordinates)
    simplexes = [[], [], []]
    for line in file_complex:
        fields = line.split()
        k = len(fields) - 1
        if k <= 2:
            simplexes[k].append([int(i) for i in fields])

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure()
    #ax = fig.gca(projection='3d') # old matplotlib
    ax = fig.add_subplot(projection='3d')
    # Plot triangles
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=simplexes[2])
    # Plot points
    points2 = points[np.array(simplexes[0]).reshape(-1)]
    ax.scatter3D(points2[:,0], points2[:,1], points2[:,2])
    # Plot edges
    ax.add_collection3d(Line3DCollection([points[e] for e in simplexes[1]]))
    plt.savefig('complex_plot/myplot.pdf')
    plt.show()



parser = argparse.ArgumentParser(description='Process a simplicial complex.')
parser.add_argument('--complex', type=argparse.FileType('r'))
parser.add_argument('--coordinates', type=argparse.FileType('r'))
parser.add_argument('action', choices=['betti','plot3d'])
args = parser.parse_args()
if args.action in ['betti']:
    if not args.complex:
        print("missing complex")
        sys.exit()
    compute_diagram(args.action, args.complex)
elif args.action in ['plot3d']:
    if not args.complex:
        print("missing complex")
        sys.exit()
    if not args.coordinates:
        print("missing coordinates")
        sys.exit()
    plot3d(args.action, args.complex, args.coordinates)