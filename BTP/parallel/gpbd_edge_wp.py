import os
import numpy as np
from typing import Any
from tqdm import tqdm, trange
import scipy
import trimesh
import time
from itertools import cycle
import warp as wp

@wp.func
def LDLT(A: wp.array(dtype = wp.float32, ndim = 2), L: wp.array(dtype = wp.float32, ndim = 2)): 
    for i in range(A.shape[0]): 
        for j in range(0, i+1): 
            sum = float(0.0)
            sum1 = float(0.0)
            for k in range(0, j): 
                sum += L[j, k] * L[j, k] *  L[k, k]
                sum1 += L[i, k] * L[j, k] * L[k, k]
            
            if (i == j): 
                L[i, j] = A[i, j] - sum
            else: 
                L[i, j] = 1.0 / L[j, j] * (A[i, j] - sum1)

@wp.func
def LLT(A: wp.array(dtype = wp.float32, ndim = 2), L: wp.array(dtype = wp.float32, ndim = 2)): 
    for i in range(A.shape[0]): 
        for j in range(0, i+1): 
            sum = float(0.0)
            for k in range(0, j): 
                sum += L[i, k] * L[j, k]

            if (i == j): 
                value = A[i, i] - sum
                L[i, j] = wp.sqrt(value)
            else: 
                L[i, j] = 1.0 / L[j, j] * (A[i, j] - sum)

@wp.func
def forward_substitution(L: wp.array(dtype = wp.float32, ndim = 2), b: wp.array(dtype = wp.float32), y: wp.array(dtype = wp.float32)): 
    for i in range(L.shape[0]): 
        sum = float(0.0)
        for j in range(0, i): 
            sum += y[j] * L[i, j]
        y[i] = (b[i] - sum)/L[i, i] 

@wp.func
def back_substitution(U: wp.array(dtype = wp.float32, ndim = 2), y: wp.array(dtype = wp.float32), x: wp.array(dtype = wp.float32)): 
    n = U.shape[0]
    for i in range(n-1, -1, -1): 
        sum = float(0.0)
        for j in range(i+1, n): 
            sum += x[j] * U[j, i]
        x[i] = (y[i] - sum) / U[i, i]

@wp.func
def LLTSolve(A: wp.array(dtype = wp.float32, ndim = 2), 
              b: wp.array(dtype = wp.float32), 
              L: wp.array(dtype = wp.float32, ndim = 2), 
              y: wp.array(dtype = wp.float32), 
              x: wp.array(dtype = wp.float32)): 

    LLT(A, L)
    forward_substitution(L, b, y)
    back_substitution(L, y, x)


def generate_triangulated_square_mesh(bottom_left, side_length, tessellation_level):
    vertices = []
    faces = []
    edges = set()

    x0, y0 = bottom_left
    step = side_length / tessellation_level

    # Generate vertices
    for i in range(tessellation_level + 1):
        for j in range(tessellation_level + 1):
            vertices.append((x0 + i * step, 1, y0 + j * step))

    # Generate faces (triangles)
    for i in range(tessellation_level):
        for j in range(tessellation_level):
            v0 = i * (tessellation_level + 1) + j
            v1 = v0 + 1
            v2 = v0 + (tessellation_level + 1)
            v3 = v2 + 1

            # Two triangles for each square
            f0 = (v0, v1, v2) 
            f1 = (v1, v3, v2)
            faces.append(f0)
            faces.append(f1)



            for k,l in zip([v0, v1, v2], [v1, v2, v0]): 
                edges.add(tuple(sorted((k, l))))

            for k,l in zip([v1, v3, v2], [v3, v2, v1]): 
                edges.add(tuple(sorted((k, l))))

    return np.array(vertices), np.array(faces), np.array(list(edges))

def compute_vertex_masses(rest_positions, triangles, density):
    """
    Compute the mass of each vertex given the density.
    """
    num_vertices = rest_positions.shape[0]
    masses = np.zeros(num_vertices)

    for tri in triangles:
        X_rest = rest_positions[tri]

        # Compute the volume of the triangle
        area = compute_triangle_area(X_rest)

        # Compute the mass of the triangle
        mass_tri = density * area
        
        # Distribute the mass equally to the four vertices
        mass_contribution = mass_tri / 3.0
        for vertex in tri:
            masses[vertex] += mass_contribution
    
    return masses

def compute_triangle_area(vertices):
    """
    Compute the area of a triangle given its vertices.
    """
    if vertices.shape != (3, 3):
        raise ValueError("The input must be a 3x3 array representing the coordinates of the triangle vertices.")

    # Using the shoelace formula to compute the area of the triangle
    vec1 = vertices[0] - vertices[1]
    vec2 = vertices[0] - vertices[2]
    area = 0.5 * np.linalg.norm(np.cross(vec1, vec2))

    return area


def compute_B(x_): 
    H = np.array([
        [-1, -1],
        [1, 0],
        [0, 1]
    ])

    X_ = np.stack([x_[..., 0], x_[..., 2]])
    XH = X_ @ H
    B = np.matmul(H, np.linalg.inv(XH))
    return B

def visualize_forces(x_inertial, faces, f0): 
    ps_mesh = ps.register_surface_mesh("mesh", x_inertial, faces)
    ps_mesh.add_vector_quantity("force", f0, defined_on = 'vertices', enabled = True)

def simulate(h, rest_vertices, faces, edges, masses, constraint_vertices, total_frames, stiffness, gpbd_iters, exp_dir): 
    x = np.copy(rest_vertices)
    v = np.zeros((rest_vertices.shape[0], 3))

    all_x = []

    for i in trange(total_frames): 
        x_prev = np.copy(x)
        v_prev = np.copy(v)

        # x_inertial = x_prev + h * v_prev
        
        ## Gravity force
        gravity_vector = - np.array([0, 9.8, 0])
        a = np.tile(gravity_vector[None], (x_prev.shape[0], 1))

        x = x_prev + h * v_prev + h*h*a

        x = gpbd_step(h, x, masses, rest_vertices, edges, constraint_vertices, gpbd_iters, stiffness)

        v = (x - x_prev) / h
            
        output_path = os.path.join(exp_dir, f'{i}.obj')
        dump_mesh(x, faces, output_path)

def dump_mesh(vertices, faces, filepath): 
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(filepath)

def load_mesh(filepath): 
    mesh = trimesh.load(filepath, process = False, maintain_order = True)
    return mesh.vertices, mesh.faces, mesh.edges

@wp.func
def norm(vec: wp.array(dtype = wp.float32)): 
    sum = float(0.0)
    for i in range(vec.shape[0]): 
        sum += vec[i] * vec[i]
    return wp.sqrt(sum)

@wp.kernel
def solve_iteration(edges: wp.array(dtype=wp.vec2i),
                    Wfjs: wp.array(dtype=wp.float32, ndim = 3), 
                    positions: wp.array(dtype=wp.vec3),
                    x_temp: wp.array(dtype=wp.vec3), 
                    temp_count: wp.array(dtype=wp.int32), 
                    masses: wp.array(dtype=float),
                    rest_vertices: wp.array(dtype=wp.vec3),
                    h: float,
                    stiffness: float, 
                    jacobians: wp.array(dtype=wp.float32, ndim = 2), 
                    hessians: wp.array(dtype = wp.float32, ndim = 3), 
                    L: wp.array(dtype = wp.float32, ndim =3), 
                    y: wp.array(dtype = wp.float32, ndim = 2), 
                    x: wp.array(dtype = wp.float32, ndim = 2)
                    ): 

    t = wp.tid()

    # Wfj = Wfjs[t]  # 2x3
    e = edges[t]
        
    ## Calculate x_j
    Wfj0 = wp.vec3()
    Wfj1 = wp.vec3()

    for i in range(3): 
        Wfj0[i] = Wfjs[t][0][i]
        Wfj1[i] = Wfjs[t][1][i]

    x_j0 = positions[e[0]] - Wfj0
    x_j1 = positions[e[1]] - Wfj1

    x_rest0 = rest_vertices[e[0]]
    x_rest1 = rest_vertices[e[1]]

    m0 = masses[e[0]]
    m1 = masses[e[1]]

    max_newton_iters = 1
    eps = 1e-8

    for cur_newton_iter in range(max_newton_iters): 
    # while (True): 
        x0 = wp.vec3()
        x1 = wp.vec3()

        x0 = x_j0 + Wfj0
        x1 = x_j1 + Wfj1

        # calculate_jacobians(x0, x1, 
        #                     x_rest0, x_rest1, 
        #                     Wfjs[wt], Wfjs[wt+1], 
        #                     m0, m1, 
        #                     stiffness, 
        #                     h, 
        #                     jacobians[t])
        
        f0, f1 = spring_force(x_rest0, x_rest1, x0, x1, stiffness)

        for i in range(3): 
            jacobians[t][i] = - (m0 * Wfj0[i] / h / h - f0[i])
        for j in range(3): 
            jacobians[t][j+3] = - (m1 * Wfj1[j] / h / h - f1[j])
            

        calculate_hessians(x0, x1, 
                           x_rest0, x_rest1, 
                           Wfj0, Wfj1, 
                           m0, m1, 
                           stiffness, 
                           h, 
                           hessians[t])
        
        LLTSolve(hessians[t], jacobians[t], L[t], y[t], x[t])
        
        if (norm(x[t]) < eps): 
            break

        Wfj0 += wp.vec3(x[t][0], x[t][1], x[t][2])
        Wfj1 += wp.vec3(x[t][3], x[t][4], x[t][5])
    
    wp.atomic_add(x_temp, e[0], Wfj0 + x_j0)
    wp.atomic_add(x_temp, e[1], Wfj1 + x_j1)
    wp.atomic_add(temp_count, e[0], 1)
    wp.atomic_add(temp_count, e[1], 1)
    
    for i in range(3): 
        Wfjs[t][0][i] = Wfj0[i]
        Wfjs[t][1][i] = Wfj1[i]


    
@wp.func
def spring_force(x0_rest: wp.vec3, x1_rest: wp.vec3, x0_cur: wp.vec3, x1_cur: wp.vec3, stiffness: float): 
    L0 = wp.length(x1_rest - x0_rest)
    L1 = wp.length(x1_cur - x0_cur)
    x21 = wp.vec3()
    x21 = x1_cur - x0_cur
    force_magnitude = stiffness * (L1 - L0) / L1
    f0 = force_magnitude * x21 
    f1 = -f0
    return f0, -f0

@wp.func
def calculate_jacobians(x0: wp.vec3, x1: wp.vec3, 
                        x0_rest: wp.vec3, x1_rest: wp.vec3, 
                        Wf0: wp.vec3, Wf1: wp.vec3, 
                        m0: float, m1: float, 
                        stiffness: float, 
                        h: float, 
                        J: wp.array(dtype = wp.float32)): 
    ## Calculate Spring Jacobian
    f0 = wp.vec3()
    f1 = wp.vec3()

    f0, f1 = spring_force(x0_rest, x1_rest, x0, x1, stiffness)

    for i in range(3): 
        J[i] = - (m0 * Wf0[i] / h / h - f0[i])
    for j in range(3): 
        J[j+3] = - (m1 * Wf1[j] / h / h - f1[j])


@wp.func
def spring_force_derivatives(x0_rest: wp.vec3, x1_rest: wp.vec3, x0_cur: wp.vec3, x1_cur: wp.vec3, stiffness: float, H: wp.array(dtype = wp.float32, ndim = 2)): 
    L0 = wp.length(x1_rest - x0_rest)
    L1 = wp.length(x1_cur - x0_cur)
    x21 = wp.vec3()
    x21 = x1_cur - x0_cur
    
    x21x21Dyadic = wp.mat33()
    x21x21Dyadic = wp.outer(x21 / L1, x21 / L1)
    
    mat = wp.mat33()
    mat = - stiffness * ((1.0 - L0 / L1) * (wp.identity(3, dtype = wp.float32) - x21x21Dyadic) + x21x21Dyadic)

    for i in range(3): 
        for j in range(3): 
            H[i, j] = mat[i, j]

    for i in range(3): 
        for j in range(3): 
            H[i+3, j+3] = mat[i, j]

    for i in range(3): 
        for j in range(3): 
            H[i, j+3] = -mat[i, j]

    for i in range(3): 
        for j in range(3): 
            H[i+3, j] = -mat[i, j]

@wp.func
def calculate_hessians(x0: wp.vec3, x1: wp.vec3, 
                       x0_rest: wp.vec3, x1_rest: wp.vec3, 
                       Wf0: wp.vec3, Wf1: wp.vec3, 
                       m0: float, m1: float, 
                       stiffness: float, 
                       h: float, 
                       H: wp.array(dtype = wp.float32, ndim = 2)): 

    spring_force_derivatives(x0_rest, x1_rest, x0, x1, stiffness, H)
    
    for k in range(3): 
        H[k,k] = m0 / h / h - H[k, k]
    for k in range(3): 
        H[k+3,k+3] = m1 / h / h - H[k+3, k+3]


def gpbd_step(h, x_inertial, v_prev, masses, rest_vertices, edges, constraint_vertices, gpbd_iters, stiffness): 

    positions = x_inertial 

    Wfjs = np.zeros((edges.shape[0], 2, 3))

    ## Loop through all the constraints
    for k in range(gpbd_iters): 
        
        edges_gpu = wp.array(edges, dtype=wp.vec2i)
        Wfjs_gpu = wp.array(Wfjs, dtype=wp.float32)
        positions_gpu = wp.array(positions, dtype=wp.vec3)
        # x_temp_gpu = wp.zeros_like(Wfjs_gpu)
        x_temp_gpu = wp.zeros_like(positions_gpu)
        temp_count = wp.zeros((positions.shape[0]), dtype = wp.int32)
        masses_gpu = wp.array(masses, dtype=float)
        rest_vertices_gpu = wp.array(rest_vertices, dtype=wp.vec3)
        jacobians = wp.zeros((edges.shape[0], 6), dtype = wp.float32)
        hessians = wp.zeros((edges.shape[0], 6, 6), dtype = wp.float32)
        L = wp.zeros((edges.shape[0], 6, 6), dtype = wp.float32)
        y = wp.zeros((edges.shape[0], 6), dtype = wp.float32)
        x = wp.zeros((edges.shape[0], 6), dtype = wp.float32)

        wp.launch(kernel=solve_iteration, 
                dim=edges.shape[0], 
                inputs=[edges_gpu, 
                        Wfjs_gpu, 
                        positions_gpu, 
                        x_temp_gpu, 
                        temp_count, 
                        masses_gpu, 
                        rest_vertices_gpu, 
                        h, 
                        stiffness, 
                        jacobians, 
                        hessians, 
                        L, 
                        y, 
                        x
                        ])
        
        positions = x_temp_gpu.numpy()
        temp_count = temp_count.numpy()[..., None]
        positions = positions / temp_count

        Wfjs = Wfjs_gpu.numpy()
        # A = hessians.numpy()[-1]
        # b = jacobians.numpy()[-1]

        for cv in constraint_vertices:
            positions[cv] = rest_vertices[cv]

        x_new = positions
    return x_new

def main(): 
    side_length = 500
    bottom_left = (0, 0)
    tessellation_level = 256
    timestep = 0.005
    density = 10
    total_frames = 1000
    stiffness = 8000
    gpbd_iters = 1
    output_dir = './outputs/gpbd_exps/'
    exp_name = 'verts256'
    
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok = True)

    rest_vertices, faces, edges = generate_triangulated_square_mesh(bottom_left, side_length, tessellation_level)
    masses = compute_vertex_masses(rest_vertices, faces, density)
    constraint_vertices = [0, tessellation_level]
    # constraint_vertices = [0]
    simulate(timestep, rest_vertices, faces, edges, masses, constraint_vertices, total_frames, stiffness, gpbd_iters, exp_dir)
    

if __name__ == '__main__': 
    main()
