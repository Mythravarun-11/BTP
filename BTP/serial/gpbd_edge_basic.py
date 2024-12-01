import os
import numpy as np
import rtree
from typing import Any
from copy import deepcopy
from tqdm import tqdm, trange
import scipy
import trimesh
import time
from itertools import cycle

def group(x):
    return x.reshape(-1, 3)

def dump_mesh(vertices, faces, filepath): 
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(filepath)


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


class SpringModel: 
    def __init__(self, k): 
        self.k = k

    def get_energy(self, x_rest, x_cur): 
        rest_length =  np.linalg.norm(x_rest[1] - x_rest[0])
        cur_length = np.linalg.norm(x_cur[1] - x_cur[0])
        return 0.5 * self.k * (cur_length - rest_length)**2

    def get_force(self, x_rest, x_cur): 
        rest_length =  np.linalg.norm(x_rest[1] - x_rest[0])
        cur_length = np.linalg.norm(x_cur[1] - x_cur[0])
        x21 = x_cur[1] - x_cur[0]
        force_magnitude = self.k * (cur_length - rest_length) 
        force = force_magnitude * (x21 / cur_length)
        force1 = force
        force2 = -force
        forces = np.array([force1, force2])
        return forces

    def get_force_jacobian(self, x_rest, x_cur):
        rest_length =  np.linalg.norm(x_rest[1] - x_rest[0])
        cur_length = np.linalg.norm(x_cur[1] - x_cur[0])
        x21 = x_cur[1] - x_cur[0]
        x21x21Dyadic = np.outer(x21 / cur_length, x21 / cur_length)
        assert x21x21Dyadic.shape == (3, 3)
        mat = -self.k * ((1 - rest_length / cur_length) * (np.eye(3, 3) - x21x21Dyadic) + x21x21Dyadic)
        neg_mat = -mat
        top_row = np.hstack((mat, neg_mat))
        bottom_row = np.hstack((neg_mat, mat))
        jacobian = np.vstack((top_row, bottom_row))
        return jacobian


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


def generate_triangulated_square_mesh(bottom_left, side_length, tessellation_level, MESH_ALIGNMENT, scale = 1): 
    vertices = []
    faces = []
    edges = set()

    x0, y0 = bottom_left
    step = side_length / tessellation_level

    # Generate vertices
    for i in range(tessellation_level + 1):
        for j in range(tessellation_level + 1):
            if MESH_ALIGNMENT == 'XZ': 
                vertices.append((x0 + i * step,  0, y0 + j * step))
            elif MESH_ALIGNMENT == 'XY': 
                vertices.append((x0 + i * step,  y0 + j * step, 0))


    # Generate faces (triangles)
    for i in range(tessellation_level):
        for j in range(tessellation_level):

            v0 = i * (tessellation_level + 1) + j
            v1 = v0 + 1
            v2 = v0 + (tessellation_level + 1)
            v3 = v2 + 1

            # Two triangles for each square
            # faces.append((v0, v1, v2))
            # faces.append((v1, v3, v2))

            faces.append((v0, v2, v1))
            faces.append((v1, v2, v3))

            for k,l in zip([v0, v1, v2], [v1, v2, v0]): 
                edges.add(tuple(sorted((k, l))))

            for k,l in zip([v1, v3, v2], [v3, v2, v1]): 
                edges.add(tuple(sorted((k, l))))
    
    vertices = np.array(vertices)
    if scale != 1: 
        com = np.mean(vertices, axis = 0)
        vertices = (vertices - com) * scale + com
    
    
    if MESH_ALIGNMENT == 'XY' : 
        uvs = vertices[..., :2]
    elif MESH_ALIGNMENT == 'XZ': 
        uvs = np.stack([vertices[..., 0], vertices[..., 2]], axis = -1)
    uvs = (uvs - np.min(uvs, axis=0)) / (np.max(uvs, axis=0) - np.min(uvs, axis=0))
    
    return vertices, np.array(faces), np.array(sorted(list(edges))), uvs


def simulate(h, rest_vertices, faces, edges, masses, constraint_vertices, total_frames, stiffness, gpbd_iters, exp_dir): 

    x = np.copy(rest_vertices)
    v = np.zeros((rest_vertices.shape[0], 3))

    fem_model = SpringModel(stiffness)
    
    all_x = []

    for i in trange(total_frames): 
        x_prev = np.copy(x)
        v_prev = np.copy(v)

        # x_inertial = x_prev + h * v_prev
        
        ## Gravity force
        gravity_vector = - np.array([0, 9.8, 0])
        a = np.tile(gravity_vector[None], (x_prev.shape[0], 1))

        x = x_prev + h * v_prev + h*h*a ###????????????????????????

        x = gpbd_step(h, x, masses, rest_vertices, edges, fem_model, constraint_vertices, gpbd_iters,i)
        v = (x - x_prev) / h
            
        output_path = os.path.join(exp_dir, f'{i}.obj')
        dump_mesh(x, faces, output_path)

def gpbd_step(h, x_inertial, masses, rest_vertices, edges, fem_model, constraint_vertices, gpbd_iters,frame): 

    positions = x_inertial 

    Wfjs = np.zeros((edges.shape[0], 2, 3))

    ## Loop through all the constraints
    for k in range(gpbd_iters): 
        for (t, tri) in enumerate(edges): 
            Wfj = Wfjs[t]
            x_j = positions[tri] - Wfj

            mass = masses[tri]
            WInv = mass.repeat(3) / h / h

            x_rest = rest_vertices[tri]
            
            def objective(curr_Wfk): 
                x = x_j.flatten() +  curr_Wfk
                energy = fem_model.get_energy(x_rest, group(x))
                return 0.5 * (curr_Wfk.T @ np.diag(WInv) @ curr_Wfk) + energy 

            def jacobian(curr_Wfk): 
                x = x_j.flatten() +  curr_Wfk
                force = fem_model.get_force(x_rest, group(x))
                return np.diag(WInv) @ curr_Wfk - force.flatten() 

            def hessian(curr_Wfk): 
                x = x_j.flatten() +  curr_Wfk
                jacobian = fem_model.get_force_jacobian(x_rest, group(x))
                return np.diag(WInv) - jacobian 
                        
            res = scipy.optimize.minimize(objective, Wfj.flatten(), method='Newton-CG', jac=jacobian, hess=hessian)
            
            if not res.success:
                eps = 1e-6
                if np.linalg.norm(res.jac) < eps:
                    print(f"minimize reported an error but function norm < {eps}")
                else:
                    print(f"Solve failed. iter: {iter}, triangle: {t+1}, vertices: {tri}, message: {res.message}")

            Wfj = group(res.x)
            positions[tri] = Wfj + x_j
            Wfjs[t] = Wfj

        for cv in constraint_vertices:
            positions[cv] = rest_vertices[cv]

        x_new = positions
    return x_new

def main(): 
    side_length = 2
    bottom_left = (-2/2, -2/2)
    tessellation_level = 8
    timestep = 0.005
    density = 10
    total_frames = 200
    stiffness = 1000
    gpbd_iters = 2
    output_dir = 'outputs'
    exp_name = 'python'
    MESH_ALIGNMENT='XZ'
    
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok = True)

    rest_vertices, faces, edges, uvs = generate_triangulated_square_mesh(bottom_left, side_length, tessellation_level, MESH_ALIGNMENT)
    masses = compute_vertex_masses(rest_vertices, faces, density)
    constraint_vertices = [0, tessellation_level]
    simulate(timestep, rest_vertices, faces, edges, masses, constraint_vertices, total_frames, stiffness, gpbd_iters, exp_dir)

if __name__ == '__main__': 
    main()
