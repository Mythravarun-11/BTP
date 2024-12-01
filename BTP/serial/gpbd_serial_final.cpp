#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <set>
#include <algorithm>
#include <iomanip>
#include <dlib/optimization.h>

using namespace Eigen;
using namespace std;


typedef dlib::matrix<double, 0, 1> column_vector;

// Convert Eigen VectorXd to Dlib column_vector
column_vector eigen_to_dlib_vector(const VectorXd& v) {
    column_vector cv(v.size());
    for (int i = 0; i < v.size(); ++i)
        cv(i) = v(i);
    return cv;
}

// Convert Dlib column_vector to Eigen VectorXd
VectorXd dlib_to_eigen_vector(const column_vector& cv) {
    VectorXd v(cv.size());
    for (int i = 0; i < v.size(); ++i)
        v(i) = cv(i);
    return v;
}

class SpringModel {
public:
    double k;
    
    SpringModel(double k) : k(k) {}

    double get_energy(const MatrixXd &x_rest, const MatrixXd &x_cur) {
        double rest_length = (x_rest.row(1) - x_rest.row(0)).norm();
        double cur_length = (x_cur.row(1) - x_cur.row(0)).norm();
        return 0.5 * k * pow(cur_length - rest_length, 2);
    }

    MatrixXd get_force(const MatrixXd &x_rest, const MatrixXd &x_cur) {
        double rest_length = (x_rest.row(1) - x_rest.row(0)).norm();
        double cur_length = (x_cur.row(1) - x_cur.row(0)).norm();
        Vector3d x21 = x_cur.row(1) - x_cur.row(0);
        double force_magnitude = k * (cur_length - rest_length);
        MatrixXd forces(2, 3);
        Vector3d force = force_magnitude * (x21 / cur_length);
        forces.row(0) = force;
        forces.row(1) = -force;
        return forces;
    }

    MatrixXd get_force_jacobian(const MatrixXd &x_rest, const MatrixXd &x_cur) {
        double rest_length = (x_rest.row(1) - x_rest.row(0)).norm();
        double cur_length = (x_cur.row(1) - x_cur.row(0)).norm();
        Vector3d x21 = x_cur.row(1) - x_cur.row(0);
        Matrix3d x21x21Dyadic = (x21 / cur_length) * (x21 / cur_length).transpose();
        Matrix3d mat = -k * ((1 - rest_length / cur_length) * (Matrix3d::Identity() - x21x21Dyadic) + x21x21Dyadic);
        Matrix3d neg_mat = -mat;
        MatrixXd jacobian(6, 6);
        jacobian.topLeftCorner(3, 3) = mat;
        jacobian.topRightCorner(3, 3) = neg_mat;
        jacobian.bottomLeftCorner(3, 3) = neg_mat;
        jacobian.bottomRightCorner(3, 3) = mat;
        return jacobian;
    }
};



void dump_mesh(const MatrixXd &vertices, const MatrixXi &faces, const string &filepath) {
    
    ofstream file(filepath);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filepath << endl;
        return;
    }

    for (int i = 0; i < vertices.rows(); ++i) {
        file << "v " << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << "\n";
    }

    for (int i = 0; i < faces.rows(); ++i) {
        file << "f " << faces(i, 0) + 1 << " " << faces(i, 1) + 1 << " " << faces(i, 2) + 1 << "\n";
    }

    file.close();
}

double compute_triangle_area(const MatrixXd &vertices) {
    Vector3d vec1 = vertices.row(0) - vertices.row(1);
    Vector3d vec2 = vertices.row(0) - vertices.row(2);
    return 0.5 * vec1.cross(vec2).norm();
}

VectorXd compute_vertex_masses(const MatrixXd &rest_positions, const MatrixXi &triangles, double density) {
    int num_vertices = rest_positions.rows();
    VectorXd masses = VectorXd::Zero(num_vertices);

    for (int i = 0; i < triangles.rows(); ++i) {
        MatrixXd X_rest(3, 3);
        for (int j = 0; j < 3; ++j) {
            X_rest.row(j) = rest_positions.row(triangles(i, j));
        }

        double area = compute_triangle_area(X_rest);
        double mass_tri = density * area;
        double mass_contribution = mass_tri / 3.0;

        for (int j = 0; j < 3; ++j) {
            masses(triangles(i, j)) += mass_contribution;
        }
    }

    return masses;
}

tuple<MatrixXd, MatrixXi, vector<pair<int, int>>, MatrixXd> generate_triangulated_square_mesh(
    const Vector2d &bottom_left, double side_length, int tessellation_level, const string &MESH_ALIGNMENT, double scale = 1.0) {

    vector<Vector3d> vertices;
    vector<Vector3i> faces;
    set<pair<int, int>> edges;

    double x0 = bottom_left(0);
    double y0 = bottom_left(1);
    double step = side_length / tessellation_level;
    
    for (int i = 0; i <= tessellation_level; ++i) {
        for (int j = 0; j <= tessellation_level; ++j) {
            
            if (MESH_ALIGNMENT == "XZ") {
                vertices.emplace_back(x0 + i * step, 0, y0 + j * step);
            } else if (MESH_ALIGNMENT == "XY") {
                vertices.emplace_back(x0 + i * step, y0 + j * step, 0);
            }
        }
    }
    
    for (int i = 0; i < tessellation_level; ++i) {
        for (int j = 0; j < tessellation_level; ++j) {
            int v0 = i * (tessellation_level + 1) + j;
            int v1 = v0 + 1;
            int v2 = v0 + (tessellation_level + 1);
            int v3 = v2 + 1;

            faces.emplace_back(v0, v2, v1);
            faces.emplace_back(v1, v2, v3);

            edges.insert({min(v0, v1), max(v0, v1)});
            edges.insert({min(v0, v2), max(v0, v2)});
            edges.insert({min(v1, v3), max(v1, v3)});
            edges.insert({min(v2, v3), max(v2, v3)});
            edges.insert({min(v1, v2), max(v1, v2)});
        }
    }
    
    MatrixXd vertex_matrix(vertices.size(), 3);
    for (int i = 0; i < vertices.size(); ++i) {
        vertex_matrix.row(i) = vertices[i];
    }

    if (scale != 1.0) {
        Vector3d com = vertex_matrix.colwise().mean();
        vertex_matrix = (vertex_matrix.rowwise() - com.transpose()) * scale + com.transpose();
    }

    MatrixXi face_matrix(faces.size(), 3);
    for (int i = 0; i < faces.size(); ++i) {
        face_matrix.row(i) = faces[i];
    }

    MatrixXd uvs(vertices.size(), 2);    
    vector<pair<int, int>> edge_list(edges.begin(), edges.end());
    return {vertex_matrix, face_matrix, edge_list, uvs};
}

MatrixXd gpbd_step(double h, const MatrixXd &x_inertial, const VectorXd &masses, const MatrixXd &rest_vertices,
                   const vector<pair<int, int>> &edges, SpringModel &fem_model, const vector<int> &constraint_vertices, int gpbd_iters, int frame) {

    MatrixXd positions = x_inertial;
    vector<MatrixXd> Wfjs(edges.size(), MatrixXd::Zero(2, 3));

    for (int k = 0; k < gpbd_iters; ++k) {
        for (int t = 0; t < edges.size(); ++t) {
            MatrixXd Wfj = Wfjs[t];

            MatrixXd x_j(2, 3);
            x_j.row(0) = positions.row(edges[t].first) - Wfj.row(0);
            x_j.row(1) = positions.row(edges[t].second) - Wfj.row(1);

            double mass1 = masses(edges[t].first);
            double mass2 = masses(edges[t].second);
            VectorXd mass(6); 
            mass << mass1, mass1, mass1, mass2, mass2, mass2;

            VectorXd WInv = mass / pow(h, 2);

            MatrixXd x_rest(2, 3);
            x_rest.row(0) = rest_vertices.row(edges[t].first);
            x_rest.row(1) = rest_vertices.row(edges[t].second);

            // Objective function
            auto objective = [&fem_model, &x_rest, &x_j, &WInv](const column_vector& curr_Wfk_dlib) {
                VectorXd curr_Wfk = dlib_to_eigen_vector(curr_Wfk_dlib); // Updated function name
                MatrixXd x = x_j;
                x.row(0) += curr_Wfk.segment<3>(0).transpose();
                x.row(1) += curr_Wfk.segment<3>(3).transpose();
                double energy = fem_model.get_energy(x_rest, x);
                return 0.5 * curr_Wfk.transpose() * WInv.asDiagonal() * curr_Wfk + energy;
            };

            // Gradient function
            auto gradient = [&fem_model, &x_rest, &x_j, &WInv](const column_vector& curr_Wfk_dlib) {
                VectorXd curr_Wfk = dlib_to_eigen_vector(curr_Wfk_dlib);
                MatrixXd x = x_j;
                x.row(0) += curr_Wfk.segment<3>(0).transpose();
                x.row(1) += curr_Wfk.segment<3>(3).transpose();
                MatrixXd force = fem_model.get_force(x_rest, x);
                MatrixXd WInvdiag = WInv.asDiagonal();
                VectorXd result = WInvdiag * curr_Wfk - force.reshaped();
                return eigen_to_dlib_vector(result);
            };

            VectorXd curr_Wfk(6);
            curr_Wfk.segment<3>(0) = Wfj.row(0).transpose();
            curr_Wfk.segment<3>(3) = Wfj.row(1).transpose();
            column_vector curr_Wfk_dlib = eigen_to_dlib_vector(curr_Wfk);

            dlib::find_min_using_approximate_derivatives(
                dlib::bfgs_search_strategy(),
                dlib::objective_delta_stop_strategy(1e-7),
                objective,
                curr_Wfk_dlib,
                -1
            );

            curr_Wfk = dlib_to_eigen_vector(curr_Wfk_dlib);
            Wfj.row(0) = curr_Wfk.segment<3>(0).transpose();
            Wfj.row(1) = curr_Wfk.segment<3>(3).transpose();

            positions.row(edges[t].first) = Wfj.row(0) + x_j.row(0);
            positions.row(edges[t].second) = Wfj.row(1) + x_j.row(1);

            Wfjs[t] = Wfj;
        }

        for (int cv : constraint_vertices) {
            positions.row(cv) = rest_vertices.row(cv);
        }
    }
    return positions;
}


void simulate(double h, const MatrixXd &rest_vertices, const MatrixXi &faces, const vector<pair<int, int>> &edges,
              const VectorXd &masses, const vector<int> &constraint_vertices, int total_frames, double stiffness,
              int gpbd_iters, const string &exp_dir) {

    MatrixXd x = rest_vertices;
    MatrixXd v = MatrixXd::Zero(rest_vertices.rows(), 3);
    SpringModel fem_model(stiffness);

    for (int i = 0; i < total_frames; ++i) {
        MatrixXd x_prev = x;
        MatrixXd v_prev = v;

        Vector3d gravity_vector(0, -9.8, 0);
        MatrixXd a = MatrixXd::Constant(x.rows(), 3, 1).array().rowwise() * gravity_vector.transpose().array();

        x = x_prev + h * v_prev + h * h * a;
        x = gpbd_step(h, x, masses, rest_vertices, edges, fem_model, constraint_vertices, gpbd_iters, i);
        v = (x - x_prev) / h;

        string output_path = exp_dir + "/" + to_string(i) + ".obj";
        dump_mesh(x, faces, output_path);
    }
}

int main() {
    double side_length = 2;
    Vector2d bottom_left(-1, -1);
    int tessellation_level = 8;
    double timestep = 0.005;
    double density = 10;
    int total_frames = 200;
    double stiffness = 1000;
    int gpbd_iters = 2;
    string output_dir = "outputs";
    string exp_name = "cpp";
    string MESH_ALIGNMENT = "XZ";

    string exp_dir = output_dir + "/" + exp_name;
    auto result = generate_triangulated_square_mesh(bottom_left, side_length, tessellation_level, MESH_ALIGNMENT);
    auto& rest_vertices = std::get<0>(result);
    auto& faces = std::get<1>(result);
    auto& edges = std::get<2>(result);
    auto& uvs = std::get<3>(result);
    VectorXd masses = compute_vertex_masses(rest_vertices, faces, density);
    vector<int> constraint_vertices = {0, tessellation_level};
    simulate(timestep, rest_vertices, faces, edges, masses, constraint_vertices, total_frames, stiffness, gpbd_iters, exp_dir);

    return 0;
}