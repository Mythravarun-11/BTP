#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>
#include <utility>
#include <set>

using namespace Eigen;
using namespace std;

void dump_mesh(const MatrixXf &vertices, const MatrixXi &faces, const string &filepath) {
    
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

float compute_triangle_area(const MatrixXf &vertices){
    Vector3f vec1 = vertices.row(0) - vertices.row(1);
    Vector3f vec2 = vertices.row(0) - vertices.row(2);
    return 0.5 * vec1.cross(vec2).norm();
}

VectorXf compute_vertex_masses(const MatrixXf &rest_positions, const MatrixXi &triangles, float density){
    int num_vertices = rest_positions.rows();
    VectorXf masses = VectorXf::Zero(num_vertices);

    for(int i=0; i<triangles.rows();++i){
        MatrixXf X_rest(3,3);
        for(int j=0;j<3;++j){
            X_rest.row(j) = rest_positions.row(triangles(i,j));
        }

        float area = compute_triangle_area(X_rest);
        float mass_tri = density*area;
        float mass_contribution = mass_tri/(3.0f);

        for(int j=0;j<3;++j){
            masses(triangles(i,j)) += mass_contribution;
        }
    }
    return masses;
}

tuple<MatrixXf, MatrixXi, vector<pair<int,int>>, MatrixXf> generate_triangulated_square_mesh(
    const Vector2f &bottom_left, float side_length, int tesselation_level, const string& mesh_alignment, float scale = 1.0) {
    
    vector<Vector3f> vertices;
    vector<Vector3i> faces;
    set<pair<int, int>> edges;
    float x0 = bottom_left(0);
    float y0 = bottom_left(1);
    float step = side_length / tesselation_level;

    for(int i=0; i<=tesselation_level; ++i){
        for(int j=0; j<=tesselation_level; ++j){
            if(mesh_alignment=="XZ"){
                vertices.emplace_back(x0+i*step,0,y0+j*step);
            } else if(mesh_alignment=="XY"){
                vertices.emplace_back(x0+i*step,y0+j*step,0);
            }
        }
    }

    for(int i=0; i<tesselation_level; ++i){
        for(int j=0; j<tesselation_level; ++j){
            int v0 = i * (tesselation_level + 1) + j;
            int v1 = v0 + 1;
            int v2 = v0 + (tesselation_level + 1);
            int v3 = v2 + 1;

            faces.emplace_back(v0,v2,v1);
            faces.emplace_back(v1, v2, v3);

            edges.insert({min(v0, v1), max(v0, v1)});
            edges.insert({min(v0, v2), max(v0, v2)});
            edges.insert({min(v1, v3), max(v1, v3)});
            edges.insert({min(v2, v3), max(v2, v3)});
            edges.insert({min(v1, v2), max(v1, v2)});
        }
    }

    MatrixXf vertex_matrix(vertices.size(), 3);
    for (int i = 0; i < vertices.size(); ++i) {
        vertex_matrix.row(i) = vertices[i];
    }

    if (scale != 1.0) {
        Vector3f com = vertex_matrix.colwise().mean();
        vertex_matrix = (vertex_matrix.rowwise() - com.transpose()) * scale + com.transpose();
    }

    MatrixXi face_matrix(faces.size(), 3);
    for (int i = 0; i < faces.size(); ++i) {
        face_matrix.row(i) = faces[i];
    }

    MatrixXf uvs(vertices.size(), 2);
    vector<pair<int, int>> edge_list(edges.begin(), edges.end());
    return {vertex_matrix, face_matrix, edge_list, uvs};
}


__device__ void spring_force(float x_rest0[], float x_rest1[], float x0[], float x1[], float f0[], float f1[], float stiffness, int t){
    float L0 = 0.0f, L1 = 0.0f, x21[3];
    for(int i=0;i<3;++i){
        L0 += (x_rest0[i]-x_rest1[i])*(x_rest0[i]-x_rest1[i]);
        L1 += (x0[i]-x1[i])*(x0[i]-x1[i]);
        x21[i] = x1[i]-x0[i];
    }
    L0 = sqrtf(L0); L1 = sqrtf(L1);
    // printf("t,l0,l1 %d %f %f \n",t,L0,L1);
    // printf("t,x0,x1 %d %f %f %f %f %f %f \n",t,x0[0],x0[1],x0[2],x1[0],x1[1],x1[2]);
    float force_magnitude = (stiffness)*((L1-L0)/L1);
    for(int i=0;i<3;i++){
        f0[i] = force_magnitude*x21[i];
        f1[i] = -f0[i];
    }
    return;
}

__device__ void spring_force_derivatives(float x_rest0[], float x_rest1[], float x0[], float x1[], int thread_id, float *hessians, float stiffness){
    float L0 = 0.0f, L1 = 0.0f, x21[3], x21_normalized[3];
    for(int i=0;i<3;++i){
        L0 += (x_rest0[i]-x_rest1[i])*(x_rest0[i]-x_rest1[i]);
        L1 += (x0[i]-x1[i])*(x0[i]-x1[i]);
        x21[i] = x1[i]-x0[i];
    }
    L0 = sqrtf(L0); L1 = sqrtf(L1);
    for (int i = 0; i < 3; ++i) {
        x21_normalized[i] = x21[i] / L1;
    }
    float x21x21Dyadic[3][3], mat[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            x21x21Dyadic[i][j] = x21_normalized[i] * x21_normalized[j];
        }
    }
    float identity[3][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat[i][j] = -stiffness * ((1.0f - L0 / L1) * (identity[i][j] - x21x21Dyadic[i][j]) + x21x21Dyadic[i][j]);
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            hessians[thread_id*36 + i * 6 + j] = mat[i][j];            // Top-left 3x3 block
            hessians[thread_id*36 + (i + 3) * 6 + (j + 3)] = mat[i][j]; // Bottom-right 3x3 block
            hessians[thread_id*36 + i * 6 + (j + 3)] = -mat[i][j];      // Top-right 3x3 block
            hessians[thread_id*36 + (i + 3) * 6 + j] = -mat[i][j];      // Bottom-left 3x3 block
        }
    }
    return;
}

__device__ void LLT_custom(float *A, float *L, int thread_id){
    for(int i=0;i<6;++i){
        for(int j=0;j<=i;++j){
            float sum = 0.0f;
            for(int k=0;k<j;++k){
                sum += L[thread_id*36 + i*6 + k] * L[thread_id*36 + j*6 + k];
            }

            if(i==j){
                float value = A[thread_id*36 + i*6 + i] - sum;
                L[thread_id*36 + i*6 + j] = sqrtf(value);
            } else{
                L[thread_id*36 + i*6 + j] = (A[thread_id*36 + i*6 + j] - sum)/L[thread_id*36 + j*6 + j];
            }
        }
    }
    return;
}

__device__ void forward_substitution(float *L, float *b, float *y, int thread_id){
    for(int i=0;i<6;++i){
        float sum = 0.0f;
        for(int j=0;j<i;++j){
            sum += y[thread_id*6 + j]*L[thread_id*36 + i*6 + j];
        }
        y[thread_id*6 + i] = (b[thread_id*6 + i]-sum)/L[thread_id*36 + i*6 + i];
    }
    return;
}

__device__ void backward_substituion(float *L, float *y, float *x, int thread_id){
    for(int i=5;i>=0;i--){
        float sum = 0.0f;
        for(int j=i+1;j<6;j++){
            sum += x[thread_id*6 + j]*L[thread_id*36 + j*6 + i];
        }
        x[thread_id*6 + i] = (y[thread_id*6 + i]-sum)/L[thread_id*36 + i*6 + i];
    }
    return;
}

__device__ void LLTSolve(float *A, float *b, float *L, float *y, float *x, int thread_id){
    LLT_custom(A,L,thread_id);
    forward_substitution(L,b,y,thread_id);
    backward_substituion(L,y,x,thread_id);
    return;
}


__global__ void solve_iteration(const int num_edges, float stiffness, int *edges_gpu, int *temp_count_gpu, float *Wfjs_gpu, 
                                float *positions_gpu, float *x_temp_gpu, float *masses_gpu, float *rest_vertices_gpu,
                                float h, float *jacobians, float *hessians, float *L, float *y, float *x) {
    
    int t = blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=num_edges) return;

    int e0 = edges_gpu[t*2], e1 = edges_gpu[t*2+1];
    // printf("t,e0,e1 %d %d %d \n",t,e0,e1);
    float Wfj0[3], Wfj1[3], x_j0[3], x_j1[3], x_rest0[3], x_rest1[3], m0 = masses_gpu[e0], m1 = masses_gpu[e1];
    for(int i=0;i<3;++i){
        Wfj0[i] = Wfjs_gpu[t*6+i];
        Wfj1[i] = Wfjs_gpu[t*6+i+3];
    }
    for(int i=0;i<3;++i){
        x_j0[i] = positions_gpu[e0*3+i]-Wfj0[i];
        x_j1[i] = positions_gpu[e1*3+i]-Wfj1[i];
        x_rest0[i] = rest_vertices_gpu[e0*3+i];
        x_rest1[i] = rest_vertices_gpu[e1*3+i];
    }
    // printf("t,x_j0,x_j1 %d %f %f %f %f %f %f \n",t,positions_gpu[e0*3+0],positions_gpu[e0*3+1],positions_gpu[e0*3+2],positions_gpu[e1*3+0],positions_gpu[e1*3+1],positions_gpu[e1*3+2]);

    const int max_newton_iters = 1;
    const float eps = 1e-8;

    for (int cur_newton_iter = 0; cur_newton_iter < max_newton_iters; cur_newton_iter++){
        float x0[3], x1[3], f0[3], f1[3];
        for(int i=0;i<3;i++){
            x0[i] = x_j0[i]+Wfj0[i];
            x1[i] = x_j1[i]+Wfj1[i];
        }
        spring_force(x_rest0, x_rest1, x0, x1, f0, f1, stiffness,t); // calculate spring force
        // printf("force vals 0 %d %f %f %f \n",t,f0[0],f0[1],f0[2]);
        // printf("force vals 1 %d %f %f %f \n",t,f1[0],f1[1],f1[2]);

        for(int i=0;i<3;i++){                                      // calculate jacobians
            jacobians[t*6+i] = f0[i] - (m0*Wfj0[i])/(h*h);
            jacobians[t*6+i+3] = f1[i] - (m1*Wfj1[i])/(h*h);
        }
        spring_force_derivatives(x_rest0, x_rest1, x0, x1, t, hessians, stiffness);
        for(int i=0;i<3;++i){                                      // calculate hessians
            hessians[t*36 + i*6 + i] = m0/(h*h) - hessians[t*36 + i*6 + i];
            hessians[t*36 + (i+3)*6 + i+3] = m0/(h*h) - hessians[t*36 + (i+3)*6 + i+3];
            // printf("t,i,hessians %d %d %f \n",t,i,hessians[t*36 + i*6 + i]);
        }
        
        LLTSolve(hessians,jacobians,L,y,x,t);
        float norm_val = 0.0f;
        for(int i=0;i<6;i++){
            // printf("t,i,x after LLTSolve %d %d %f \n",t,i,x[t*6 + i]);
            norm_val += x[t*6 + i]*x[t*6 + i];
        }
        norm_val = sqrtf(norm_val);
        if(norm_val<eps){
            break;
        }
        
        for(int i=0;i<3;++i){
            Wfj0[i] += x[t*6 + i]; 
            Wfj1[i] += x[t*6 + i+3];
        }
    }

    // Atomic add
    for(int i=0;i<3;i++){
        // printf("cdsdfrx%f \n", *(x_temp_gpu+e0*3+i));
        atomicAdd(x_temp_gpu+e0*3+i,Wfj0[i]+x_j0[i]);
        atomicAdd(x_temp_gpu+e1*3+i,Wfj1[i]+x_j1[i]);
        // printf("x_temp_updates %f %f \n", Wfj0[i],Wfj1[i]);
    }
    atomicAdd(temp_count_gpu+e0,1);
    atomicAdd(temp_count_gpu+e1,1);

    for(int i=0;i<3;i++){
        Wfjs_gpu[t*6 + i] = Wfj0[i];
        Wfjs_gpu[t*6 + i+3] = Wfj1[i];
    }
    return;
}

MatrixXf gpbd_step(float h, const MatrixXf &x_inertial,const VectorXf &masses, const MatrixXf &rest_vertices,
                   const vector<pair<int,int>> &edges, const vector<int> &constraint_vertices, int gpbd_iters, float stiffness, int frame) {
    
    int num_vertices = x_inertial.rows();
    int num_edges = edges.size();
    // cout<<"\n\n\n\n\n\n\n\n\n\n\n"<<"frame count "<<frame<<"\n";
    
    MatrixXf positions = x_inertial;
    vector<MatrixXf> Wfjs(edges.size(), MatrixXf::Zero(2, 3));
    
    for(int k=0;k<gpbd_iters;++k){
        int *edges_gpu, *temp_count_gpu;
        float *Wfjs_gpu, *positions_gpu, *x_temp_gpu, *rest_vertices_gpu, *masses_gpu;

        cudaMalloc(&edges_gpu, 2*num_edges*sizeof(int));
        cudaMalloc(&temp_count_gpu, num_vertices*sizeof(int));
        cudaMalloc(&Wfjs_gpu, num_edges*2*3*sizeof(float));
        cudaMalloc(&positions_gpu, num_vertices*3*sizeof(float));
        cudaMalloc(&x_temp_gpu, num_vertices*3*sizeof(float));
        cudaMalloc(&rest_vertices_gpu, num_vertices*3*sizeof(float));
        cudaMalloc(&masses_gpu, num_vertices*sizeof(float));

        int *edges_flat = new int[num_edges*2];
        for (int i = 0; i < num_edges; ++i) {
            edges_flat[2 * i] = edges[i].first;
            edges_flat[2 * i + 1] = edges[i].second;
        }

        float *Wfjs_flat = new float[num_edges*2*3];
        for (int i = 0; i < num_edges; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int d = 0; d < 3; ++d) {
                    Wfjs_flat[i * 6 + j * 3 + d] = Wfjs[i](j, d);
                }
            }
        }

        float *positions_flat = new float[num_vertices*3];
        for(int i=0;i<num_vertices;++i){
            for(int j=0;j<3;j++){
                positions_flat[i*3+j] = positions(i,j);
            }
        }

        float *rest_vertices_flat = new float[num_vertices*3];
        for(int i=0;i<num_vertices;++i){
            for(int j=0;j<3;j++){
                rest_vertices_flat[i*3+j] = rest_vertices(i,j);
            }
        }

        float *masses_flat = new float[num_vertices];
        for(int i=0;i<num_vertices;++i){
            masses_flat[i] = masses(i);
        }


        cudaMemcpy(edges_gpu, edges_flat, 2*num_edges*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(temp_count_gpu, 0, num_vertices*sizeof(int));
        cudaMemset(x_temp_gpu, 0 ,num_vertices*3*sizeof(float));
        cudaMemcpy(Wfjs_gpu, Wfjs_flat, num_vertices*2*3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(positions_gpu, positions_flat, num_vertices*3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(rest_vertices_gpu, rest_vertices_flat, num_vertices*3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(masses_gpu, masses_flat, num_vertices*sizeof(float), cudaMemcpyHostToDevice);

        float *jacobians, *hessians, *L, *y, *x;
        cudaMalloc(&jacobians, num_edges*6*sizeof(float));
        cudaMalloc(&hessians, num_edges*6*6*sizeof(float));
        cudaMalloc(&L, num_edges*6*6*sizeof(float));
        cudaMalloc(&y, num_edges*6*sizeof(float));
        cudaMalloc(&x, num_edges*6*sizeof(float));

        cudaMemset(jacobians, 0, num_edges*6*sizeof(float));
        cudaMemset(hessians, 0, num_edges*6*6*sizeof(float));
        cudaMemset(L, 0, num_edges*6*6*sizeof(float));
        cudaMemset(y, 0, num_edges*6*sizeof(float));
        cudaMemset(x, 0, num_edges*6*sizeof(float));
        
        int threads_per_block = 256;
        int blocks_per_grid = (num_edges+threads_per_block)/(threads_per_block);

        solve_iteration<<<blocks_per_grid,threads_per_block>>>(num_edges,stiffness,edges_gpu,temp_count_gpu,
                                                             Wfjs_gpu, positions_gpu, x_temp_gpu, masses_gpu, rest_vertices_gpu,
                                                             h, jacobians, hessians, L, y ,x);

        cudaDeviceSynchronize();

        float *x_temp_final = new float[num_vertices*3];
        cudaMemcpy(x_temp_final, x_temp_gpu, num_vertices*3*sizeof(float), cudaMemcpyDeviceToHost);

        int *temp_count_host = new int[num_vertices];
        cudaMemcpy(temp_count_host, temp_count_gpu, num_vertices*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(Wfjs_flat, Wfjs_gpu, num_edges*2*3*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_edges; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int d = 0; d < 3; ++d) {
                    Wfjs[i](j, d) = Wfjs_flat[i * 6 + j * 3 + d];
                }
            }
        }
        for(int i=0; i<num_vertices;i++){
            // cout<<"x_temp "<<x_temp_final[i*3]<<" "<<x_temp_final[i*3+1]<<" "<<x_temp_final[i*3+2]<<'\n';
            positions.row(i) = Vector3f(x_temp_final[i*3],x_temp_final[i*3+1],x_temp_final[i*3+2]);
            if(temp_count_host[i]>0){
                positions.row(i) /= static_cast<float>(temp_count_host[i]);
            }
        }
        for(int cv : constraint_vertices){
            positions.row(cv) = rest_vertices.row(cv);
        }
        delete[] x_temp_final;
        delete[] temp_count_host;
        delete[] positions_flat;
        delete[] Wfjs_flat;
        delete[] masses_flat;
        delete[] edges_flat;
        delete[] rest_vertices_flat;
        cudaFree(edges_gpu);
        cudaFree(temp_count_gpu);
        cudaFree(Wfjs_gpu);
        cudaFree(positions_gpu);
        cudaFree(x_temp_gpu);
        cudaFree(rest_vertices_gpu);
        cudaFree(masses_gpu);
        cudaFree(jacobians);
        cudaFree(hessians);
        cudaFree(L);
        cudaFree(y);
        cudaFree(x);
    }
    return positions;
}


void simulate(float h, const MatrixXf &rest_vertices, const MatrixXi &faces, const vector<pair<int,int>> &edges,
              const VectorXf &masses, const vector<int> &constraint_vertices, int total_frames, float stiffness,
              int gpbd_iters, const string &exp_dir) {
    
    MatrixXf x = rest_vertices;
    MatrixXf v = MatrixXf::Zero(rest_vertices.rows(),3);

    for(int i=0;i<total_frames;++i){
        MatrixXf x_prev = x;
        MatrixXf v_prev = v;
        Vector3f gravity_vector(0.0f,-9.8f,0.0f);
        MatrixXf a = (MatrixXf::Constant(x.rows(),3,1).array().rowwise())*(gravity_vector.transpose().array());

        x = x_prev + h*v_prev + h*h*a;
        x = gpbd_step(h, x, masses, rest_vertices, edges, constraint_vertices, gpbd_iters, stiffness, i);
        v = (x-x_prev)/h;

        string output_path = exp_dir + "/" + to_string(i) + ".obj";
        dump_mesh(x,faces,output_path);
    }
}

int main(){
    float side_length = 2.0f;
    Eigen::Vector2f bottom_left(-1.0f,-1.0f);
    int tesselation_level = 8;
    float time_step = 0.005f;
    int total_frames = 500;
    float stiffness = 1000.0f;
    float density = 10.0f;
    int gpbd_iters = 1;
    string output_dir = "outputs";
    string exp_name = "cu";
    string mesh_alignment = "XZ";

    string exp_dir = output_dir + "/" + exp_name;
    auto result = generate_triangulated_square_mesh(bottom_left, side_length, tesselation_level, mesh_alignment);
    auto& rest_vertices = std::get<0>(result);
    auto& faces = std::get<1>(result);
    auto& edges = std::get<2>(result);
    auto& uvs = std::get<3>(result);
    vector<int> constraint_vertices = {0,tesselation_level};
    VectorXf masses = compute_vertex_masses(rest_vertices, faces, density);
    simulate(time_step, rest_vertices, faces, edges, masses, constraint_vertices, total_frames, stiffness, gpbd_iters, exp_dir);

    return 0;
}