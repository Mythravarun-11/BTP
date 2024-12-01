module load compiler/gcc/7.4.0/compilervars
module load lib/eigen3/3.4.0/gcc9.1
module load compiler/cuda/10.2/compilervars
# g++ -std=c++14 gpbd.cpp -o gpbd
# g++ -std=c++14 check.cpp -o check
# nvcc checker.cu -o checker
# nvcc --expt-relaxed-constexpr -o gpbd_parallel gpbd_parallel.cu
nvcc --expt-relaxed-constexpr -o gpbd_coloring gpbd_graph_coloring.cu
./gpbd_coloring > out.txt


# module load compiler/gcc/7.4.0/compilervars
# module load lib/eigen3/3.4.0/gcc9.1

# g++ -std=c++14 gpbd.cpp -o gpbd -I./lbfgspp/include -I/home/apps/centos7/eigen3/include/eigen3
# module load compiler/python/3.6.0/ucs4/gnu/447
# module load pythonpackages/3.6.0/numpy/1.16.1/gnu
# module load pythonpackages/3.6.0/tqdm/4.25.0/gnu
# module load pythonpackages/3.6.0/ucs4/gnu/447/scipy/0.18.1/intel
# python3 gpbd_edge_basic.py

# cmake .. -DCMAKE_INSTALL_PREFIX=/home/cse/btech/cs1210106/dlib_install -DUSE_CUDA=1