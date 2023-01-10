compilemac:
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup -I eigen-3.4.0/ `python3 -m pybind11 --includes` sparse_solve.cpp -o custom_call_sparse_solve`python3-config --extension-suffix`

compilelinux:
	c++ -O3 -Wall -shared -std=c++11 -I eigen-3.4.0/ `python3 -m pybind11 --includes` sparse_solve.cpp -o custom_call_sparse_solve`python3-config --extension-suffix`

clean:
