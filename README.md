# Invariant Preservation in Machine Learned PDE Solvers
Code for JCP 2023 paper on preserving invariants in machine learned PDE solvers.



## How do I generate figures from section 4 and 5 of the paper?

In these figures, we illustrate the effect of applying these invariant-preserving error-correcting algorithms to traditional numerical methods. To generate and save the figures in section 4 and section 5, download this directory and type `cd traditional_demos`. You will then see 4 directories, `1d_advection`, `1d_burgers`, `2d_euler` and `1d_compressible_euler`. You will need to have jax and sympy installed.

To generate the figures involving the 1D Burgers' equation, type `cd 1d_burgers` and then `python burgers_4_?.py` for each of the files in the directory. 

To generate the figures involving the 1D Advection equation, type `cd 1d_advection` and then `python advection_4_3.py`.

To generate the figure for the 1D compressible Euler equations, type `cd 1d_compressible_euler/scripts`. Edit the beginning of the path in line 1 of `save_plot_data.py`. Once this is done, type `python save_plot_data.py`.

To generate the figures in section 4.6 for the 2D incompressible Euler equation, type `cd 2d_euler/euler_sim_dir/code/simcode/generate_sparse_solve`. Next type `tar -xvf eigen-3.4.0.tar`. If you are running on a mac, type `make compilemac` and if you are running on a linux type `make compilelinux`. Then type `mv custom_call_sparse_solve* ..`. Then type `cd ../..`. Now edit the file `makefile` and change the beginning of the path where it says "EDIT BELOW". Next, type `make generate_data plot_data plot_diagnostics`. This last step should take at least an hour to run.

## How do I generate figures from section 6 of the paper?

In these figures, we train machine learned solvers to solve various equations, and demonstrate that the error-correcting algorithms we introduce don't degrade the accuracy of an already-accurate solver. 