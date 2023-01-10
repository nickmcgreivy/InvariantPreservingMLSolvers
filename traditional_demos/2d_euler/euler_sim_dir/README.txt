Hi reader,

This file contains the code which runs simulations that give us our data, as well as the plots that analyze that data. You will need to download and possibly update Jax and Jaxlib, as well as have numpy/sympy/scipy installed. To run the code, you only need to edit two lines:

(1) makefile: edit the line that says "INSERT_DIRECTORY_HERE"

(2) data/args/2d_euler_args.txt: edit the "--cfl_safety 2.0", which when you type "make generate_energy_conservation_data" will need to be set to something smaller. I believe our energy-conserving experiments were run at "--cfl_safety 0.1".


To generate figure 2, type:


make generate_data
make plot_data
make plot_diagnostics


This will take a few hours to run, with the vast majority of that time being spent generating 1024x1024 simulation data. 
After you generate figure 2, you can generate figure 3. You will need to edit "--cfl_safety" (see above) and type:


make generate_energy_conservation_data
make plot_energy_data
make plot_energy_diagnostics



We use a precompiled C++ binary that works on Mac (MacOS Big Sur, Version 11.6.1) and possibly only works for Python 3.7. The code should run fine on Linux, since we have two different binaries in the folder. It may only run on Python 3.8. You will likely have trouble running on Windows due to the binary not being compiled for Windows. You may have trouble using a different version of Python than 3.7.

If you want to compile a new binary, you can go to code/simcode/generate_sparse_solve and type "make compilelinux" or "make compilemac", which should compile the C++ code and create a new binary which should run for your version of Python. You'd then need to move that binary into the folder code/simcode. This compilation will require installing additional packages, such as PyBind11.