# Step-by-Step Instructions

    1. Save Files: Save each code block to its respective file (tmm.cu, tmm.cpp, setup.py, and example.py).

    2. Compile the Extension: Run the following command in the directory containing setup.py to compile the extension:


	'''bash
	python setup.py install
	'''

    3. Run the Example: Once the extension is compiled and installed, you can test it by running:

	'''bash
	python example.py
	'''

Summary of Files

    - tmm.cu: Contains the CUDA kernel.
    - tmm.cpp: C++ file for binding the CUDA kernel to PyTorch.
    - setup.py: Script to compile and install the extension.
    - example.py: Python script to test the custom matrix multiplication function.

