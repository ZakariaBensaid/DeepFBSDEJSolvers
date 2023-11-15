README for "Deep Learning Algorithms for FBSDEs with Jumps: Applications to Option Pricing and a MFG Model for Smart Grids"
Abstract
In this repository, we present the code accompanying our paper titled "Deep Learning Algorithms for FBSDEs with Jumps: Applications to Option Pricing and a MFG Model for Smart Grids". Our work introduces advanced machine learning solvers for coupled Forward-Backward Stochastic Differential Equations (FBSDEs) with jumps. We provide detailed numerical simulations to compare our algorithms and demonstrate their effectiveness, particularly in option pricing and mean field game (MFG) models for smart grids. The codebase includes an extension of the MFG model to accommodate jumps modeled by a doubly Poisson process and a FBSDE system driven by a Cox process. Our approach provides new insights into existence results, a comparison with central planner problems, and showcases the utility of deep learning algorithms in managing jumps processes with stochastic intensity.

Contents
coupledMFG: This directory contains the codes for solvers, the mathematical mean field game model, networks, etc.
coupledPricing: This directory houses codes for solvers, including two different pricing models (Merton jump and Variance Gamma), networks, etc.
Installation
To install and run the code, ensure you have Python installed on your system. You can then clone this repository and install the required packages using:

Copy code
pip install tensorflow numpy scipy
Usage
After installation, you can run the scripts in the coupledMFG and coupledPricing directories. Refer to the individual README files in these directories for specific usage instructions.

Dependencies
This project depends on the following Python packages:

TensorFlow
NumPy
SciPy
Make sure these are installed as mentioned in the Installation section.

Contributing
We welcome contributions to this project. If you have suggestions or improvements, please create a pull request or open an issue.

License
[Specify the license under which this code is released, e.g., MIT, GPL, etc.]

Reference
If you use our code in your research, please cite our paper. [Include BibTeX entry here]

Contact
For questions or collaboration requests, please contact [Your Contact Information].

Acknowledgments
We would like to thank all contributors, funders, and supporters of this project.
