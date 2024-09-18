# README for "Deep Learning Algorithms for FBSDEs with Jumps: Applications to Option Pricing and a MFG Model for Smart Grids"

## Abstract
In this repository, we present the code accompanying our paper titled "Deep Learning Algorithms for FBSDEs with Jumps: Applications to Option Pricing and a MFG Model for Smart Grids". Our work introduces advanced machine learning solvers for coupled Forward-Backward Stochastic Differential Equations (FBSDEs) with jumps. We provide detailed numerical simulations to compare our algorithms and demonstrate their effectiveness, particularly in option pricing and mean field game (MFG) models for smart grids. The codebase includes an extension of the MFG model to accommodate jumps modeled by a doubly Poisson process and a FBSDE system driven by a Cox process. Our approach provides new insights into existence results, a comparison with central planner problems, and showcases the utility of deep learning algorithms in managing jumps processes with stochastic intensity.

## Contents
- `coupledMFG`: This directory contains the codes for MFG solvers, the code for the mathematical mean field game model, the neural networks and scripts to plot and compute the price of anarchy.
- `coupledPricing`: This directory houses codes for Pricing models solvers, including two different pricing models (Merton jump and Variance Gamma), the neural networks and a main script to compare all the algorithms.

## Dependencies
This project depends on the following Python packages:
- TensorFlow (GPU or CPU)
- NumPy
- SciPy

Make sure these are installed as mentioned in the Installation section.

## Contributing
We welcome contributions to this project. If you have suggestions or improvements, please create a pull request or open an issue.


## Reference
If you use our code in your research, please cite our paper as follows: 
@article{alasseur2023deep,
  title={Deep Learning Algorithms for FBSDEs with Jumps: Applications to Option Pricing and a MFG Model for Smart Grids},
  author={Alasseur, Clemence and Bensaid, Zakaria and Dumitrescu, Roxana and Warin, Xavier},
  note={Preprint November 2023}
}

## Acknowledgments
We would like to thank all contributors, funders, and supporters of this project.
