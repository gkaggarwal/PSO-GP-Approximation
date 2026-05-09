# PSO-GP Framework

## Overview

PSO-GP is a hybrid evolutionary optimization framework for hardware-efficient approximation of nonlinear activation functions. The framework combines Particle Swarm Optimization (PSO) and Genetic Programming (GP) to generate optimized piecewise linear (PWL) approximations suitable for FPGA and ASIC implementations.

---

## Features

* PSO-based interval optimization
* GP-based linear approximation
* Hardware-efficient PWL generation
* Support for Sigmoid and tanh functions
* MATLAB implementation
* FPGA/ASIC-oriented evaluation

---

## Repository Structure

```text
PSO-GP/
├── datasets/
├── pso/
├── gp/
├── hardware/
├── results/
├── main.m
└── README.md
```

---

## Requirements

* MATLAB
* GP-OLS Toolbox
* Vivado 2020.2 (optional for FPGA synthesis)

---

## Run

```matlab
main
```

---

## Hardware Implementation

* Verilog HDL
* Xilinx XC7Z020 FPGA
* Vivado Design Suite 2020.2
* 45 nm Nangate Open Cell Library
* Single-precision floating-point arithmetic

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{pso_gp_activation,
  title={PSO-GP: An Evolutionary Optimization-based Activation Function Approximation for Hardware Efficient Neural Network},
  author={Gurve, Mahendra Kumar and Kumar, Gaurav and Kumar, Anuj and Ahlawat, Satyadev and Prasad, Yamuna},
  journal={Computers \& Electrical Engineering},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---

## Contact

### Gaurav Kumar

Department of Electrical Engineering
Indian Institute of Technology Jammu
Email: [gaurav.kumar@iitjammu.ac.in](mailto:gaurav.kumar@iitjammu.ac.in)

---

## Acknowledgment

This work was carried out at the Indian Institute of Technology Jammu as part of research on hardware-efficient neural network acceleration and nonlinear activation function approximation.
