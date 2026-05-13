# PSO-GP: Evolutionary Optimization-based Activation Function Approximation

> Piecewise linear approximation of neural network activation functions using **Particle Swarm Optimization (PSO)** for interval partitioning and **Genetic Programming (GP)** for coefficient optimization.

---

## Overview

PSO-GP jointly optimizes interval boundaries and linear approximation coefficients for any nonlinear activation function. The output is a set of piecewise linear equations `L_j(x) = a_j*x + b_j` that closely approximate the target function with minimal hardware resources.

**Paper:** *PSO-GP: An Evolutionary Optimization-based Activation Function Approximation for Hardware Efficient Neural Network*  
**Authors:** Mahendra Kumar Gurve, Gaurav Kumar, Anuj Kumar, Satyadev Ahlawat, Yamuna Prasad  
**Affiliation:** Indian Institute of Technology Jammu, India  
**Code:** https://github.com/xxx/PSO-GP-Activation

---

## Requirements

- MATLAB R2019b or later
- **GP-OLS Toolbox** (mandatory): http://www.fmt.veim.hu/softcomp

Download the toolbox and add it to your MATLAB path:
```matlab
addpath('path/to/gpols_toolbox');
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/xxx/PSO-GP-Activation.git
cd PSO-GP-Activation
```

### 2. Add GP-OLS Toolbox to MATLAB path
```matlab
addpath('path/to/gpols_toolbox');
```

### 3. Set your configuration in `pso_gp_approximation.m`

Open the file and edit the **User Configuration** section:

```matlab
ACTIVATION_FUNCTION = 'sigmoid';   % 'sigmoid', 'tanh', or 'gelu'
NUM_INTERVALS       = 13;          % number of piecewise linear segments
LOWER_BOUND         = 0;           % start of input domain
UPPER_BOUND         = 8;           % end of input domain
```

### 4. Run
```matlab
run('pso_gp_approximation.m')
```

---

## Configuration Options

| Parameter | Default | Description |
|---|---|---|
| `ACTIVATION_FUNCTION` | `'sigmoid'` | Target function: `'sigmoid'`, `'tanh'`, `'gelu'` |
| `NUM_INTERVALS` | `13` | Number of piecewise linear segments |
| `LOWER_BOUND` | `0` | Start of approximation domain |
| `UPPER_BOUND` | `8` | End of approximation domain |
| `NUM_PARTICLES` | `30` | Number of PSO particles |
| `NUM_ITERATIONS` | `20` | Number of PSO iterations |
| `INERTIA` | `0.5` | PSO inertia weight (ω) |
| `COGNITIVE` | `1.5` | PSO cognitive coefficient (c₁) |
| `SOCIAL` | `1.5` | PSO social coefficient (c₂) |
| `GP_POP_SIZE` | `50` | GP population size |
| `GP_MAX_DEPTH` | `2` | GP maximum tree depth |
| `GP_GENERATIONS` | `10` | GP generations per segment |
| `SAVE_RESULTS` | `true` | Save results to text file |
| `OUTPUT_FILE` | `'pso_gp_results.txt'` | Output filename |

---

## Adding a Custom Activation Function

In the **Activation Function Definition** section, add a new case:

```matlab
case 'swish'
    act_fn  = @(x) x ./ (1 + exp(-x));
    fn_name = 'Swish';
```

Then set:
```matlab
ACTIVATION_FUNCTION = 'swish';
```

---

## Output

### Console
```
=============================================================
 PSO-GP Activation Function Approximation
=============================================================
 Function  : Sigmoid
 Intervals : 13
 Domain    : [0.0, 8.0]
 Particles : 30 | Iterations: 20
=============================================================

 Starting PSO optimization...

 Iteration   1/20 | Best MAE: 8.432100e-04
 Iteration   2/20 | Best MAE: 5.217300e-04
 ...
 Iteration  20/20 | Best MAE: 2.781000e-04

 Extracting final piecewise linear equations...

=============================================================
 RESULTS: Piecewise Linear Approximation of Sigmoid
=============================================================

 Seg    Interval                Slope (a)       Intercept (b)   Seg MAE
 ------------------------------------------------------------------------
 1      [  0.0000,   0.5831]   +0.24379300     +0.50080000     1.2300e-05
 2      [  0.5831,   1.0761]   +0.21113200     +0.52027800     9.8700e-06
 ...
 13     [  6.0626,   8.0000]   +0.00096700     +0.99217100     3.1200e-07

 Overall MAE : 2.781000e-04
 Overall EMAX: 1.720000e-03
=============================================================
```

### Output File (`pso_gp_results.txt`)
```
PSO-GP Approximation Results
============================
Activation Function : Sigmoid
Number of Segments  : 13
Domain              : [0.00, 8.00]
Overall MAE         : 2.781000e-04

Piecewise Linear Equations: F(x) = a*x + b
------------------------------------------------------------------------
Seg    Interval                Slope (a)       Intercept (b)   Seg MAE
------------------------------------------------------------------------
1      [  0.0000,   0.5831]   +0.24379300     +0.50080000     1.2300e-05
...
------------------------------------------------------------------------

Interval Boundaries:
[0 0.5831 1.0761 ... 8]
```

---

## Recommended Settings

| Function | `NUM_INTERVALS` | `LOWER_BOUND` | `UPPER_BOUND` |
|---|---|---|---|
| Sigmoid | 13 | 0 | 8 |
| Tanh    | 14 | 0 | 5 |
| GELU    | 13 | 0 | 8 |

> **Note:** PSO-GP exploits functional symmetry. The approximation is performed over `[0, UPPER_BOUND]` and the full range `[-UPPER_BOUND, UPPER_BOUND]` is recovered using `F(-x) = 1 - F(x)` for Sigmoid and `F(-x) = -F(x)` for Tanh/GELU.

---

## How It Works

```
Dataset Generation  →  PSO: global search over interval boundaries
                              ↓ (for each candidate partition)
                        GP: evolve linear coefficients L_j(x) = a_j·x + b_j
                              ↓
                        Evaluate TMAE fitness
                              ↓
                        Update PSO personal/global best
                              ↓
                    Return optimal piecewise linear approximation F'(x)
```

1. **PSO** searches over all interval boundary positions across a swarm of particles.
2. For each candidate partition, **GP** evolves the optimal linear coefficients within each segment.
3. Boundary optimization is always informed by the best achievable coefficient quality — this joint coupling is the key novelty of PSO-GP.

---

## Results (from paper)

| Function | Segments | MAE | EMAX | LUT | FF |
|---|---|---|---|---|---|
| Sigmoid | 13 | 2.78×10⁻⁴ | 1.72×10⁻³ | 31 | 27 |
| Tanh    | 14 | 4.74×10⁻⁴ | 2.63×10⁻³ | — | — |

---

## Citation

If you use this code, please cite:

```bibtex
@article{gurve2025psogp,
  title   = {PSO-GP: An Evolutionary Optimization-based Activation Function
             Approximation for Hardware Efficient Neural Network},
  author  = {Gurve, Mahendra Kumar and Kumar, Gaurav and Kumar, Anuj and
             Ahlawat, Satyadev and Prasad, Yamuna},
  journal = {Computers and Electrical Engineering},
  year    = {2025}
}
```

---

## License

This project is released for academic and research use.  
For commercial use, please contact the authors.

---

## Contact

- **Gaurav Kumar** — gaurav.kumar@iitjammu.ac.in
- **Mahendra Kumar Gurve** — mahendra.gurve@iitjammu.ac.in
- Indian Institute of Technology Jammu, India
