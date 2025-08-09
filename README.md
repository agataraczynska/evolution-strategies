# Evolution Strategies – Coursework Implementation

## Overview
This repository contains a Python implementation of **Evolution Strategies (ES)**, an optimisation method inspired by natural evolution.  
The program was developed as part of an academic assignment on **Evolutionary Algorithms**, and it applies ES to fit a given mathematical model to experimental data.

The implementation includes both `(μ, λ)` and `(μ + λ)` selection strategies and supports **mutation** as the primary genetic operator. Two **crossover** methods (discrete and intermediate) are also implemented and can be activated in the code.

## Problem Description
The task was to solve a provided optimisation problem using **Evolution Strategy**:
1. Implement both `(μ, λ)` and `(μ + λ)` selection approaches.
2. Use mutation as the population-varying operator.
3. Implement **discrete** and **intermediate** crossover operators.
4. Evaluate the influence of ES parameters (population size, number of offspring, selection method, crossover type) on performance and computation time.

The optimisation goal is to find parameters `a`, `b`, and `c` of a mathematical function:

f(x) = a * (x² - b * cos(c * π * x))

so that it best fits experimental data `(X, Y)` from the file `model1.txt`.

The **fitness function** is defined as the inverse of the total absolute error between the model output and the target data.

## Features
- **Random initialisation** of individuals.
- **Self-adaptive mutation** following ES methodology (step sizes evolve alongside parameters).
- **Two crossover methods**:
  - **Discrete crossover** – randomly picks parameters from either parent.
  - **Intermediate crossover** – takes the average of parents’ parameters.
- **Two selection strategies**:
  - `(μ, λ)` – selects the best μ offspring from λ generated.
  - `(μ + λ)` – selects the best μ individuals from both parents and offspring.
- **Performance tracking**:
  - Fitness progression over generations.
  - Final model fit plotted against experimental data.

## File Structure
.
├── evolution_strategies.py   # Main Python script
├── model1.txt                # Input data file (X and Y values)
└── README.md                 # This file

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib

Install dependencies via:
pip install numpy pandas matplotlib

## Usage
1. Place `model1.txt` in the same directory as `evolution_strategies.py`.
   - The file should contain two space-separated columns: `X` and `Y`.
2. Run the script:
python evolution_strategies.py
3. During execution, the script will:
   - Print fitness values for each generation.
   - Display two plots:
     - Fitness over time.
     - Final function approximation vs. target data.

## Example Output
After 100 generations, the program prints the best fitness found and shows:
- **Plot 1**: Fitness curve increasing over iterations.
- **Plot 2**: Model output closely following the given data points.

## Notes
- Mutation parameters `τ₁` and `τ₂` follow the standard ES formulas:
  τ₁ = 1 / sqrt(2 * μ)
  τ₂ = 1 / sqrt(2 * sqrt(μ))
- For reproducibility, the script uses a fixed random seed (`123`).
- You can toggle between selection strategies and crossover types by commenting/uncommenting relevant lines in the script.

## References
- Z. Michalewicz, *Genetic Algorithms + Data Structures = Evolution Programs*, Springer.
- I. Rechenberg, *Evolutionsstrategie*, 1973.
- H.-P. Schwefel, *Numerical Optimization of Computer Models*, Wiley, 1981.

