"""
Main runner script for the 2D Ising Model MCMC project.
Runs all tasks sequentially.
"""
import matplotlib
matplotlib.use('Agg')
import os

os.makedirs('plots', exist_ok=True)

print("Running Task 2 (MCMC exploration)...")
print()
exec(open('task2.py').read())

print()
print("Running Task 3 (Statistical properties)...")
print()
exec(open('Ising_properties.py').read())

print()
print("All tasks complete! Plots saved to plots/ directory.")
