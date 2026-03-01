"""
Task 2: Explore MCMC parameters
  2a - Visualize configurations at different T and N
  2b - Energy traces, acceptance rates, cost during sampling
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from Ising import Ising
from mcmc import mcmc

os.makedirs('plots', exist_ok=True)

# ----------------------------------------------------------------
# Task 2a: 3x3 grid of configurations
# ----------------------------------------------------------------
print("=" * 60)
print("Task 2a: Visualizing configurations")
print("=" * 60)

temperatures = [0.5, 2.5, 5.0]
betas = [1.0 / T for T in temperatures]
lattice_sizes = [10, 25, 100]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row, (T, beta) in enumerate(zip(temperatures, betas)):
    for col, N in enumerate(lattice_sizes):
        if abs(T - 2.269) < 1.0:
            burn_in = 40 * N**2
        elif T < 2.0:
            burn_in = 30 * N**2
        else:
            burn_in = 10 * N**2

        ising = Ising(N, seed=42)
        snaps, diag = mcmc(ising, burn_in=burn_in, samples=1,
                           wait=5 * N**2, beta=beta, seed=42)

        ax = axes[row, col]
        ax.imshow(snaps[0], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f'T={T}, N={N}', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  T={T}, N={N}: done (acc_burn={diag['acc_burn']:.3f})")

plt.suptitle('Spin Configurations at Different T and N', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('plots/task2a_configurations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/task2a_configurations.png\n")

# ----------------------------------------------------------------
# Task 2b: Energy trace, acceptance rates, cost during sampling
# ----------------------------------------------------------------
print("=" * 60)
print("Task 2b: Energy traces and acceptance rates")
print("=" * 60)

# --- Energy trace plot ---
fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for row, (T, beta) in enumerate(zip(temperatures, betas)):
    for col, N in enumerate(lattice_sizes):
        burn_in = 50 * N**2
        ising = Ising(N, seed=42)
        snaps, diag = mcmc(ising, burn_in=burn_in, samples=1,
                           wait=N**2, beta=beta, seed=42)

        trace = diag['energy_trace']
        sweeps = np.arange(len(trace))
        ax = axes[row, col]
        ax.plot(sweeps, trace, linewidth=0.8)
        ax.set_title(f'T={T}, N={N}', fontsize=10)
        ax.set_xlabel('Sweep')
        ax.set_ylabel('Energy H')
        print(f"  Energy trace T={T}, N={N}: {len(trace)} sweeps")

plt.suptitle('Energy During Burn-in (per sweep)', fontsize=14)
plt.tight_layout()
plt.savefig('plots/task2b_energy_trace.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/task2b_energy_trace.png\n")

# --- Acceptance rate vs T ---
print("Computing acceptance rates across temperatures...")
T_range = np.linspace(0.5, 5.0, 15)
fig, ax = plt.subplots(figsize=(8, 5))

for N in lattice_sizes:
    acc_rates = []
    for T in T_range:
        beta = 1.0 / T
        burn_in = 10 * N**2
        ising = Ising(N, seed=42)
        _, diag = mcmc(ising, burn_in=burn_in, samples=1,
                       wait=5 * N**2, beta=beta, seed=42)
        acc_rates.append(diag['acc_meas'])
    ax.plot(T_range, acc_rates, 'o-', markersize=4, label=f'N={N}')

ax.axvline(x=2.269, color='red', linestyle='--', alpha=0.7,
           label=r'$T_C \approx 2.269$')
ax.set_xlabel('Temperature T')
ax.set_ylabel('Acceptance Rate (measurement phase)')
ax.set_title('Acceptance Rate vs Temperature')
ax.legend()
plt.tight_layout()
plt.savefig('plots/task2b_acceptance_rates.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/task2b_acceptance_rates.png\n")

# --- Cost during sampling ---
print("Computing cost during sampling...")
fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for row, (T, beta) in enumerate(zip(temperatures, betas)):
    for col, N in enumerate(lattice_sizes):
        burn_in = 30 * N**2
        n_samp = 50
        wait = 3 * N**2
        ising = Ising(N, seed=42)
        snaps, diag = mcmc(ising, burn_in=burn_in, samples=n_samp,
                           wait=wait, beta=beta, seed=42)

        energies = []
        for snap in snaps:
            tmp = Ising(N)
            tmp.s = snap
            energies.append(tmp.cost())

        ax = axes[row, col]
        ax.plot(energies, linewidth=0.8)
        ax.set_title(f'T={T}, N={N}', fontsize=10)
        ax.set_xlabel('Sample index')
        ax.set_ylabel('Energy H')

plt.suptitle('Energy of Collected Samples (after burn-in)', fontsize=14)
plt.tight_layout()
plt.savefig('plots/task2b_cost_during_sampling.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/task2b_cost_during_sampling.png\n")

print("Task 2 complete!")
