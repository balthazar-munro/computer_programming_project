"""
Task 3: Statistical properties and critical temperature
  3a - Magnetization vs T, power-law fit for beta_tilde
  3b - Susceptibility vs T, power-law fit for gamma
  3c - Cluster size histograms
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from Ising import Ising
from mcmc import mcmc
from Clusters_Ising import find_clusters

os.makedirs('plots', exist_ok=True)

# ================================================================
# Tasks 3a & 3b: Run MCMC once, compute magnetization + susceptibility
# ================================================================
print("=" * 60)
print("Tasks 3a & 3b: Magnetization and Susceptibility vs Temperature")
print("=" * 60)

N = 25
beta_values = np.arange(0.1, 1.05, 0.05)
T_values = 1.0 / beta_values
T_C_theory = 2.0 / np.log(1 + np.sqrt(2))
n_samples = 100

mag_means = []
mag_stderrs = []
chi_values = []
chi_abs_values = []

for beta in beta_values:
    T = 1.0 / beta
    # Scale burn-in: much more at low T to escape random initial config
    if T < 1.5:
        burn_in = 200 * N**2
    elif abs(T - T_C_theory) < 0.5:
        burn_in = 150 * N**2
    else:
        burn_in = 50 * N**2

    wait = 5 * N**2
    if abs(T - T_C_theory) < 0.5:
        wait = 10 * N**2

    ising = Ising(N, seed=42)
    snaps, diag = mcmc(ising, burn_in=burn_in, samples=n_samples,
                       wait=wait, beta=beta, seed=42)

    # Magnetization per spin for each sample
    m_samples = np.array([np.mean(snap) for snap in snaps])
    abs_m_samples = np.abs(m_samples)

    # Task 3a: magnetization
    m_mean = np.mean(abs_m_samples)
    m_stderr = np.std(abs_m_samples) / np.sqrt(n_samples)
    mag_means.append(m_mean)
    mag_stderrs.append(m_stderr)

    # Task 3b: susceptibility
    avg_m = np.mean(m_samples)
    avg_m2 = np.mean(m_samples**2)
    avg_abs_m = np.mean(abs_m_samples)

    chi_std = beta * N**2 * (avg_m2 - avg_m**2)
    chi_abs = beta * N**2 * (avg_m2 - avg_abs_m**2)
    chi_values.append(chi_std)
    chi_abs_values.append(chi_abs)

    print(f"  T={T:.3f} (beta={beta:.2f}): <|m|>={m_mean:.4f} +/- {m_stderr:.4f}"
          f"  chi={chi_std:.1f}  acc={diag['acc_burn']:.3f}")

mag_means = np.array(mag_means)
mag_stderrs = np.array(mag_stderrs)
chi_values = np.array(chi_values)
chi_abs_values = np.array(chi_abs_values)

# ================================================================
# Task 3a plots
# ================================================================
print("\n--- Task 3a: Magnetization plots ---")

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(T_values, mag_means, yerr=mag_stderrs, fmt='o-', capsize=3,
            markersize=5, label=f'N={N}')
ax.axvline(x=T_C_theory, color='red', linestyle='--', alpha=0.7,
           label=r'$T_C = 2/\ln(1+\sqrt{2}) \approx 2.269$')
ax.set_xlabel('Temperature T')
ax.set_ylabel(r'$\langle |m| \rangle$')
ax.set_title('Average Absolute Magnetization vs Temperature')
ax.legend()
plt.tight_layout()
plt.savefig('plots/task3a_magnetization_vs_T.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/task3a_magnetization_vs_T.png")

# Identify T_C from magnetization: steepest drop in the physical region T > 1.5
sort_idx = np.argsort(T_values)
T_sorted = T_values[sort_idx]
m_sorted = mag_means[sort_idx]

# Only look for the transition where T > 1.5 (avoid low-T noise)
phys_mask = T_sorted > 1.5
T_phys = T_sorted[phys_mask]
m_phys = m_sorted[phys_mask]
diffs = np.diff(m_phys) / np.diff(T_phys)
idx_steep = np.argmin(diffs)
T_C_mag = (T_phys[idx_steep] + T_phys[idx_steep + 1]) / 2.0

print(f"  Estimated T_C from magnetization (steepest drop): {T_C_mag:.3f}")
print(f"  Theoretical T_C: {T_C_theory:.3f}")

# Power-law fit for beta_tilde: T < T_C, magnetization significant
T_C_fit = T_C_theory
mask_below = (T_values < T_C_fit - 0.1) & (mag_means > 0.05)
if np.sum(mask_below) >= 3:
    x_log = np.log((T_C_fit - T_values[mask_below]) / T_C_fit)
    y_log = np.log(mag_means[mask_below])
    coeffs = np.polyfit(x_log, y_log, 1)
    beta_tilde = coeffs[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_log, y_log, 'bo', markersize=5, label='Data')
    x_fit = np.linspace(x_log.min(), x_log.max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-',
            label=rf'Fit: $\tilde{{\beta}}$ = {beta_tilde:.3f} (theory: 0.125)')
    ax.set_xlabel(r'$\ln((T_C - T)/T_C)$')
    ax.set_ylabel(r'$\ln(\langle |m| \rangle)$')
    ax.set_title(r'Power-law fit for critical exponent $\tilde{\beta}$')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/task3a_powerlaw_beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Estimated beta_tilde: {beta_tilde:.4f} (theoretical: 0.125)")
    print("Saved plots/task3a_powerlaw_beta.png")
else:
    beta_tilde = float('nan')
    print("  Not enough data below T_C for power-law fit")

# ================================================================
# Task 3b plots
# ================================================================
print("\n--- Task 3b: Susceptibility plots ---")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(T_values, chi_values, 'bo-', markersize=5,
        label=r'$\chi$ (standard: $\langle m^2\rangle - \langle m\rangle^2$)')
ax.plot(T_values, chi_abs_values, 'gs-', markersize=5,
        label=r'$\chi$ (with $|m|$: $\langle m^2\rangle - \langle |m|\rangle^2$)')
ax.axvline(x=T_C_theory, color='red', linestyle='--', alpha=0.7,
           label=rf'$T_C \approx {T_C_theory:.3f}$')
ax.set_xlabel('Temperature T')
ax.set_ylabel(r'Susceptibility $\chi$')
ax.set_title('Magnetic Susceptibility vs Temperature')
ax.legend()
plt.tight_layout()
plt.savefig('plots/task3b_susceptibility_vs_T.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/task3b_susceptibility_vs_T.png")

# Find chi peak only in the physical region T > 1.5
phys_chi_mask = T_values > 1.5
T_phys_chi = T_values[phys_chi_mask]
chi_phys = chi_values[phys_chi_mask]
chi_abs_phys = chi_abs_values[phys_chi_mask]

idx_peak_std = np.argmax(chi_phys)
idx_peak_abs = np.argmax(chi_abs_phys)
T_C_chi = T_phys_chi[idx_peak_std]
T_C_chi_abs = T_phys_chi[idx_peak_abs]
print(f"  T_C from susceptibility peak (standard): {T_C_chi:.3f}")
print(f"  T_C from susceptibility peak (|m| conv): {T_C_chi_abs:.3f}")
print(f"  T_C from magnetization: {T_C_mag:.3f}")
print(f"  Theoretical T_C: {T_C_theory:.3f}")

# Power-law fit for gamma (T > T_C)
mask_above = (T_values > T_C_theory + 0.2) & (chi_values > 0)
if np.sum(mask_above) >= 3:
    x_log = np.log(np.abs(T_values[mask_above] - T_C_theory) / T_C_theory)
    y_log = np.log(chi_values[mask_above])
    coeffs = np.polyfit(x_log, y_log, 1)
    gamma_est = -coeffs[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_log, y_log, 'bo', markersize=5, label='Data')
    x_fit = np.linspace(x_log.min(), x_log.max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-',
            label=rf'Fit: $\gamma$ = {gamma_est:.3f} (theory: 1.75)')
    ax.set_xlabel(r'$\ln(|T - T_C|/T_C)$')
    ax.set_ylabel(r'$\ln(\chi)$')
    ax.set_title(r'Power-law fit for critical exponent $\gamma$')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/task3b_powerlaw_gamma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Estimated gamma: {gamma_est:.4f} (theoretical: 1.75)")
    print("Saved plots/task3b_powerlaw_gamma.png")
else:
    gamma_est = float('nan')
    print("  Not enough data above T_C for gamma fit")

# ================================================================
# Task 3c: Cluster size histograms
# ================================================================
print("\n" + "=" * 60)
print("Task 3c: Cluster size analysis")
print("=" * 60)

cluster_temps = [0.5, 2.5, 5.0]
cluster_Ns = [10, 50, 100, 200]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for t_idx, T in enumerate(cluster_temps):
    beta = 1.0 / T
    ax = axes[t_idx]

    for N_cl in cluster_Ns:
        burn_in = 30 * N_cl**2
        if abs(T - T_C_theory) < 1.0:
            burn_in = 60 * N_cl**2

        ising = Ising(N_cl, seed=42)
        snaps, diag = mcmc(ising, burn_in=burn_in, samples=1,
                           wait=5 * N_cl**2, beta=beta, seed=42)
        sizes = find_clusters(snaps[0])
        print(f"  T={T}, N={N_cl}: {len(sizes)} clusters, "
              f"sizes [{min(sizes)}, {max(sizes)}], sum={sum(sizes)}")

        if max(sizes) > 1:
            bins = np.logspace(0, np.log10(max(sizes) + 1), 20)
        else:
            bins = np.array([0.5, 1.5])
        ax.hist(sizes, bins=bins, alpha=0.6, label=f'N={N_cl}',
                edgecolor='black', linewidth=0.5)

    ax.set_xscale('log')
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Count')
    ax.set_title(f'T = {T}')
    ax.legend(fontsize=8)

plt.suptitle('Cluster Size Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('plots/task3c_cluster_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/task3c_cluster_histograms.png")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY OF KEY RESULTS")
print("=" * 60)
print(f"  Theoretical T_C = 2/ln(1+sqrt(2)) = {T_C_theory:.4f}")
print(f"  Estimated T_C from magnetization:      {T_C_mag:.4f}")
print(f"  Estimated T_C from susceptibility peak: {T_C_chi:.4f} (standard), {T_C_chi_abs:.4f} (|m|)")
print(f"  Estimated beta_tilde: {beta_tilde:.4f}  (theoretical: 0.125)")
print(f"  Estimated gamma:      {gamma_est:.4f}  (theoretical: 1.750)")
print()
print("Convention note: susceptibility computed both ways:")
print("  - Standard: chi = beta * N^2 * (<m^2> - <m>^2)")
print("  - With |m|: chi = beta * N^2 * (<m^2> - <|m|>^2)")
print("The standard convention is used for T_C estimation and gamma fit.")
