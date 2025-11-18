import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Parameters
# ==========================
S0 = 100.0    # initial stock price
tau = 0.5     # time to maturity in years
r = 0.06      # annual risk-free rate

# Heston parameters (risk-neutral)
kappa = 3.0            # mean reversion speed
theta = 0.20**2        # long-term variance
v0    = 0.20**2        # initial variance
rho   = 0.98           # correlation
sigma = 0.20           # vol of vol
lambd = 0.0            # variance risk-premium

# ==========================
# Heston characteristic function
# ==========================
def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    a = kappa * theta
    b = kappa + lambd

    rspi = rho * sigma * phi * 1j

    d = np.sqrt((rspi - b)**2 + (phi*1j + phi**2) * sigma**2)
    g = (b - rspi + d) / (b - rspi - d)

    exp1 = np.exp(r * phi * 1j * tau)
    term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / sigma**2)
    exp2 = np.exp(
        a * tau * (b - rspi + d) / sigma**2
        + v0 * (b - rspi + d) * (1 - np.exp(d * tau)) / ((1 - g * np.exp(d * tau)) * sigma**2)
    )

    return exp1 * term2 * exp2

# ==========================
# Heston call price via rectangular integration (scalar K)
# ==========================
def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    P, umax, N = 0.0, 100.0, 650
    dphi = umax / N  # integration step

    for j in range(1, N):
        # midpoint for rectangular rule
        phi = dphi * (2 * j + 1) / 2.0
        numerator = heston_charfunc(phi - 1j, *args) - K * heston_charfunc(phi, *args)
        denominator = 1j * phi * K**(1j * phi)
        P += dphi * numerator / denominator

    return np.real((S0 - K * np.exp(-r * tau)) / 2.0 + P / np.pi)

# ==========================
# Compute call prices on a strike grid
# ==========================
K_min, K_max, dK = 60.0, 180.0, 1.0
strikes = np.arange(K_min, K_max + dK, dK)

call_prices = np.array([
    heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
    for K in strikes
])

prices = pd.DataFrame({
    "strike": strikes,
    "call": call_prices
})

# ==========================
# Breeden–Litzenberger: f_Q(K, τ) = e^{rτ} * d²C/dK²
# ==========================
curvature = np.full_like(call_prices, np.nan)

# central second difference for interior points
for i in range(1, len(strikes) - 1):
    curvature[i] = (
        call_prices[i + 1]
        - 2.0 * call_prices[i]
        + call_prices[i - 1]
    ) / (dK**2)

# risk-neutral PDF at K (Breeden–Litzenberger)
pdf_BL = np.exp(r * tau) * curvature

prices["pdf_BL"] = pdf_BL

# clip tiny negative values
pdf_BL = np.maximum(pdf_BL, 0.0)

# ==========================
# Plot ONLY the Breeden–Litzenberger PDF
# ==========================
# Use only interior points where second derivative is defined
valid = ~np.isnan(pdf_BL)
K_plot = strikes[valid]
pdf_plot = pdf_BL[valid]

plt.figure(figsize=(10, 6))
plt.plot(K_plot, pdf_plot, label="Breeden–Litzenberger PDF")
plt.fill_between(K_plot, pdf_plot, alpha=0.2)
plt.xlabel("Strike K")
plt.ylabel(r"$f_{\mathbb{Q}}(K, \tau)$")
plt.title(r"Risk-neutral PDF from Breeden–Litzenberger: $f_{\mathbb{Q}}(K, \tau)$")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()