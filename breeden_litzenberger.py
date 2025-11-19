import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class HestonBL:
    def __init__(
        self,
        S0,
        tau,
        r,
        kappa,
        theta,
        v0,
        rho,
        sigma,
        lambd,
        K_min=60.0,
        K_max=180.0,
        dK=1.0,
        umax=100.0,
        N=650,
    ):
        # core params
        self.S0 = S0
        self.tau = tau
        self.r = r

        self.kappa = kappa
        self.theta = theta
        self.v0 = v0
        self.rho = rho
        self.sigma = sigma
        self.lambd = lambd

        # grid / integration params
        self.K_min = K_min
        self.K_max = K_max
        self.dK = dK
        self.umax = umax
        self.N = N

        # containers
        self.strikes = None
        self.call_prices = None
        self.pdf_BL = None

    # ==========================
    # Heston characteristic function
    # ==========================
    def heston_charfunc(self, phi):
        S0 = self.S0
        v0 = self.v0
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        lambd = self.lambd
        tau = self.tau
        r = self.r

        a = kappa * theta
        b = kappa + lambd

        rspi = rho * sigma * phi * 1j

        d = np.sqrt((rspi - b) ** 2 + (phi * 1j + phi**2) * sigma**2)
        g = (b - rspi + d) / (b - rspi - d)

        exp1 = np.exp(r * phi * 1j * tau)
        term2 = S0 ** (phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g)) ** (-2 * a / sigma**2)
        exp2 = np.exp(
            a * tau * (b - rspi + d) / sigma**2
            + v0 * (b - rspi + d) * (1 - np.exp(d * tau))
            / ((1 - g * np.exp(d * tau)) * sigma**2)
        )

        return exp1 * term2 * exp2

    # ==========================
    # Heston call price via rectangular integration (scalar K)
    # ==========================
    def heston_price_rec(self, K):
        P = 0.0
        dphi = self.umax / self.N

        for j in range(1, self.N):
            # midpoint for rectangular rule
            phi = dphi * (2 * j + 1) / 2.0
            numerator = self.heston_charfunc(phi - 1j) - K * self.heston_charfunc(phi)
            denominator = 1j * phi * K ** (1j * phi)
            P += dphi * numerator / denominator

        return np.real(
            (self.S0 - K * np.exp(-self.r * self.tau)) / 2.0 + P / np.pi
        )

    # ==========================
    # Compute call prices on a strike grid
    # ==========================
    def compute_call_curve(self):
        strikes = np.arange(self.K_min, self.K_max + self.dK, self.dK)
        call_prices = np.array([self.heston_price_rec(K) for K in strikes])

        self.strikes = strikes
        self.call_prices = call_prices

        return pd.DataFrame({"strike": strikes, "call": call_prices})

    # ==========================
    # Breeden–Litzenberger: f_Q(K, τ) = e^{rτ} * d²C/dK²
    # ==========================
    def compute_pdf(self):
        if self.call_prices is None or self.strikes is None:
            self.compute_call_curve()

        strikes = self.strikes
        call_prices = self.call_prices
        dK = self.dK

        curvature = np.full_like(call_prices, np.nan)

        # central second difference for interior points
        for i in range(1, len(strikes) - 1):
            curvature[i] = (
                call_prices[i + 1]
                - 2.0 * call_prices[i]
                + call_prices[i - 1]
            ) / (dK**2)

        pdf_BL = np.exp(self.r * self.tau) * curvature
        pdf_BL = np.maximum(pdf_BL, 0.0)  # clip tiny negatives

        self.pdf_BL = pdf_BL

        return pd.DataFrame(
            {
                "strike": strikes,
                "call": call_prices,
                "pdf_BL": pdf_BL,
            }
        )
    
    def prob_between(self, K_low, K_high, renormalize=False):
        if self.pdf_BL is None or self.strikes is None:
            self.compute_pdf()

        strikes = self.strikes
        pdf_BL = self.pdf_BL

        mask = (~np.isnan(pdf_BL)) & (strikes >= K_low) & (strikes <= K_high)

        if not np.any(mask):
            return 0.0

        num = np.trapz(pdf_BL[mask], strikes[mask])

        if not renormalize:
            return num

        mask_all = ~np.isnan(pdf_BL)
        denom = np.trapz(pdf_BL[mask_all], strikes[mask_all])
        if denom <= 0:
            return num
        return num / denom

    # ==========================
    # Plot the Breeden–Litzenberger PDF
    # ==========================
    def plot_pdf(self):
        if self.pdf_BL is None:
            self.compute_pdf()

        pdf_BL = self.pdf_BL
        strikes = self.strikes

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

if __name__ == "__main__":
    model = HestonBL(
        S0=100.0,       # can get from YF
        tau=0.5,        # can get from YF
        r=0.06,         # can get from YF
        kappa=2.0,
        theta=0.04,
        v0=0.04,        # can get from YF
        rho=-0.7,
        sigma=0.30,
        lambd=0.0,
        K_min=60.0,     # can get from YF
        K_max=180.0,    # can get from YF
        dK=1.0,         # can get from YF
    )

    df = model.compute_pdf()
    model.plot_pdf()