import breeden_litzenberger as bl
import get_heston_inputs as ghi
import pred_market_data as pmd


def main(mkt):
    strikes = mkt.strikes
    if len(strikes) < 3:
        raise ValueError("Need at least 3 strikes for BL second derivative")

    dK = float(strikes[1] - strikes[0])

    if mkt.atm_iv is not None:
        v0 = float(mkt.atm_iv) ** 2
    else:
        v0 = 0.04

    return bl.HestonBL(
        S0=mkt.S0,
        tau=mkt.tau,
        r=mkt.r,
        kappa=2.0,
        theta=0.04,
        v0=v0,
        rho=-0.7,
        sigma=0.30,
        lambd=0.0,
        K_min=float(strikes.min()),
        K_max=float(strikes.max()),
        dK=dK,
    )


def main():
    ticker = "GLD"
    expiry = "2025-12-19"

    mkt = ghi.get_market_inputs_from_yf(ticker, expiry)
    model = main(mkt)
    model.compute_pdf()
    model.plot_pdf()

if __name__ == "__main__":
    main()