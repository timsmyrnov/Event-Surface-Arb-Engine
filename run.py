import breeden_litzenberger as bl
import black_scholes as bs
import get_heston_inputs as ghi
import pred_market_data as pmd

def run(mkt):
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
    ticker = "^SPX"
    expiry = "2025-12-31"

    mkt = ghi.get_market_inputs_from_yf(ticker, expiry)
    bl_model = run(mkt)
    bs_model = bs.LognormalPDF(ticker, expiry)

    bl_model.compute_pdf()
    bl_model.plot_pdf()
    bs_model.plot_pdf()

    bl_p_raw = bl_model.prob_between(6200, 6400, renormalize=False)
    bl_p_norm = bl_model.prob_between(6200, 6400, renormalize=True)

    bs_p_raw = bs_model.prob_between(6200, 6400, renormalize=False)
    bs_p_norm = bs_model.prob_between(6200, 6400, renormalize=True)

    print("BL Raw prob      =", bl_p_raw)
    print("BL Renormalized prob =", bl_p_norm)
    print("BS Raw prob       =", bs_p_raw)
    print("BS Renormalized prob =", bs_p_norm)

if __name__ == "__main__":
    main()