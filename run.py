import re
import json
import breeden_litzenberger as bl
import black_scholes as bs
import get_heston_inputs as ghi
import pred_market_data as pmd


def run_bl(mkt):
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


def parse_price_range(
    label: str,
    default_min: int = 0,
    default_max: int = 1_000_000_000,
) -> tuple[int, int]:
    s = label.replace("$", "").replace(",", "").strip()

    if "-" in s:
        lo_str, hi_str = s.split("-", 1)
        lo = int(lo_str)
        hi = int(hi_str)
        return lo, hi

    if s.startswith("<"):
        hi = int(s[1:].strip())
        return default_min, hi

    if s.startswith(">"):
        lo = int(s[1:].strip())
        return lo, default_max

    raise ValueError(f"Unrecognized price range label: {label!r}")


def parse_above_price(
    label: str,
    default_max: int = 1_000_000_000,
) -> tuple[int, int]:
    s = label.replace("$", "").replace(",", "").strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"Unrecognized 'above' price label: {label!r}")
    lo = int(m.group(1))
    return lo, default_max


def compute_edge(data, event_links):
    for ticker, outcomes in data.items():
        url = event_links.get(ticker)
        if url is None:
            raise ValueError(f"No event URL found for ticker {ticker}")

        expiry_dt = pmd.get_event_expiry(url)
        expiry = expiry_dt.date().strftime("%Y-%m-%d")

        is_above_event = "above" in url.lower()

        for price, prob in outcomes.items():
            if is_above_event:
                price_range = parse_above_price(price)
            else:
                price_range = parse_price_range(price)

            mkt_data = ghi.get_market_inputs_from_yf(ticker, expiry)
            bl_model = run_bl(mkt_data)
            bl_prob_norm = bl_model.prob_between(
                price_range[0], price_range[1], renormalize=True
            )

            print(f"Ticker: {ticker}, Price Range: {price_range[0], price_range[1]}")
            print(
                f"Polymarket P: {prob}, Actual P: {bl_prob_norm}, "
                f"Expected Edge: {abs(prob - bl_prob_norm)}"
            )


def main():
    event_links = {
        "NVDA": "https://polymarket.com/event/nvda-week-november-21-2025?tid=1763587728930",
        "TSLA": "https://polymarket.com/event/tsla-above-in-november-2025?tid=1763587682040",
        "PLTR": "https://polymarket.com/event/pltr-above-in-november-2025?tid=1763587666119"
    }

    pred_market_data = pmd.parse_events_data(event_links.values())
    compute_edge(pred_market_data, event_links)
    # print(json.dumps(pred_market_data, indent=2))


if __name__ == "__main__":
    main()