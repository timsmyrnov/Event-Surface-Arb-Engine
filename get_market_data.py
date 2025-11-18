import yfinance as yf
from datetime import datetime, timezone

TICKER = "GLD"
RISK_FREE = 0.05

def get_options_data():
    underlying = yf.Ticker(TICKER)

    expirations = underlying.options
    if not expirations:
        raise RuntimeError(f"No options expirations found for {TICKER}")

    expiry = expirations[0]
    print(f"Using ticker: {TICKER}")
    print(f"Using nearest expiry: {expiry}")

    chain = underlying.option_chain(expiry)
    calls = chain.calls.copy()

    if calls.empty:
        raise RuntimeError("No call options returned for this expiry.")

    calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
    calls.loc[(calls["mid"].isna()) | (calls["mid"] <= 0), "mid"] = calls["lastPrice"]

    calls = calls.dropna(subset=["mid"])
    calls = calls[calls["mid"] > 0]

    calls = calls.sort_values("strike").reset_index(drop=True)

    hist = underlying.history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not retrieve underlying price history.")
    S0 = float(hist["Close"].iloc[-1])

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    tau = (expiry_dt - now).total_seconds() / (365.0 * 24 * 60 * 60)

    K = calls["strike"].to_numpy()
    C = calls["mid"].to_numpy()

    print("\n=== Breeden–Litzenberger inputs ===")
    print(f"S0 (spot):             {S0:.4f}")
    print(f"Risk-free rate r:      {RISK_FREE:.4f}")
    print(f"Expiry:                {expiry} (T ≈ {tau:.6f} years)")
    print(f"Number of strikes:     {len(K)}")

    print("\nStrikes K (sorted):")
    print(K)

    print("\nCall prices C(K, T) (mid):")
    print(C)

    print("\nSample of option chain used (first 10 rows):")
    print(calls[["strike", "bid", "ask", "lastPrice", "mid"]].head(10).to_string(index=False))

if __name__ == "__main__":
    get_options_data()