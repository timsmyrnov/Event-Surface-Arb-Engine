import requests
import json

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

def get_markets(limit: int = 50, offset: int = 0):
    url = f"{GAMMA_BASE_URL}/markets"
    params = {
        "limit": limit,
        "offset": offset,
        "closed": "false",
        "order": "endDate",
        "ascending": "true",
        "end_date_min": "2025-11-19T00:00:00Z",
        "end_date_max": "2025-11-20T00:00:00Z"
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def parse_outcomes(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(o) for o in raw]

    try:
        decoded = json.loads(raw)
        if isinstance(decoded, list):
            return [str(o) for o in decoded]
    except Exception:
        pass

    return [str(raw)]

def print_simple_market_info(market: dict):
    outcomes = ", ".join(parse_outcomes(market.get("outcomes"))) or "N/A"

    print("\n=== Market ===")
    print("ID:        ", market.get("id"))
    print("Question:  ", market.get("question"))
    print("Slug:      ", market.get("slug"))
    print("Category:  ", market.get("category"))
    print("Outcomes:  ", outcomes)
    print("Active:    ", market.get("active"))
    print("Closed:    ", market.get("closed"))
    print("End Date:  ", market.get("endDate"))
    print("==============")

if __name__ == "__main__":
    markets = get_markets(limit=20, offset=0)

    if not markets:
        print("No markets returned.")
        raise SystemExit(0)

    print(f"Got {len(markets)} markets.")

    for m in markets[:20]:
        print_simple_market_info(m)