from datetime import datetime
import requests
import json
from urllib.parse import urlparse

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
        "end_date_max": "2025-12-31T00:00:00Z",
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


def _extract_ticker(url: str) -> str | None:
    parsed = urlparse(url)
    path = parsed.path

    parts = [p for p in path.split("/") if p]
    if len(parts) < 2 or parts[0] != "event":
        return None

    slug = parts[1]
    ticker = slug.split("-")[0]
    return ticker.upper()


def _extract_slug(url: str) -> str | None:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2 or parts[0] not in {"event", "market"}:
        return None
    return parts[1]


def _get_markets_for_slug(slug: str):
    url_event = f"{GAMMA_BASE_URL}/events/slug/{slug}"
    try:
        r = requests.get(url_event, timeout=10)
        if r.status_code == 200:
            event = r.json()
            markets = event.get("markets") or []
            return markets
        if r.status_code != 404:
            r.raise_for_status()
    except Exception:
        pass

    url_market = f"{GAMMA_BASE_URL}/markets/slug/{slug}"
    try:
        r2 = requests.get(url_market, timeout=10)
        if r2.status_code == 200:
            market = r2.json()
            return [market]
        if r2.status_code != 404:
            r2.raise_for_status()
    except Exception:
        pass

    return []


def _parse_probabilities(raw):
    vals = parse_outcomes(raw)
    probs = []
    for v in vals:
        try:
            probs.append(float(v))
        except (TypeError, ValueError):
            continue
    return probs


def get_event_expiry(url: str):
    slug = _extract_slug(url)
    if slug is None:
        return None

    resp = requests.get(f"{GAMMA_BASE_URL}/events/slug/{slug}", timeout=10)
    if resp.status_code != 200:
        return None

    event = resp.json()
    markets = event.get("markets") or []
    if not markets:
        return None

    end_str = (
        markets[0].get("endDateIso")
        or markets[0].get("endDate")
        or event.get("endDateIso")
        or event.get("endDate")
    )
    if not end_str:
        return None

    end_str = end_str.replace("Z", "+00:00")
    return datetime.fromisoformat(end_str)


def parse_events_data(links):
    output: dict[str, dict[str, float]] = {}

    for url in links:
        slug = _extract_slug(url)
        ticker = _extract_ticker(url)

        if slug is None or ticker is None:
            continue

        markets = _get_markets_for_slug(slug)
        if not markets:
            continue

        if ticker not in output:
            output[ticker] = {}

        for m in markets:
            outcomes = parse_outcomes(m.get("outcomes"))
            probs = _parse_probabilities(m.get("outcomePrices"))

            label = (
                m.get("groupItemTitle")
                or m.get("groupItemRange")
                or m.get("question")
            )

            yes_p = None
            if "Yes" in outcomes:
                yes_idx = outcomes.index("Yes")
                if yes_idx < len(probs):
                    yes_p = probs[yes_idx]

            if label and yes_p is not None:
                output[ticker][label] = yes_p

    return output


if __name__ == "__main__":
    links = {
        "NVDA": "https://polymarket.com/event/nvda-week-november-21-2025?tid=1763587728930",
        # "TSLA": "https://polymarket.com/event/tsla-above-in-november-2025?tid=1763587682040",
        # "PLTR": "https://polymarket.com/event/pltr-above-in-november-2025?tid=1763587666119",
    }

    data = parse_events_data(links)
    print(json.dumps(data, indent=2))