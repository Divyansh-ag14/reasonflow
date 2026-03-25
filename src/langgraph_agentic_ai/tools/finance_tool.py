from langchain_core.tools import tool
import yfinance as yf


@tool
def get_stock_info(ticker: str) -> str:
    """Fetch current stock price and financial data for a given ticker symbol such as NVDA, AAPL, or MSFT."""
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        return f"Invalid ticker symbol: {ticker}"

    try:
        stock = yf.Ticker(ticker)

        # --- Try the standard .info dict first ---
        info = {}
        try:
            info = stock.info or {}
        except Exception:
            pass

        price = info.get("currentPrice") or info.get("regularMarketPrice")

        # --- Fallback: use fast_info (more reliable in newer yfinance) ---
        if price is None:
            try:
                fi = stock.fast_info
                price = getattr(fi, "last_price", None)
                if price is None and hasattr(fi, "previous_close"):
                    price = fi.previous_close
            except Exception:
                pass

        # --- Fallback: use recent history ---
        if price is None:
            try:
                hist = stock.history(period="5d")
                if not hist.empty:
                    price = round(float(hist["Close"].iloc[-1]), 2)
            except Exception:
                pass

        if price is None:
            return f"Could not retrieve stock data for {ticker}. The ticker may be invalid or the data source is temporarily unavailable."

        # Gather additional info from the .info dict (best-effort)
        name = info.get("longName") or info.get("shortName") or ticker
        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")
        sector = info.get("sector")
        week_52_high = info.get("fiftyTwoWeekHigh")
        week_52_low = info.get("fiftyTwoWeekLow")
        description = (info.get("longBusinessSummary") or "")[:300]

        parts = [f"{name} ({ticker})", f"Current Price: ${price}"]

        if isinstance(market_cap, (int, float)):
            parts.append(f"Market Cap: ${market_cap:,.0f}")
        if pe_ratio is not None:
            parts.append(f"P/E Ratio: {pe_ratio:.2f}")
        if week_52_high is not None and week_52_low is not None:
            parts.append(f"52-Week Range: ${week_52_low} - ${week_52_high}")
        if sector:
            parts.append(f"Sector: {sector}")
        if description:
            parts.append(f"About: {description}")

        return "\n".join(parts)

    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)[:200]}"
