# add constructor (even if "empty")
# add methods for other finanical analyis
# add method for your overall score
# method to load/calculate benchmarks (?)


# this class will not be the main class. Once the above is done, you'll start on the main class that
# will get the symbol and send it to this class. The main class receives the data from here then sends the data to the "graphics" class
# and then gets the graphics and displays them



import yfinance as yf
import numpy as np

class Analysis:


    def get_debt_to_equity(self, ticker_symbol: str) -> float | None:
        """
        Fetch the debt-to-equity ratio for a given stock ticker.
        Returns the ratio as a decimal or None if unavailable.
        """
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        de = info.get('debtToEquity')
        if de is None:
            return None
        return round(de / 100, 2)

    def get_price_history(self, ticker_symbol: str, period: str = "1y"):
        """
        Fetch daily closing prices for a stock.
        Period can be: 1mo, 3mo, 6mo, 1y, 2y, 5y
        """
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period=period)
        if history.empty:
            return None
        return history['Close']

    def get_daily_returns(self, prices):
        """
        Convert a series of prices into daily percentage returns.
        e.g. price goes from $100 to $102 = 0.02 (2% return)
        """
        if prices is None or len(prices) < 2:
            return None
        return prices.pct_change().dropna()

    def calculate_volatility(self, daily_returns) -> float | None:  # FIX 1: missing self
        """
        Annualized volatility.
        We multiply by sqrt(252) because there are ~252 trading days in a year.
        Returns a decimal e.g. 0.25 = 25% annual volatility
        """
        if daily_returns is None:
            return None
        return round(float(daily_returns.std() * np.sqrt(252)), 4)

    def score_volatility(self, ticker_symbol: str) -> tuple[float, str]:  # FIX 2: added colon, changed symbol param to ticker_symbol
        prices        = self.get_price_history(ticker_symbol)             # FIX 3: fetch prices from ticker
        daily_returns = self.get_daily_returns(prices)                    # FIX 4: get returns from prices
        volatility    = self.calculate_volatility(daily_returns)          # FIX 5: pass returns not undefined 'returns'

        if volatility is None:
            return 50, "Data unavailable — neutral score assigned"
        if volatility < 0.15:
            return 10, f"Volatility of {volatility:.1%} — very stable stock. Low risk."
        elif volatility < 0.25:
            return 30, f"Volatility of {volatility:.1%} — moderate swings. Low-medium risk."
        elif volatility < 0.40:
            return 60, f"Volatility of {volatility:.1%} — significant swings. Medium-high risk."
        else:
            return 90, f"Volatility of {volatility:.1%} — very wild swings. High risk."

    def calculate_beta(self, ticker_symbol: str) -> float | None:
        """
        Beta measures how much the stock moves relative to the S&P 500.
        Beta = 1.0  → moves with the market
        Beta > 1.0  → amplifies market moves (more risky)
        Beta < 1.0  → dampens market moves (more stable)
        Beta < 0    → moves opposite to the market (rare)
        """
        # Get stock prices and market prices (^GSPC = S&P 500)
        stock_prices = self.get_price_history(ticker_symbol)
        market_prices = self.get_price_history("^GSPC")

        if stock_prices is None or market_prices is None:
            return None

        # Convert both to daily returns
        stock_returns = self.get_daily_returns(stock_prices)
        market_returns = self.get_daily_returns(market_prices)

        # Align to the same dates in case of any missing days
        stock_returns, market_returns = stock_returns.align(market_returns, join='inner')

        # Beta = covariance(stock, market) / variance(market)
        covariance = np.cov(stock_returns, market_returns)[0][1]
        variance = np.var(market_returns)

        return round(covariance / variance, 2)

    def score_beta(self, ticker_symbol: str) -> tuple[float, str]:

        """

         Score the beta risk for a stock on a 0-100 scale.
         Higher score means the stock is more sensitive to market moves.
         Returns a tuple of (score, human-readable comment).
        """


        beta = self.calculate_beta(ticker_symbol)

        if beta is None:
            return 50, "Data unavailable — neutral score assigned"
        if beta < 0:
            return 40, f"Beta of {beta} — moves opposite to market. Unusual."
        elif beta < 0.8:
            return 10, f"Beta of {beta} — low market sensitivity. Defensive stock."
        elif beta < 1.2:
            return 30, f"Beta of {beta} — moves in line with the market. Average risk."
        elif beta < 1.8:
            return 60, f"Beta of {beta} — amplifies market moves. Medium-high risk."
        else:
            return 90, f"Beta of {beta} — very high market sensitivity. High risk."

    def calculate_liquidity(self, ticker_symbol: str) -> dict | None:
        """
        Fetch average daily volume and bid-ask spread for a stock.
        """
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        avg_volume = info.get('averageVolume')
        bid = info.get('bid')
        ask = info.get('ask')

        if avg_volume is None:
            return None

        # Calculate bid-ask spread as a percentage of the midpoint price
        # e.g. bid=99.90 ask=100.10 → spread = 0.20 / 100.00 = 0.20%
        if bid and ask and bid > 0:
            spread = round((ask - bid) / ((ask + bid) / 2), 4)
        else:
            spread = None

        return {
            'avg_volume': avg_volume,
            'spread': spread,
        }

    def score_liquidity(self, ticker_symbol: str) -> tuple[float, str]:
        """
        Score liquidity risk based on average volume and bid-ask spread.
        Higher score = harder to exit = more risk.
        """
        data = self.calculate_liquidity(ticker_symbol)

        if data is None:
            return 50, "Data unavailable — neutral score assigned"

        avg_volume = data['avg_volume']
        spread = data['spread']

        # Score volume (higher volume = easier to sell = lower risk)
        if avg_volume > 10_000_000:
            volume_score = 10  # very liquid e.g. AAPL, MSFT
        elif avg_volume > 1_000_000:
            volume_score = 30  # liquid e.g. mid-cap stocks
        elif avg_volume > 100_000:
            volume_score = 60  # moderate liquidity
        else:
            volume_score = 90  # illiquid — hard to exit

        # Score spread (tighter spread = cheaper to trade = lower risk)
        if spread is None:
            spread_score = 50
        elif spread < 0.001:
            spread_score = 10  # very tight spread
        elif spread < 0.005:
            spread_score = 30  # normal spread
        elif spread < 0.01:
            spread_score = 60  # wide spread
        else:
            spread_score = 90  # very wide spread

        # Volume matters more than spread for personal investing
        weights = {
            'volume': 0.70,
            'spread': 0.30,
        }

        composite = round(
            volume_score * weights['volume'] +
            spread_score * weights['spread'],
            1
        )

        spread_str = f"{spread:.3%}" if spread is not None else "unavailable"

        comment = (
            f"Volume: {avg_volume:,} shares/day — "
            f"Spread: {spread_str} — "
            f"Liquidity risk score: {composite}/100"
        )

        return composite, comment







# --- Example usage ---
if __name__ == "__main__":
    this_analysis = Analysis()
    symbols = ["AAPL", "TSLA", "F", "MSFT","IBM", "INDV", "XOM", "AMZN", "BRK-B", "TD", "MRSH","NVDA"]


    for symbol in symbols:
        d_to_e                 = this_analysis.get_debt_to_equity(symbol)
        price_history          = this_analysis.get_price_history(symbol)

        volatility, v_comment  = this_analysis.score_volatility(symbol)
        beta = this_analysis.calculate_beta(symbol)
        liquidity, l_comment = this_analysis.score_liquidity(symbol)
       # volatility = this_analysis.calculate_volatility(volatility)

        print(f"\n{symbol}")
        print(f"  D/E:        {d_to_e}")
        print(f"  Volatility: {volatility}/100 — {v_comment}")
        print(f"  Beta: {beta}")
        print(f"  Liquidity: {liquidity}")

