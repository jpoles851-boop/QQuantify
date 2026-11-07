# Visualizer.py

import matplotlib.pyplot as plt
from Analysis import Analysis

import matplotlib.pyplot as plt
import numpy as np
from Analysis import Analysis


class Visualizer:
    """
    Generates matplotlib figures for one or more stock tickers.
    Works alongside the Analysis class to visualize risk data.
    """

    def __init__(self):
        self.analysis = Analysis()


    def plot_price_history(self, symbols: list, period: str = "1y"):
        """
        Line chart of closing prices for one or more stocks over time.
        All stocks are plotted on the same chart for easy comparison.
        """
        plt.figure(figsize=(12, 6))
        for symbol in symbols:
            prices = self.analysis.get_price_history(symbol, period)
            if prices is not None:
                plt.plot(prices.index, prices.values, label=symbol)
        plt.title("Stock Price History")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_risk_scores(self, symbols: list):
        """
        Grouped bar chart comparing volatility, beta, and liquidity
        risk scores across multiple stocks side by side.
        """
        vol_scores  = []
        beta_scores = []
        liq_scores  = []

        for symbol in symbols:
            vol,  _ = self.analysis.score_volatility(symbol)
            beta, _ = self.analysis.score_beta(symbol)
            liq,  _ = self.analysis.score_liquidity(symbol)
            vol_scores.append(vol)
            beta_scores.append(beta)
            liq_scores.append(liq)

        x     = range(len(symbols))
        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar([i - width for i in x], vol_scores,  width=width, label='Volatility', color='tomato')
        plt.bar([i          for i in x], beta_scores, width=width, label='Beta',       color='steelblue')
        plt.bar([i + width for i in x], liq_scores,  width=width, label='Liquidity',  color='seagreen')
        plt.title("Risk Score Comparison")
        plt.xlabel("Stock")
        plt.ylabel("Risk Score (0-100)")
        plt.xticks(x, symbols)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()


    def plot_rolling_beta(self, ticker_symbol: str, window: int = 30):
        """
        Line chart showing how beta has changed over time using a
        rolling window. Useful for spotting when a stock became more
        or less sensitive to the market over a given period.
        """
        stock_prices  = self.analysis.get_price_history(ticker_symbol)
        market_prices = self.analysis.get_price_history("^GSPC")

        if stock_prices is None or market_prices is None:
            print("Data unavailable")
            return

        stock_returns  = self.analysis.get_daily_returns(stock_prices)
        market_returns = self.analysis.get_daily_returns(market_prices)
        stock_returns, market_returns = stock_returns.align(market_returns, join='inner')

        rolling_cov  = stock_returns.rolling(window).cov(market_returns)
        rolling_var  = market_returns.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var

        plt.figure(figsize=(12, 5))
        plt.plot(rolling_beta.index, rolling_beta.values, color='steelblue', label=f'Rolling Beta ({window}d)')
        plt.axhline(y=1.0, color='red',  linestyle='--', label='Beta = 1.0 (market)')
        plt.axhline(y=0.0, color='gray', linestyle='--', label='Beta = 0.0')
        plt.title(f"{ticker_symbol} — Rolling Beta vs S&P 500")
        plt.xlabel("Date")
        plt.ylabel("Beta")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_beta_scatter(self, ticker_symbol: str):
        """
        Scatter plot of stock daily returns vs S&P 500 daily returns.
        The slope of the trend line is the beta value.
        A steep slope means the stock amplifies market moves significantly.
        """
        stock_prices  = self.analysis.get_price_history(ticker_symbol)
        market_prices = self.analysis.get_price_history("^GSPC")

        if stock_prices is None or market_prices is None:
            print("Data unavailable")
            return

        stock_returns  = self.analysis.get_daily_returns(stock_prices)
        market_returns = self.analysis.get_daily_returns(market_prices)
        stock_returns, market_returns = stock_returns.align(market_returns, join='inner')

        beta = self.analysis.calculate_beta(ticker_symbol)

        plt.figure(figsize=(8, 6))
        plt.scatter(market_returns, stock_returns, alpha=0.4, color='steelblue', label='Daily returns')

        x = np.linspace(market_returns.min(), market_returns.max(), 100)
        plt.plot(x, beta * x, color='tomato', linewidth=2, label=f'Beta = {beta}')

        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        plt.title(f"{ticker_symbol} — Returns vs S&P 500")
        plt.xlabel("S&P 500 Daily Return")
        plt.ylabel(f"{ticker_symbol} Daily Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_rolling_volatility(self, ticker_symbol: str, window: int = 30):
        """
        Line chart showing how volatility has changed over time.
        Includes threshold lines at 25% (medium risk) and 40% (high risk)
        so you can quickly see when the stock entered a risky period.
        """
        prices        = self.analysis.get_price_history(ticker_symbol)
        daily_returns = self.analysis.get_daily_returns(prices)

        if daily_returns is None:
            print("Data unavailable")
            return

        rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252)

        plt.figure(figsize=(12, 5))
        plt.plot(rolling_vol.index, rolling_vol.values, color='steelblue', label=f'{window}d Rolling Volatility')
        plt.axhline(y=0.25, color='orange', linestyle='--', linewidth=1, label='Medium risk (25%)')
        plt.axhline(y=0.40, color='tomato', linestyle='--', linewidth=1, label='High risk (40%)')
        plt.title(f"{ticker_symbol} — Rolling Volatility Over Time")
        plt.xlabel("Date")
        plt.ylabel("Annualized Volatility")
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_cumulative_returns(self, symbols: list, period: str = "1y"):
        """
        Line chart showing the total growth of $1 invested in each stock.
        Green shading above $1 means profit, red shading below means loss.
        Makes it easy to compare actual performance across multiple stocks.
        """
        plt.figure(figsize=(12, 6))

        for symbol in symbols:
            prices        = self.analysis.get_price_history(symbol, period)
            daily_returns = self.analysis.get_daily_returns(prices)

            if daily_returns is not None:
                cumulative = (1 + daily_returns).cumprod()
                plt.plot(cumulative.index, cumulative.values, label=symbol)

        plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, label='Break even')
        plt.title("Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:.2f}"))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_returns_distribution(self, ticker_symbol: str):
        """
        Histogram of daily returns showing the spread of gains and losses.
        Includes a vertical line for the mean return and the 5th percentile
        (VaR) so you can see the typical best and worst case daily moves.
        """
        prices        = self.analysis.get_price_history(ticker_symbol)
        daily_returns = self.analysis.get_daily_returns(prices)

        if daily_returns is None:
            print("Data unavailable")
            return

        plt.figure(figsize=(10, 5))
        plt.hist(daily_returns, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(
            x=daily_returns.mean(),
            color='seagreen',
            linestyle='--',
            linewidth=1.5,
            label=f"Mean: {daily_returns.mean():.2%}"
        )
        plt.axvline(
            x=daily_returns.quantile(0.05),
            color='tomato',
            linestyle='--',
            linewidth=1.5,
            label=f"VaR (5th %): {daily_returns.quantile(0.05):.2%}"
        )
        plt.title(f"{ticker_symbol} — Daily Returns Distribution")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()


# ── Example usage ──
if __name__ == "__main__":
    viz     = Visualizer()
    symbols = ["AAPL", "TSLA", "MSFT", "NVDA"]

    # Multi-stock graphs
    viz.plot_price_history(symbols, period="1y")
    viz.plot_risk_scores(symbols)
    viz.plot_cumulative_returns(symbols)

    # Single stock graphs
    viz.plot_rolling_beta("TSLA", window=30)
    viz.plot_beta_scatter("TSLA")
    viz.plot_rolling_volatility("TSLA")
    viz.plot_returns_distribution("TSLA")