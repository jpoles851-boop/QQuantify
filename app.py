from flask import Flask, render_template, request, jsonify
from Analysis import Analysis
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — required for Flask (no display)
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)
analysis = Analysis()


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


@app.route('/')
def home():
    return render_template('Maininterface.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form['symbol'].strip().upper()

    d_to_e             = analysis.get_debt_to_equity(symbol)
    volatility, v_comment = analysis.score_volatility(symbol)
    beta               = analysis.calculate_beta(symbol)
    liquidity, l_comment = analysis.score_liquidity(symbol)

    result = {
        "symbol":        symbol,
        "debt_to_equity": d_to_e,
        "volatility":    volatility,
        "v_comment":     v_comment,
        "beta":          beta,
        "liquidity":     liquidity,
        "l_comment":     l_comment,
    }

    return render_template('anotherone.html', result=result)


@app.route('/graphs/<symbol>')
def graphs(symbol):
    """
    Generate all four charts for the given symbol and return them
    as base64-encoded PNG images in a JSON response.
    The results page fetches this endpoint when the user clicks 'View Graphs'.
    """
    symbol = symbol.strip().upper()
    images = {}

    # ── 1. Price History ──────────────────────────────────────────────
    prices = analysis.get_price_history(symbol)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    if prices is not None:
        ax.plot(prices.index, prices.values, color='#4fc3f7', linewidth=1.5, label=symbol)
        ax.fill_between(prices.index, prices.values, alpha=0.15, color='#4fc3f7')
    ax.set_title(f"{symbol} — Price History", color='white', fontsize=13, pad=12)
    ax.set_xlabel("Date", color='#aaaaaa', fontsize=10)
    ax.set_ylabel("Price (USD)", color='#aaaaaa', fontsize=10)
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#333333')
    ax.grid(True, color='#222222', linewidth=0.7)
    ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white')
    fig.tight_layout()
    images['price_history'] = fig_to_base64(fig)

    # ── 2. Risk Score Bar Chart ───────────────────────────────────────
    vol,  _ = analysis.score_volatility(symbol)
    beta_s, _ = analysis.score_beta(symbol)
    liq,  _ = analysis.score_liquidity(symbol)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    categories = ["Volatility", "Beta", "Liquidity"]
    scores     = [vol, beta_s, liq]
    colors     = ['#ef5350', '#4fc3f7', '#66bb6a']
    bars = ax.bar(categories, scores, color=colors, width=0.45, zorder=3)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{score}/100", ha='center', color='white', fontsize=11, fontweight='bold')
    ax.set_title(f"{symbol} — Risk Score Breakdown", color='white', fontsize=13, pad=12)
    ax.set_ylabel("Risk Score (0–100)", color='#aaaaaa', fontsize=10)
    ax.set_ylim(0, 115)
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#333333')
    ax.grid(axis='y', color='#222222', linewidth=0.7, zorder=0)
    fig.tight_layout()
    images['risk_scores'] = fig_to_base64(fig)

    # ── 3. Beta Scatter ───────────────────────────────────────────────
    stock_prices  = analysis.get_price_history(symbol)
    market_prices = analysis.get_price_history("^GSPC")
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    if stock_prices is not None and market_prices is not None:
        sr = analysis.get_daily_returns(stock_prices)
        mr = analysis.get_daily_returns(market_prices)
        sr, mr = sr.align(mr, join='inner')
        beta_val = analysis.calculate_beta(symbol)
        ax.scatter(mr, sr, alpha=0.35, color='#4fc3f7', s=12, label='Daily returns')
        x = np.linspace(mr.min(), mr.max(), 100)
        ax.plot(x, beta_val * x, color='#ef5350', linewidth=2, label=f'Beta = {beta_val}')
        ax.axhline(0, color='#555555', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--')
    ax.set_title(f"{symbol} — Returns vs S&P 500", color='white', fontsize=13, pad=12)
    ax.set_xlabel("S&P 500 Daily Return", color='#aaaaaa', fontsize=10)
    ax.set_ylabel(f"{symbol} Daily Return", color='#aaaaaa', fontsize=10)
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#333333')
    ax.grid(True, color='#222222', linewidth=0.7)
    ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white')
    fig.tight_layout()
    images['beta_scatter'] = fig_to_base64(fig)

    # ── 4. Rolling Volatility ─────────────────────────────────────────
    daily_returns = analysis.get_daily_returns(prices)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')
    if daily_returns is not None:
        rolling_vol = daily_returns.rolling(30).std() * np.sqrt(252)
        ax.plot(rolling_vol.index, rolling_vol.values, color='#4fc3f7',
                linewidth=1.5, label='30d Rolling Volatility')
        ax.axhline(0.25, color='#ffa726', linestyle='--', linewidth=1, label='Medium risk (25%)')
        ax.axhline(0.40, color='#ef5350', linestyle='--', linewidth=1, label='High risk (40%)')
        ax.fill_between(rolling_vol.index, rolling_vol.values, alpha=0.12, color='#4fc3f7')
    ax.set_title(f"{symbol} — Rolling Volatility", color='white', fontsize=13, pad=12)
    ax.set_xlabel("Date", color='#aaaaaa', fontsize=10)
    ax.set_ylabel("Annualized Volatility", color='#aaaaaa', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#333333')
    ax.grid(True, color='#222222', linewidth=0.7)
    ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white')
    fig.tight_layout()
    images['rolling_vol'] = fig_to_base64(fig)

    return jsonify(images)


if __name__ == '__main__':
    app.run(debug=True, port=3000)