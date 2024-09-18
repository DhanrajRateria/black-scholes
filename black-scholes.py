import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes formula for a European call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Simulating stock price changes
def simulate_stock_prices(S0, T, r, sigma, steps=100):
    dt = T / steps
    prices = np.zeros(steps)
    prices[0] = S0
    for t in range(1, steps):
        Wt = np.random.normal(0, np.sqrt(dt))
        prices[t] = prices[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * Wt)
    return prices

# Plotting the stock prices and option prices
def plot_simulation(S0, K, T, r, sigma):
    time = np.linspace(0, T, 100)
    prices = simulate_stock_prices(S0, T, r, sigma)

    # Plotting the stock price simulation
    plt.figure(figsize=(10, 6))
    plt.plot(time, prices, label="Simulated Stock Price")
    plt.title("Simulated Stock Price Over Time")
    plt.xlabel("Time (Years)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/simulated_stock_price.png")
    plt.show()

    # Black-Scholes Call Option Price
    call_prices = [black_scholes_call(S0, K, t, r, sigma) for t in time]
    plt.figure(figsize=(10, 6))
    plt.plot(time, call_prices, label="Black-Scholes Call Option Price")
    plt.title("Black-Scholes Call Option Price Over Time")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/call_option_price.png")
    plt.show()

if __name__ == "__main__":
    # Parameters for the Black-Scholes model
    S0 = 100   # Initial stock price
    K = 105    # Strike price
    T = 1.0    # Time to maturity (in years)
    r = 0.05   # Risk-free interest rate
    sigma = 0.2  # Volatility

    # Run the simulation and plotting
    plot_simulation(S0, K, T, r, sigma)
