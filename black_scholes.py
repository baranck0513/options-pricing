from __future__ import annotations

from math import exp, log, sqrt
from typing import Literal, Tuple

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


# compute the black scholes d1 term
def _d1(spot_price: float, strike_price: float, time_to_maturity: float, risk_free_rate: float, volatility: float, dividend_yield: float = 0.0) -> float:
    return (log(spot_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity) / (volatility * sqrt(time_to_maturity))


# compute the black scholes d2 term
def _d2(d1: float, volatility: float, time_to_maturity: float) -> float:
    return d1 - volatility * sqrt(time_to_maturity)


def black_scholes(spot_price: float, strike_price: float, time_to_maturity: float, risk_free_rate: float, volatility: float, option_type: OptionType, dividend_yield: float = 0.0) -> float:
    """
    price a european call or put option using the black scholes formula

    spot_price --- current spot price of the underlying asset
    strike_price --- option strike price
    time_to_maturity --- time to maturity in years
    risk_free_rate --- continuously compounded risk free interest rate
    volatility --- annualised volatility of the underlying asset
    """

    if spot_price <= 0 or strike_price <= 0:
        raise ValueError("spot_price and strike_price must be positive")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")
    if option_type not in {"call", "put"}: # option payoff type must be either call or put
        raise ValueError("option_type must be 'call' or 'put'")

    d1 = _d1(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, dividend_yield=dividend_yield)
    adjusted_spot_price = spot_price * exp(-dividend_yield * time_to_maturity) # adjust the spot price for continuous dividends

    d2 = _d2(d1, volatility, time_to_maturity)

    # N(d1) and N(d2) are the risk neutral probabilities used in pricing
    discounted_strike = strike_price * exp(-risk_free_rate * time_to_maturity)

    if option_type == "call":
        return adjusted_spot_price * norm.cdf(d1) - discounted_strike * norm.cdf(d2)

    return discounted_strike * norm.cdf(-d2) - adjusted_spot_price * norm.cdf(-d1) # the theoretical black scholes price of the option


def greeks(spot_price: float, strike_price: float, time_to_maturity: float, risk_free_rate: float, volatility: float, dividend_yield: float = 0.0, option_type: OptionType = "call") -> Tuple[float, float, float, float]:
    """
    compute analytical black scholes greeks

    spot_price --- current spot price of the underlying asset
    strike_price --- option strike price
    time_to_maturity --- time to maturity in years
    risk_free_rate --- continuously compounded risk free interest rate
    volatility --- annualised volatility of the underlying asset
    dividend_yield --- continuously compounded dividend yield
    option_type --- 'call' or 'put'
    """

    if spot_price <= 0 or strike_price <= 0:
        raise ValueError("spot_price and strike_price must be positive.")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    d1 = _d1(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, dividend_yield=dividend_yield)
    d2 = _d2(d1, volatility, time_to_maturity)

    # standard normal density, which appears in the sensitivity formulas
    pdf_d1 = norm.pdf(d1)
    sqrt_time_to_maturity = sqrt(time_to_maturity)
    discount_factor = exp(-risk_free_rate * time_to_maturity)
    dividend_discount = exp(-dividend_yield * time_to_maturity)

    # delta --- sensitivity of option value to a small change in spot price
    if option_type == "call":
        delta = dividend_discount * norm.cdf(d1)
    else:
        delta = dividend_discount * (norm.cdf(d1) - 1.0)

    # gamma --- curvature of option value with respect to spot price (same for call and put)
    gamma = dividend_discount * pdf_d1 / (spot_price * volatility * sqrt_time_to_maturity)

    # vega --- sensitivity to volatility, scaled per 1.0 change in volatility (same for call and put)
    vega = spot_price * dividend_discount * pdf_d1 * sqrt_time_to_maturity

    # theta --- sensitivity to the passage of time, reported per year
    common_theta = -(spot_price * dividend_discount * pdf_d1 * volatility) / (2.0 * sqrt_time_to_maturity)
    if option_type == "call":
        theta = common_theta - risk_free_rate * strike_price * discount_factor * norm.cdf(d2) + dividend_yield * spot_price * dividend_discount * norm.cdf(d1)
    else:
        theta = common_theta + risk_free_rate * strike_price * discount_factor * norm.cdf(-d2) - dividend_yield * spot_price * dividend_discount * norm.cdf(-d1)

    return delta, gamma, theta, vega


def verify_put_call_parity(spot_price: float, strike_price: float, time_to_maturity: float, risk_free_rate: float, volatility: float, dividend_yield: float = 0.0, tolerance: float = 1e-10) -> bool:
    """
    verify the european put call parity relationship

    spot_price --- current spot price of the underlying asset
    strike_price --- option strike price
    time_to_maturity --- time to maturity in years
    risk_free_rate --- continuously compounded risk free interest rate
    volatility --- annualised volatility of the underlying asset
    dividend_yield --- continuously compounded dividend yield
    tolerance --- absolute tolerance for parity checking
    """

    call_price = black_scholes(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, "call", dividend_yield=dividend_yield)
    put_price = black_scholes(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, "put", dividend_yield=dividend_yield)
    parity_left = call_price - put_price
    parity_right = spot_price * exp(-dividend_yield * time_to_maturity) - strike_price * exp(-risk_free_rate * time_to_maturity)
    return bool(np.isclose(parity_left, parity_right, atol=tolerance, rtol=0.0)) # true if put call parity holds within the tolerance, otherwise false


if __name__ == "__main__":
    spot = 100.0
    strike = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    print("q = 0.0")
    call = black_scholes(spot, strike, T, r, sigma, "call", 0.0)
    put = black_scholes(spot, strike, T, r, sigma, "put", 0.0)
    delta, gamma, theta, vega = greeks(spot, strike, T, r, sigma, 0.0, "call")
    parity = verify_put_call_parity(spot, strike, T, r, sigma, 0.0)

    print(f"Call : {call:.6f}")
    print(f"Put  : {put:.6f}")
    print(f"Delta: {delta:.6f}")
    print(f"Gamma: {gamma:.6f}")
    print(f"Theta: {theta:.4f}")
    print(f"Vega : {vega:.4f}")
    print(f"Parity holds: {parity}\n")

    print("q = 0.03")
    call2 = black_scholes(spot, strike, T, r, sigma, "call", 0.03)
    put2 = black_scholes(spot, strike, T, r, sigma, "put", 0.03)
    delta2, gamma2, theta2, vega2 = greeks(spot, strike, T, r, sigma, 0.03, "call")
    parity2 = verify_put_call_parity(spot, strike, T, r, sigma, 0.03)

    print(f"Call : {call2:.6f}")
    print(f"Put  : {put2:.6f}")
    print(f"Delta: {delta2:.6f}")
    print(f"Gamma: {gamma2:.6f}")
    print(f"Theta: {theta2:.4f}")
    print(f"Vega : {vega2:.4f}")
    print(f"Parity holds: {parity2}")
