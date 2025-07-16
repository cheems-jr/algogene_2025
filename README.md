#Algogene 2025 trading competition

My submission to the equities trading strategy for the KSA2025 algogene competition.

Due to the tight risk profile (8% monthly drawdown, 15% max drawdown), I attempted a vol prediction scheme that decided on a % exposure based on constant vol. The target leverage k would be set by k = c / sigma, where c is a constant and sigma is the volatility.

This would allow the portfolio to have a constant risk exposure. Due to unreliability of the organiser's platform, the strategy could not be fully tested and therefore failed to qualify for the trading rounds.

The volatility was calculated using a combined weighted scheme, where the linear term of the volatility was calculated via ?? and the non-linear term was derived via a simple ML algorithm. These would then be compared on historical accuragy and weighted.
