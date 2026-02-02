# Adversarial Flow Matching with Learned Noise Prior

Extension of adversarial flow matching where the noise distribution (prior)
is learned rather than fixed to N(0, I). This allows the model to potentially
find a better matching between the prior and data manifold.

Key modifications from base adversarial flow:
- LearnedGaussianPrior: Parameterizes prior as N(mu, diag(exp(log_var)))
- KL regularization: Prevents prior from collapsing (KL to standard normal)
- Joint training: Prior parameters optimized alongside generator

The intuition is that optimal transport cost depends on the choice of source
distribution. A learned prior could reduce the total transport distance by
better aligning with the data geometry.
