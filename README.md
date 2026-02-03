# Adversarial Flow Matching with Learned Noise Prior

Extension of adversarial flow matching where the noise distribution (prior)
is learned rather than fixed to N(0, I). This allows the model to potentially
find a better matching between the prior and data manifold.

The intuition is that optimal transport cost depends on the choice of source
distribution. A learned prior could reduce the total transport distance by
better aligning with the data geometry.
