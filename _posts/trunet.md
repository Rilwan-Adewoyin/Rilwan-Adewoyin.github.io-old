---
layout: post
title:  Trunet Poster
date:   2015-03-15 16:40:16
description: Stabilising training/gradients when using Tweedie Distributions
tags: stats
categories: sample-posts
---

## Context
THE TASK: In a previous research work, I proposed a neural network structure which outputs a 3 parameters defining a predictive tweedie distribution. An important sub-task was producing a stable loss/gradient calculation for training a neural network despite the non-continuity and intractibility of the likelihood for tweedie distributions


THE MOTIVATION: Compound poisson-gamma distribution is a member of the family of 3 parameter tweedie distributions TW(\mu, \sigma, \rho ). The \rho parameter can be varied between a value of 1 and 2, which allows the distribution to be a good fit to a larger range of empirically observed observations.
Furthermore, Compound Poisson-Gamma distribution can be interpreted as the aggregated sum of N independent samples from a Gamma distribution, where N is sampled from a Poisson distribution. In the context of rain we can consider 1 day to have be N rain events of which each event was gamma distributed. 
Zero rain can be ccaptured when N=0 and the distribution for each rain event can range across distributional shapes resembling exponential, guassian and beta.

\insert picture of shapes compound-gamma can take


## Quick Background Info

The Tweedie distribution is a statistical distribution that unifies various exponential family distributions, including Poisson, gamma, and Gaussian. It is characterized by a positive power parameter, known as the Tweedie index. This versatile distribution is particularly useful in fields like insurance, finance, and ecology for modeling non-negative data with varying degrees of skewness and dispersion.


## Method: Improving upon continuous approximations
In this blog post, I will only discuss the bare minimum required for this approximation.


The Compound Poisson-Gamma distribution is a discrete distribution, but you want to approximate it with a continuous distribution to make it differentiable and get gradients to train a neural network. One way to achieve this is by using Stirling's approximation. 

Stirling's approximation is a technique used to approximate the factorial function for large values of n. It is given by:

n! ≈ sqrt(2πn) * (n / e)^n

To produce a continuous approximation of the Compound Poisson-Gamma distribution, we first need to know its probability mass function (PMF). The Compound Poisson-Gamma distribution is defined as follows:

Let X be the sum of N independent and identically distributed (i.i.d.) gamma random variables with shape parameter k and rate parameter θ, where N follows a Poisson distribution with parameter λ. The random variable X is said to follow a Compound Poisson-Gamma distribution with parameters k, θ, and λ.

The PMF of the Compound Poisson-Gamma distribution can be expressed as:

P(X = x) = sum_{n=0}^∞ (e^(-λ) * (λ^n) / n!) * (1/Gamma(nk + 1)) * (θ^(nk)) * ((x/θ)^nk) * e^(-x/θ)

Now, let's use Stirling's approximation to make it continuous. We will approximate n! with Stirling's formula in the Poisson part:

P(X = x) ≈ sum_{n=0}^∞ (e^(-λ) * (λ^n) / (sqrt(2πn) * (n / e)^n)) * (1/Gamma(nk + 1)) * (θ^(nk)) * ((x/θ)^nk) * e^(-x/θ)

However, this is still a discrete distribution because of the summation over n. To create a continuous approximation, you can use either of the two methods:

1) Replace the summation with an integral. replacing n with its continous counterpart, ν:

P(X = x) ≈ integral(ν=0 to ∞) (e^(-λ) * (λ^ν) / (sqrt(2πν) * (ν / e)^ν)) * (1/Gamma(νk + 1)) * (θ^(νk)) * ((x/θ)^(νk)) * e^(-x/θ) dν

Then we can calculate a Monte Carlo Sampling estimate with the exponential distribution or truncated normal distribution as our generative distribution.

2) Method 2 hinges on finding which terms $n_j\in \mathbb{N}$ contribute significantly to the overall sum, and evaluating the Sum based on that.

With the following reparameterization,
$$
\begin{aligned}
& \lambda=\frac{\mu^{2-p}}{\Theta(2-p)}, \\
& \alpha=\Theta(p-1) \mu^{p-1}, \\
& P=\frac{2-p}{p-1} .
\end{aligned}
$$

we can express the probability of no rainfall is expressed as:
$$
P(L=0)=\exp \left[-\frac{\mu^{2-p}}{\Theta(2-p)}\right]
$$

While the probability of a rainfall event is expressed as:
$$
\begin{aligned}
& P(L>0)=W(\lambda, \alpha, L, P) \exp \left[\frac{L}{(1-p) \mu^{p-1}}-\frac{\mu^{2-p}}{2-p}\right] \\
& W(\lambda, \alpha, L, P)= \sum_{j=1}{\infty} W_j =\sum_{j=1}^{\infty} \frac{\lambda^j(\alpha L)^{j p} e^{-\lambda}}{j ! \Gamma(j P)}
$$

 To approximate the function $W(\lambda, \alpha, L, P)$, follow the procedure to find the value of $j$ for which $W_j$ reaches its maximum. Treat $j$ as continuous, differentiate $W_j$ with respect to $j$, and set the derivative to zero. The log maximum approximation of $W_j$ is given by:
$$

\begin{aligned}
& \log W_{\max }=\frac{L^{2-p}}{(2-p) \Theta}\left[\log \frac{L^p(p-1)^p}{\Theta^{(1-P)}(2-p)}+(1+P)\right. \\
& \left.-P \log P-(1-P) \log \frac{L^{2-p}}{(2-p) \Theta}\right]-\log (2 \pi)-\frac{1}{2} \\
& \cdot \log P-\log \frac{L^{2-p}}{(2-p) \Theta}
\end{aligned}
$$
where $j_{\max }=L^{2-p} /(2-p) \Theta$.

Therefore by taking a window around j_max we can establish an estimate $\widehat{W}$ where the associated approximation error is bounded as follows:
$$
W(L, \Theta, P)-\widehat{W}(L, \Theta, P) \\
< W_{j_d-1} \frac{1-r_l^{j_d-1}}{1-r_l}+W_{j_u+1} \frac{1}{1-r_u} \\
& r_l=\left.\exp \left(\frac{\partial W_j}{\partial j}\right)\right|_j=j_d-1, \\
& r_u=\left.\exp \left(\frac{\partial W_j}{\partial j}\right)\right|_j=j_u+1 .
\end{aligned}
$$
$$





This is now an approximation of the Compound Poisson-Gamma likelihood, which can be used to compute gradients for training a neural network. 

[Further reading](https://www.kybernetika.cz/content/2011/1/15/paper.pdf) on this distribution.

Method2 follows the work of [PK Dunn](https://research.usq.edu.au/download/8969f2b8cd529381e89bd7586e184439777aabdbaabe5c4ed7f7397dcb6bda50/226888/Dunn_Smyth_Stats_and_Comp_v15n4.pdf) 


## Appendix

### Calculating j_max
To handle the sum to infinity when finding $j_{\max}$, the goal is to find the value of $j$ for which the terms in the sum are the most significant. For this purpose, the log maximum approximation of $W_j$ is considered. The steps to find $j_{\max}$ are as follows:

1. Start with the expression for $W_j$ as a part of the sum:
$$
W_j = \frac{\lambda^j(\alpha L)^{j p} e^{-\lambda}}{j ! \Gamma(j P)}
$$

2. Write down the logarithm of $W_j$:
$$
\log W_j = j \log \lambda + j p \log (\alpha L) - \lambda - \log (j !) - \log \Gamma(j P)
$$

3. Use Stirling's approximation for the Gamma function to simplify the logarithmic expression:
$$
\log \Gamma(1+j) \approx (1+j) \log (1+j)-(1+j) + \frac{1}{2} \log \left(\frac{2 \pi}{1+j}\right)
$$

4. Now, differentiate the logarithmic expression of $W_j$ with respect to $j$ and ignore the $1 / j$ term for large $j$:
$$
\frac{\partial \log W_j}{\partial j} \approx \log \lambda + p \log (\alpha L) - \log j - P \log (P j)
$$

5. Set the derivative to zero and solve for $j$:
$$
0 = \log \lambda + p \log (\alpha L) - \log j - P \log (P j)
$$

6. From the equation above, find $j_{\max}$:
$$
j_{\max} = \frac{L^{2-p}}{(2-p) \Theta}
$$

By finding the value of $j$ for which $W_j$ reaches its maximum, the sum to infinity is handled by focusing on the most significant terms. This approach allows for a more efficient and accurate approximation of the function $W(\lambda, \alpha, L, P)$, as the terms in the sum decay faster than geometrically on either side of $j_{\max}$.

