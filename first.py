import numpy as np
from scipy.stats import ncx2, chi2
from scipy.misc import derivative
import matplotlib.pyplot as plt


def approx_pdf(x, k, lam):
    p = np.zeros_like(x, dtype=np.float64)
    f = 1
    for i in range(10):
        p += np.exp(-lam/2) * (lam/2)**i * chi2.pdf(x, k + 2*i) / f
        f *= (i + 1)
    return p

# df == k on wikipedia
# nc == lambda on wikipedia



x = np.linspace(0, 8, 400)

linestyle = '-'
for df in [2, 4]:
    for nc in [1, 2, 3]:
        plt.plot(x, ncx2.pdf(x, df, nc), linestyle=linestyle, label= 'k = {df}, lambda = {nc}')
        plt.plot(x, approx_pdf(x, df, nc), 'k', alpha=0.1, linewidth=6)
    linestyle = '--'

plt.title("Noncentral chi-square distribution\nProbability density function")
plt.xlabel('x')
plt.legend(shadow=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()