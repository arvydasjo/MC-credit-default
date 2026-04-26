import numpy as np
import matplotlib.pyplot as plt

N_CLIENTS = 50
N_MAX = 20000

pd = np.random.beta(1, 5, N_CLIENTS)
lgd = np.random.uniform(0.1, 0.9, N_CLIENTS)
ead = np.random.uniform(500, 50000, N_CLIENTS)

losses = []

running_el = []
running_var = []
running_cvar = []

for i in range(1, N_MAX + 1):

    defaults = np.random.binomial(1, pd)
    loss = np.sum(defaults * lgd * ead)
    losses.append(loss)

    arr = np.array(losses)

    # EL
    el = np.mean(arr)
    running_el.append(el)

    # VaR 95
    var_95 = np.percentile(arr, 95)
    running_var.append(var_95)

    # CVaR 95
    cvar_95 = np.mean(arr[arr >= var_95])
    running_cvar.append(cvar_95)

losses = np.array(losses)

# GRAPH
plt.figure(figsize=(10,6))

plt.plot(running_el, label="Expected Loss", color="blue")
plt.plot(running_var, label="VaR 95%", color="orange")
plt.plot(running_cvar, label="CVaR 95%", color="red")

# error band (EL ± std error)
std_err = np.std(losses) / np.sqrt(np.arange(1, N_MAX + 1))

plt.fill_between(
    range(N_MAX),
    running_el - std_err,
    running_el + std_err,
    color="blue",
    alpha=0.15,
    label="EL confidence band"
)

plt.title("Monte Carlo Convergence: EL / VaR / CVaR")
plt.xlabel("Simulations")
plt.ylabel("Loss")
plt.xscale("log")
plt.legend()
plt.grid()



plt.savefig("convergence.png", dpi=300, bbox_inches="tight")
plt.show()