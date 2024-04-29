#!/usr/bin/env python3

import numpy as np

# Define physical constants:

alpha = 1.0 / 137.035999084
"""Alpha: fine structure constant"""

Eh_cm = 219474.63136290
"""Hartree, in cm^-1 (convert from cm to atomic units)"""

seconds = 2.4188843266e-17
"""hbar/E_Hartree: atomic unit of time, in seconds"""

# Define partial transition rates


def gamma_E1(de1, omega, ji):
    """Partial decay rate due to E1 transtion. inputs in atomic units. Output in s^1. ji is J of initial (upper) state."""
    return (4.0 / 3) * (omega * alpha) ** 3 * de1**2 / (2 * ji + 1) / seconds


def gamma_M1(dm1, omega, ji):
    """Partial decay rate due to m1 transtion. dm1 is in units of mu_B"""
    # note: dm1 is in units of mu_B, so extra alpha^2/4
    return gamma_E1(dm1 * alpha / 2, omega, ji)


def gamma_E2(qe2, omega, ji):
    """Partial decay rate due to E2 transtion. dm1 is in units of mu_B"""
    (1.0 / 15) * (omega * alpha) ** 5 * qe2**2 / (2 * ji + 1) / seconds


en_data = np.genfromtxt(
    "en-example.txt", dtype=[("a", "U10"), ("J", "f"), ("En", "f")], comments="#"
)
e1_data = np.genfromtxt(
    "e1-example.txt",
    dtype=[("a", "U10"), ("b", "U10"), ("d", "f"), ("err", "f")],
    comments="#",
)

print(en_data)

en_data.sort(order="En")
en_data["En"] /= Eh_cm

print(en_data)


def get_matel(data, ai, af):
    # There's got to be a better way to do this....
    t1 = data[data["a"] == ai]
    t2 = t1[t1["b"] == af]
    t3 = data[data["b"] == ai]
    t4 = t3[t3["a"] == af]
    if t2.size == 1:
        return [t2[0]["d"], t2[0]["err"]]
    if t4.size == 1:
        return [t4[0]["d"], t4[0]["err"]]
    return [0.0, 0.0]


print("\nPartial decay rates")
taus = np.empty((en_data.size, 2))
for index, [ai, ji, Ei] in enumerate(en_data):
    print(f"\nState: {ai}:")
    print(f"  En = {Ei*Eh_cm:.3f} /cm")
    Gamma = 0.0
    del_Gamma2 = 0.0
    gammas = []
    for [af, _, Ef] in en_data:
        # Can only decay to lower states
        if Ef >= Ei:
            continue
        omega = Ei - Ef
        assert omega > 0
        [dE1, del_dE1] = get_matel(e1_data, ai, af)
        if dE1 != 0.0:
            gamma = gamma_E1(dE1, omega, ji)

            del_gamma = (
                del_dE1
                * (gamma_E1(dE1 + 0.05, omega, ji) - gamma_E1(dE1 - 0.05, omega, ji))
                / 0.1
            )

            Gamma += gamma
            del_Gamma2 += del_gamma**2
            print(f"   -> {af} E1: {gamma:.3e}  +/-  {np.abs(del_gamma):.3e} /s")
            gammas.append(gamma)

    del_Gamma = np.sqrt(del_Gamma2)

    # decay fractions
    if len(gammas) > 0:
        gammas = gammas / np.sum(gammas)
        print(f"   :: {gammas}")

    tau = 1.0 / Gamma if Gamma != 0.0 else np.inf
    del_tau = (del_Gamma / Gamma) * tau if Gamma != 0.0 else 0.0
    print(f"  tau = {tau:.5e} s")
    taus[index][0] = tau
    taus[index][1] = del_tau

print()
for index, [ai, _, _] in enumerate(en_data):
    [t, dt] = taus[index]
    if t == np.inf:
        continue

    # print differently if we have errors or not
    if dt == 0.0:
        print(f"{ai} : {t:.4e} +/- {dt:.4e} s")
    else:
        exp = np.floor(np.log10(t))
        pp = np.power(10.0, -exp)

        n_digits = 1 - int(np.floor(np.log10(dt * pp)))
        pp2 = 10**n_digits
        TYPE = "f"
        print(
            f"{ai} : {t:.4e} +/- {dt:.4e} s  = {t*pp:{n_digits+2}.{n_digits}{TYPE}} ({dt*pp*pp2:{2}.{0}{TYPE}}) x10^{int(exp)} s"
        )
