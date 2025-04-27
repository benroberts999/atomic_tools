#!/usr/bin/env python3

import numpy as np
import os.path

""""""

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
    return (1.0 / 15) * (omega * alpha) ** 5 * qe2**2 / (2 * ji + 1) / seconds


def get_data(energy_file, e1_file="", m1_file="", e2_file=""):
    """Reads in data from energy and matrix element text files.
    Energy data in three collumns: a J En
    (state, J, and energy in cm^-1).
    Matrix element data in 4 columns: a b D err
    (States a and b, matrix element D, and uncertainty err).
    """
    if not os.path.isfile(energy_file):
        return [], [], [], []
    en_data = np.genfromtxt(
        energy_file,
        dtype=[("a", "U10"), ("J", "f"), ("En", "f")],
        comments="#",
    )
    en_data.sort(order="En")
    en_data["En"] /= Eh_cm

    e1_data = (
        np.genfromtxt(
            e1_file,
            dtype=[("a", "U10"), ("b", "U10"), ("d", "f"), ("err", "f")],
            comments="#",
        )
        if os.path.isfile(e1_file)
        else []
    )
    m1_data = (
        np.genfromtxt(
            m1_file,
            dtype=[("a", "U10"), ("b", "U10"), ("d", "f"), ("err", "f")],
            comments="#",
        )
        if os.path.isfile(m1_file)
        else []
    )
    e2_data = (
        np.genfromtxt(
            e2_file,
            dtype=[("a", "U10"), ("b", "U10"), ("d", "f"), ("err", "f")],
            comments="#",
        )
        if os.path.isfile(e2_file)
        else []
    )
    return en_data, e1_data, m1_data, e2_data


def get_matel(data, ai, af):
    """Finds the <f|D|i> OR <i|D|f> matrix element from array"""
    if len(data) == 0:
        return [0.0, 0.0]
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


def calculate_lifetimes(en_data, e1_data, m1_data=[], e2_data=[]):
    """Calculates lifetimes, and prints partial widths, for each state.
    Returns lifetimes as array in same order as energies."""
    en_data = np.array(en_data)  # Ensure en_data is a NumPy array
    print("\nPartial decay rates")
    taus = np.empty((en_data.size, 2))
    for index, [ai, ji, Ei] in enumerate(en_data):
        print(f"\nState: {ai}:")
        print(f"  En = {Ei*Eh_cm:.3f} /cm")
        Gamma = 0.0
        del_Gamma2 = 0.0
        gammas = []
        channel = []
        decays = []
        for [af, _, Ef] in en_data:
            # Can only decay to lower states
            if Ef >= Ei:
                continue
            omega = Ei - Ef
            assert omega > 0

            [e1x, d_e1] = get_matel(e1_data, ai, af)
            [m1, d_m1] = get_matel(m1_data, ai, af)
            [e2, d_e2] = get_matel(e2_data, ai, af)

            symbols = ["E1", "M1", "E2"]
            f_gammas = [gamma_E1, gamma_M1, gamma_E2]
            ds = [e1x, m1, e2]
            errs = [d_e1, d_m1, d_e2]

            for i in range(3):
                d, err, fgamma, s = ds[i], errs[i], f_gammas[i], symbols[i]
                if d == 0.0:
                    continue
                gamma = fgamma(d, omega, ji)
                del_gamma = (
                    err
                    * (fgamma(d + 0.05, omega, ji) - fgamma(d - 0.05, omega, ji))
                    / 0.1
                )

                Gamma += gamma
                del_Gamma2 += del_gamma**2
                decays.append(
                    f"   -> {af} {s}: {gamma:.3e}  +/-  {np.abs(del_gamma):.3e} /s"
                )
                gammas.append(gamma)
                channel.append(f"{af} {s}")

        del_Gamma = np.sqrt(del_Gamma2)

        # decay fractions
        if len(gammas) > 0:
            gammas = gammas / np.sum(gammas)
        # Print each rate, and decay fractions
        for i, decay in enumerate(decays):
            print(f"{decay}  { gammas[i]:.8e}")

        tau = 1.0 / Gamma if Gamma != 0.0 else np.inf
        del_tau = (del_Gamma / Gamma) * tau if Gamma != 0.0 else 0.0
        print(f"  tau = {tau:.8e} s")
        taus[index][0] = tau
        taus[index][1] = del_tau
    return taus


def print_summary(en_data, taus):
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


################################################################################


def main():
    """Example usage"""

    # en_data, e1_data, m1_data, e2_data = get_data(
    #     "ba+_en.txt", "ba+_E1.txt", "ba+_M1.txt", "ba+_E2.txt"
    # )

    en_data, e1_data, m1_data, e2_data = get_data("ba+_en.txt", "ba+_E1.txt")

    # en_data, e1_data, m1_data, e2_data = get_data(
    #     "ra+_en.txt", "ra+_E1_sr.txt", "ra+_M1_sr.txt", "ra+_E2_sr.txt"
    # )

    taus = calculate_lifetimes(en_data, e1_data, m1_data, e2_data)

    print("\nLifetimes Summary:")
    print_summary(en_data, taus)


if __name__ == "__main__":
    main()
