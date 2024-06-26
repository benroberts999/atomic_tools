#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import requests
import pandas as pd
from io import BytesIO


def convert_to_float_or_nan(string):
    try:
        if "/" in string:
            a, b = string.split("/")
            return float(a) / int(b)
        else:
            return float(string)
    except ValueError:
        return np.nan


def parse_rrms_data(content):

    rrms_data = pd.read_csv(BytesIO(content)).values

    # if there is preliminary rms data, use that
    for row in rrms_data:
        if not np.isnan(row[6]):
            row[4] = row[6]

    # delete any with Z<1 (has neutron)
    rrms_data = np.delete(rrms_data, rrms_data[:, 0] < 1, axis=0)

    # delete collumns we don't need
    rrms_data = np.delete(rrms_data, [1, 2, 5, 6, 7], axis=1)

    # make slots for I, pi, mu, Q
    nans = np.empty((len(rrms_data), 4))
    nans[:] = np.nan
    return np.append(rrms_data, nans, axis=1)


def parse_moment_data(content):

    data = pd.read_csv(BytesIO(content))
    data = data.drop(
        ["symbol", "halflife", "method", "description", "nsr", "journal", "indc"],
        axis=1,
    )

    # Remove neutron, and excited states
    data = data[data["z"] != 0]
    data = data[data["energy [keV]"] == "0"]
    data = data.drop(["energy [keV]"], axis=1)

    # make room for parsed data (I, pi, mu/Q)
    data = data.values
    nans = np.empty((len(data), 3))
    nans[:] = np.nan
    data = np.append(data, nans, axis=1)

    for row in data:
        jpi = row[2]
        # Some values which are uncertain are in parenthesis
        jpi = jpi.replace("(", "").replace(")", "")

        pi = -1 if jpi[-1] == "-" else +1

        jpi = jpi.replace("+", "").replace("-", "")
        jpi = jpi.replace("if", "").replace(" ", "")

        I = convert_to_float_or_nan(jpi)

        row[4] = I
        row[5] = pi

        # when the sign is uncertain, in the form "(+)1.234" or "(-)1.234"
        if "(+)" in row[3]:
            row[3] = row[3].split("(+)")[1]
        if "(-)" in row[3]:
            row[3] = "-" + row[3].split("(-)")[1]

        # Get rid of uncertainties
        string = row[3].split("(")[0].replace(" ", "")
        mu = convert_to_float_or_nan(string)
        row[6] = mu

    # delete collumns we don't need
    return np.delete(data, [2, 3], axis=1)


#########################################################################

mu_url = "https://www-nds.iaea.org/nuclearmoments/magn_mom_recomm.csv"
rrms_url = "https://www-nds.iaea.org/radii/charge_radii.csv"
Q_url = "https://www-nds.iaea.org/nuclearmoments/elec_mom_recomm.csv"
# only use this one to fill any gaps - don't update
muQ_url = "https://www-nds.iaea.org/nuclearmoments/nuc_mom_compilation.csv"


mu_recieved = requests.get(mu_url)
rrms_recieved = requests.get(rrms_url)
Q_recieved = requests.get(Q_url)
# muQ_recieved = requests.get(muQ_url)

rrms_data = parse_rrms_data(rrms_recieved.content)
mu_data = parse_moment_data(mu_recieved.content)
Q_data = parse_moment_data(Q_recieved.content)


def search(data, t_z, t_a):
    for index, row in enumerate(data):
        if row[0] == t_z and row[1] == t_a:
            return index
    return -1


# insert/update values from mu_data
for [z, a, I, pi, mu] in mu_data:
    index = search(rrms_data, z, a)
    if index == -1:
        rrms_data = np.append(rrms_data, [[z, a, np.nan, I, pi, mu, np.nan]], axis=0)
    else:
        rrms_data[index][3] = I
        rrms_data[index][4] = pi
        rrms_data[index][5] = mu


# insert/update values from Q_data
for [z, a, I, pi, q] in Q_data:
    index = search(rrms_data, z, a)
    if index == -1:
        rrms_data = np.append(rrms_data, [[z, a, np.nan, I, pi, np.nan, q]], axis=0)
    else:
        rrms_data[index][3] = I
        rrms_data[index][4] = pi
        rrms_data[index][6] = q

# re-convert floats to integers..
for index in range(len(rrms_data)):
    rrms_data[index][0] = int(rrms_data[index][0])
    rrms_data[index][1] = int(rrms_data[index][1])
    if not np.isnan(rrms_data[index][4]):
        rrms_data[index][4] = int(rrms_data[index][4])
    # If even nuclei with no moments, assume this means spin=0!
    if (
        np.isnan(rrms_data[index][3])
        and np.isnan(rrms_data[index][4])
        and np.isnan(rrms_data[index][5])
        and np.isnan(rrms_data[index][6])
        and rrms_data[index][1] % 2 == 0
    ):
        rrms_data[index][3] = 0
        rrms_data[index][4] = 1
        rrms_data[index][5] = 0
        rrms_data[index][6] = 0

# sort by Z, then A
ind = np.lexsort((rrms_data[:, 1], rrms_data[:, 0]))
rrms_data = rrms_data[ind]

################################################################################

# Save data as regular csv (missing values => nan)
np.savetxt(
    "nuclear_data.csv",
    rrms_data,
    delimiter=",",
    fmt=["%i", "%i", "%.5f", "%.1f", "%.0f", "%.8f", "%.8f"],
    header="z,a,rrms,I,pi,mu,q",
)


# Save data for ampsci
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
gen_date = now.strftime("%Y-%m-%d %H:%M:%S")

header_text = f"""#pragma once
#include "Physics/NuclearData.hpp"
#include <vector>

// Auto-generated file, using data from https://www-nds.iaea.org/

// Generated using compile_nuclear_data.py 
// Available from: https://github.com/benroberts999/atomic_tools
// Generated date: {gen_date}

// Nuclear rms radii (in fm) taken from:
// https://www-nds.iaea.org/nuclearmoments/magn_mom_recomm.csv

// Nuclear spin, parity, magnetic mu moments (in nuclear magnetons) from:
// https://www-nds.iaea.org/radii/charge_radii.csv

// And,
// Nuclear spin, parity, electric Q moments (in barns) from:
// https://www-nds.iaea.org/nuclearmoments/elec_mom_recomm.csv

namespace Nuclear {{

//! Table of nuclear isotope data: Z, A, r_rms/fm, I, pi, mu/mu_N, Q/b
/*! @details
 - Note: Always check if important, may contain errors.
 - Nuclear Data Service: https://www-nds.iaea.org/.
 - Radii: https://www-nds.iaea.org/radii/ (Marinova, Angeli).
 - Moments: https://www-nds.iaea.org/nuclearmoments/.
*/
static const std::vector<Isotope> NuclearDataTable = {{
    // {{Z, A, rrms, I, pi, mu, Q}}
"""

footer_text = """
} // namespace Nuclear
"""

with open("nuclear_data_table.hpp", "w") as hpp_file:
    hpp_file.write(header_text)
    for index, [z, a, rrms, I, pi, mu, q] in enumerate(rrms_data):
        hpp_file.write(f"    {{{z}, {a}")
        for value in [rrms, I, pi, mu, q]:
            if not np.isnan(value):
                hpp_file.write(f", {value}")
            else:
                hpp_file.write(f", {{}}")
        hpp_file.write(f"}}")
        if index != len(rrms_data) - 1:
            hpp_file.write(",\n")
        else:
            hpp_file.write("};\n")
    hpp_file.write(footer_text)
