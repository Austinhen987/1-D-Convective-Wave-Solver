import numpy as np
import math
from array import *
import matplotlib.pyplot as plt

# Define domain and mesh parameters

x_min = -200
x_max = 150  # Prescribed domain

dx = 1
dt = 0.1  # Prescribed grid size/timestep size | Suggested sizes will vary based on wave number of initial condition

x = np.arange(x_min, x_max, dx)
t = np.arange(0, 100, dt)  # Create space and time vectors

# Define scheme coefficients

# Third-order optimized multi-step time discretization coefficients
b0 = 2.302558088
b1 = -2.491007599
b2 = 1.574340933
b3 = -0.385891422

# Fourth-order central DRP spatial discretization coefficients
am3 = -0.020843142
am2 = 0.166705904
am1 = -0.770882380
a0 = 0
a1 = -am1
a2 = -am2
a3 = -am3

# Standard fourth-order spatial discretization coefficients
am2_4 = 1/12
am1_4 = -2/3
a0_4 = 0
a1_4 = -am1_4
a2_4 = -am2_4

# Standard sixth-order spatial discretization coefficients
am3_6 = -1/60
am2_6 = 3/20
am1_6 = -3/4
am0_6 = 0
a1_6 = -am1_6
a2_6 = -am2_6
a3_6 = -am3_6

drp_soln = np.zeros((len(t)+1, len(x)))
std_4_soln = np.zeros((len(t)+1, len(x)))
std_6_soln = np.zeros((len(t)+1, len(x)))  # Initialize solution matrices

init_cond = []

for ind in x:
    init_cond.append(math.exp(-math.log(2)*(ind/12)**2) * (1 + math.cos(2.5*ind)))  # Initial Condition (shape of wave)

drp_soln[0] = init_cond
std_4_soln[0] = init_cond
std_6_soln[0] = init_cond


# Group Velocity and wave number calculations

alpha = np.arange(0, math.pi, 0.01)  # Create alpha vector

dn_da_drp = []
dn_da_4 = []
dn_da_6 = []
num_alpha_drp = []
num_alpha_4 = []
num_alpha_6 = []  # Declare all lists
j = 0

# Calculate numerical/real wave number dependence

for val in alpha:

    # This could probably be done better
    num_alpha_drp.append((am3 * math.sin(val * -3) + am2 * math.sin(val * -2) + am1 * math.sin(val * -1) +
                          a1 * math.sin(val * 1) + a2 * math.sin(val * 2) + a3 * math.sin(val * 3)))

    num_alpha_4.append(2 * (a1_4 * math.sin(val * 1) + a2_4 * math.sin(val * 2)))

    num_alpha_6.append(2 * (a1_6 * math.sin(val * 1) + a2_6 * math.sin(val * 2) + a3_6 * math.sin(val * 3)))


# Calculate group velocity relations

for val2 in alpha:

    if j < len(alpha) - 1:

        dn_drp = (num_alpha_drp[j+1] - num_alpha_drp[j]) / 0.01
        dn_4 = (num_alpha_4[j+1] - num_alpha_4[j]) / 0.01
        dn_6 = (num_alpha_6[j+1] - num_alpha_6[j]) / 0.01
        da = (alpha[j+1] - alpha[j]) / 0.01

        dn_da_drp.append(dn_drp / da)
        dn_da_4.append(dn_4 / da)
        dn_da_6.append(dn_6 / da)

    j += 1


# ------------------------
# Solver Loop
# ------------------------

k = 0
for ts in t:
    i = 0
    for value in drp_soln[k]:

        if 3 < i < len(x)-4:  # Ignore boundaries

            if k == 0:  # First timestep

                # DRP
                drp_soln[k+1, i] = drp_soln[k, i] - (dt/dx) * (
                            b0 * (am3 * drp_soln[k, i-3] + am2 * drp_soln[k, i-2] + am1 * drp_soln[k, i-1] +
                                  a1 * drp_soln[k, i+1] + a2 * drp_soln[k, i+2] + a3 * drp_soln[k, i+3]))

                # 6th-Order Standard
                std_6_soln[k+1, i] = std_6_soln[k, i] - (dt/dx) * (
                            b0 * (am3_6 * std_6_soln[k, i-3] + am2_6 * std_6_soln[k, i-2] + am1_6 * std_6_soln[k, i-1] +
                                  a1_6 * std_6_soln[k, i+1] + a2_6 * std_6_soln[k, i+2] + a3_6 * std_6_soln[k, i+3]))

                # 4th-Order Standard
                std_4_soln[k+1, i] = std_4_soln[k, i] - (dt/dx) * (
                            b0 * (am2_4 * std_4_soln[k, i-2] + am1_4 * std_4_soln[k, i-1] + a1_4 * std_4_soln[k, i+1] +
                                  a2_4 * std_4_soln[k, i+2]))

            elif k == 1:  # Second timestep

                drp_soln[k+1, i] = drp_soln[k, i] - (dt/dx) * (
                            b0 * (am3 * drp_soln[k, i-3] + am2 * drp_soln[k, i-2] +
                                  am1 * drp_soln[k, i-1] + a1 * drp_soln[k, i+1] +
                                  a2 * drp_soln[k, i+2] + a3 * drp_soln[k, i+3]) +
                            b1 * (am3 * drp_soln[k-1, i-3] + am2 * drp_soln[k-1, i-2] +
                                  am1 * drp_soln[k-1, i-1] + a1 * drp_soln[k-1, i+1] +
                                  a2 * drp_soln[k-1, i+2] + a3 * drp_soln[k-1, i+3]))

                std_6_soln[k+1, i] = std_6_soln[k, i] - (dt/dx) * (
                            b0 * (am3_6 * std_6_soln[k, i-3] + am2_6 * std_6_soln[k, i-2] +
                                  am1_6 * std_6_soln[k, i-1] + a1_6 * std_6_soln[k, i+1] +
                                  a2_6 * std_6_soln[k, i+2] + a3_6 * std_6_soln[k, i+3]) +
                            b1 * (am3_6 * std_6_soln[k-1, i-3] + am2_6 * std_6_soln[k-1, i-2] +
                                  am1_6 * std_6_soln[k-1, i-1] + a1_6 * std_6_soln[k-1, i+1] +
                                  a2_6 * std_6_soln[k-1, i+2] + a3_6 * std_6_soln[k-1, i+3]))

                std_4_soln[k+1, i] = std_4_soln[k, i] - (dt/dx) * (
                            b0 * (am2_4 * std_4_soln[k, i-2] +
                                  am1_4 * std_4_soln[k, i-1] + a1_4 * std_4_soln[k, i+1] +
                                  a2_4 * std_4_soln[k, i+2]) +
                            b1 * (am2_4 * std_4_soln[k-1, i-2] +
                                  am1_4 * std_4_soln[k-1, i-1] + a1_4 * std_4_soln[k-1, i+1] +
                                  a2_4 * std_4_soln[k-1, i+2]))

            elif k == 2:  # Third timestep

                drp_soln[k+1, i] = drp_soln[k, i] - (dt/dx) * (
                            b0 * (am3 * drp_soln[k, i-3] + am2 * drp_soln[k, i-2] +
                                  am1 * drp_soln[k, i-1] + a1 * drp_soln[k, i+1] +
                                  a2 * drp_soln[k, i+2] + a3 * drp_soln[k, i+3]) +
                            b1 * (am3 * drp_soln[k-1, i-3] + am2 * drp_soln[k-1, i-2] +
                                  am1 * drp_soln[k-1, i-1] + a1 * drp_soln[k-1, i+1] +
                                  a2 * drp_soln[k-1, i+2] + a3 * drp_soln[k-1, i+3]) +
                            b2 * (am3 * drp_soln[k-2, i-3] + am2 * drp_soln[k-2, i-2] +
                                  am1 * drp_soln[k-2, i-1] + a1 * drp_soln[k-2, i+1] +
                                  a2 * drp_soln[k-2, i+2] + a3 * drp_soln[k-2, i+3]))

                std_6_soln[k+1, i] = std_6_soln[k, i] - (dt/dx) * (
                            b0 * (am3_6 * std_6_soln[k, i-3] + am2_6 * std_6_soln[k, i-2] +
                                  am1_6 * std_6_soln[k, i-1] + a1_6 * std_6_soln[k, i+1] +
                                  a2_6 * std_6_soln[k, i+2] + a3_6 * std_6_soln[k, i+3]) +
                            b1 * (am3_6 * std_6_soln[k-1, i-3] + am2_6 * std_6_soln[k-1, i-2] +
                                  am1_6 * std_6_soln[k-1, i-1] + a1_6 * std_6_soln[k-1, i+1] +
                                  a2_6 * std_6_soln[k-1, i+2] + a3_6 * std_6_soln[k-1, i+3]) +
                            b2 * (am3_6 * std_6_soln[k-2, i-3] + am2_6 * std_6_soln[k-2, i-2] +
                                  am1_6 * std_6_soln[k-2, i-1] + a1_6 * std_6_soln[k-2, i+1] +
                                  a2_6 * std_6_soln[k-2, i+2] + a3_6 * std_6_soln[k-2, i+3]))

                std_4_soln[k+1, i] = std_4_soln[k, i] - (dt/dx) * (
                            b0 * (am2_4 * std_4_soln[k, i-2] +
                                  am1_4 * std_4_soln[k, i-1] + a1_4 * std_4_soln[k, i+1] +
                                  a2_4 * std_4_soln[k, i+2]) +
                            b1 * (am2_4 * std_4_soln[k-1, i-2] +
                                  am1_4 * std_4_soln[k-1, i-1] + a1_4 * std_4_soln[k-1, i+1] +
                                  a2_4 * std_4_soln[k-1, i+2]) +
                            b2 * (am2_4 * std_4_soln[k-2, i-2] +
                                  am1_4 * std_4_soln[k-2, i-1] + a1_4 * std_4_soln[k-2, i+1] +
                                  a2_4 * std_4_soln[k-2, i+2]))

            else:  # Fourth timestep onward

                drp_soln[k+1, i] = drp_soln[k, i] - (dt/dx) * (
                        b0 * (am3 * drp_soln[k, i-3] + am2 * drp_soln[k, i-2] +
                              am1 * drp_soln[k, i-1] + a1 * drp_soln[k, i+1] +
                              a2 * drp_soln[k, i+2] + a3 * drp_soln[k, i+3]) +
                        b1 * (am3 * drp_soln[k-1, i-3] + am2 * drp_soln[k-1, i-2] +
                              am1 * drp_soln[k-1, i-1] + a1 * drp_soln[k-1, i+1] +
                              a2 * drp_soln[k-1, i+2] + a3 * drp_soln[k-1, i+3]) +
                        b2 * (am3 * drp_soln[k-2, i-3] + am2 * drp_soln[k-2, i-2] +
                              am1 * drp_soln[k-2, i-1] + a1 * drp_soln[k-2, i+1] +
                              a2 * drp_soln[k-2, i+2] + a3 * drp_soln[k-2, i+3]) +
                        b3 * (am3 * drp_soln[k-3, i-3] + am2 * drp_soln[k-3, i-2] +
                              am1 * drp_soln[k-3, i-1] + a1 * drp_soln[k-3, i+1] +
                              a2 * drp_soln[k-3, i+2] + a3 * drp_soln[k-3, i+3]))

                std_6_soln[k+1, i] = std_6_soln[k, i] - (dt/dx) * (
                        b0 * (am3_6 * std_6_soln[k, i-3] + am2_6 * std_6_soln[k, i-2] +
                              am1_6 * std_6_soln[k, i-1] + a1_6 * std_6_soln[k, i+1] +
                              a2_6 * std_6_soln[k, i+2] + a3_6 * std_6_soln[k, i+3]) +
                        b1 * (am3_6 * std_6_soln[k-1, i-3] + am2_6 * std_6_soln[k-1, i-2] +
                              am1_6 * std_6_soln[k-1, i-1] + a1_6 * std_6_soln[k-1, i+1] +
                              a2_6 * std_6_soln[k-1, i+2] + a3_6 * std_6_soln[k-1, i+3]) +
                        b2 * (am3_6 * std_6_soln[k-2, i-3] + am2_6 * std_6_soln[k-2, i-2] +
                              am1_6 * std_6_soln[k-2, i-1] + a1_6 * std_6_soln[k-2, i+1] +
                              a2_6 * std_6_soln[k-2, i+2] + a3_6 * std_6_soln[k-2, i+3]) +
                        b3 * (am3_6 * std_6_soln[k-3, i-3] + am2_6 * std_6_soln[k-3, i-2] +
                              am1_6 * std_6_soln[k-3, i-1] + a1_6 * std_6_soln[k-3, i+1] +
                              a2_6 * std_6_soln[k-3, i+2] + a3_6 * std_6_soln[k-3, i+3]))

                std_4_soln[k+1, i] = std_4_soln[k, i] - (dt/dx) * (
                        b0 * (am2_4 * std_4_soln[k, i-2] +
                              am1_4 * std_4_soln[k, i-1] + a1_4 * std_4_soln[k, i+1] +
                              a2_4 * std_4_soln[k, i+2]) +
                        b1 * (am2_4 * std_4_soln[k-1, i-2] +
                              am1_4 * std_4_soln[k-1, i-1] + a1_4 * std_4_soln[k-1, i+1] +
                              a2_4 * std_4_soln[k-1, i+2]) +
                        b2 * (am2_4 * std_4_soln[k-2, i-2] +
                              am1_4 * std_4_soln[k-2, i-1] + a1_4 * std_4_soln[k-2, i+1] +
                              a2_4 * std_4_soln[k-2, i+2]) +
                        b3 * (am2_4 * std_4_soln[k-3, i-2] +
                              am1_4 * std_4_soln[k-3, i-1] + a1_4 * std_4_soln[k-3, i+1] +
                              a2_4 * std_4_soln[k-3, i+2]))

        i += 1  # Progress space iterable
    k += 1  # Progress time iterable


time = 60  # Desired time to plot


# Plotting

fig1, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(alpha, num_alpha_drp, alpha, num_alpha_6, alpha, num_alpha_4)
ax1.set_title('Wave Number Relation')

ax2.plot(alpha[:-1], dn_da_drp, alpha[:-1], dn_da_6, alpha[:-1], dn_da_4)
ax2.set_title('Group Velocity')


fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(x, drp_soln[int(time/dt)])
ax1.set_title('4th-Order DRP')

ax2.plot(x, std_6_soln[int(time/dt)])
ax2.set_title('6th-Order Standard')

ax3.plot(x, std_4_soln[int(time/dt)])
ax3.set_title('4th-Order Standard')

ax4.plot(x, drp_soln[int(time/dt)], x, std_6_soln[int(time/dt)], x, std_4_soln[int(time/dt)])
ax4.set_title('Composite Comparison')

plt.show()

