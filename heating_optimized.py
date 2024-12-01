# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:04:14 2024

@author: wbrum
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:01:59 2024

@author: wbrum
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import dm4bem

controller = True
Kp = 1000    # W/°C, controller gain

neglect_air_capacity = False
explicit_Euler = True

imposed_time_step = False
Δt = 3600    # s, imposed time step 

# MODEL
# =====
# Thermal circuits
TC = dm4bem.file2TC('will_room.csv', name='', auto_number=False)

# by default TC['G']['q11'] = 0 # Kp -> 0, no controller (free-floating
if controller:
    TC['G']['q16'] = Kp     # G16 = Kp, conductance of edge q16
                            # Kp -> ∞, almost perfect controller
if neglect_air_capacity:
    TC['C']['θ6'] = 0       # C6, capacity of vertex θ6 (air)
   

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
dm4bem.print_TC(TC)

λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As
dtmax = 2 * min(-1. / λ)    # max time step for Euler explicit stability
dt = dm4bem.round_time(dtmax)

if imposed_time_step:
    dt = Δt

dm4bem.print_rounded_time('dt', dt)

# INPUT DATA SET
# ==============
input_data_set = pd.read_csv('input_data_set_optimized.csv',
                             index_col=0,
                             parse_dates=True)

input_data_set = input_data_set.resample(
    str(dt) + 's').interpolate(method='linear')
input_data_set.head()

# Input vector in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)
u.head()

# Initial conditions
θ0 = -10                   # °C, initial temperatures
θ = pd.DataFrame(index=u.index)
θ[As.columns] = θ0          # fill θ with initial valeus θ0

I = np.eye(As.shape[0])     # identity matrix

if explicit_Euler:
    for k in range(u.shape[0] - 1):
        θ.iloc[k + 1] = (I + dt * As) @ θ.iloc[k] + dt * Bs @ u.iloc[k]
else:
    for k in range(u.shape[0] - 1):
        θ.iloc[k + 1] = np.linalg.inv(
            I - dt * As) @ (θ.iloc[k] + dt * Bs @ u.iloc[k])
        
# outputs
y = (Cs @ θ.T + Ds @  u.T).T

Kp = TC['G']['q16']     # controller gain
S = 20                 # m², surface area of room
q_HVAC = Kp * (u['q16'] - y['θ6']) / S  # W/m²
y['θ6']

data = pd.DataFrame({'To': input_data_set['To'],
                     'θi': y['θ6'],
                     'Etot': input_data_set['Etot'],
                     'q_HVAC': q_HVAC})

fig, axs = plt.subplots(2, 1)
data[['To', 'θi']].plot(ax=axs[0],
                        xticks=[],
                        ylabel='Temperature, $θ$ / °C')

axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'],
              loc='upper right')

data[['Etot', 'q_HVAC']].plot(ax=axs[1],
                              ylabel='Heat rate, $q$ / (W·m⁻²)')
axs[1].set(xlabel='Time')
axs[1].legend(['$E_{total}$', '$q_{HVAC}$'],
              loc='upper right')
plt.show();