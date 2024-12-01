# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:50:23 2024

@author: wbrum
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

# Disassembled thermal circuits
folder_bldg = 'bldg'
TCd = dm4bem.bldg2TCd(folder_bldg,
                      TC_auto_number=True)

# Assembled thermal circuit
ass_lists = pd.read_csv(folder_bldg + '/assembly_lists.csv')
ass_matrix = dm4bem.assemble_lists2matrix(ass_lists)
TC = dm4bem.assemble_TCd_matrix(TCd, ass_matrix)

TC['G']['c3_q0'] = 0  # Kp, controler gain
# TC['C']['c2_θ0'] = 0    # indoor air heat capacity
TC['C']['c1_θ0'] = 1089000    # glass (window) heat capacity

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As

dt_max = 2 * min(-1. / λ)    # max time step for Euler explicit stability
print(f'dt_max = {dt_max:.1f} s')
dt = dm4bem.round_time(dt_max)
dm4bem.print_rounded_time('dt', dt)

t_settle = 4 * max(-1. / λ)
# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)

# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# Create a DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                     periods=n, freq=f"{int(dt)}s")

To = 10.0 * np.ones(n)
Ti_sp = 20.0 * np.ones(n)
Φa = 0.0 * np.ones(n)
Qa = Φo = Φi = Φa

# key <- symbol in b and f of thermal circuit TC
# value <- time series of the source
data = {'To': To, 'Ti_sp': Ti_sp, 'Qa': Qa, 'Φo': Φo, 'Φi': Φi, 'Φa': Φa}
input_data_set = pd.DataFrame(data, index=time)

# Get input u from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# Initial conditions
θ_exp = pd.DataFrame(index=u.index)   # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)   # empty df with index for implicit Euler

θ0 = 0.0                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix

for k in range(n - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k + 1])
        
# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)
y.plot()
plt.xlabel('Time')
plt.ylabel('Temperature, $θ_i$ / °C')
plt.title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {dt_max:.0f} s')
plt.show()

date_start = '01-01 00:00:00'
date_end = '01-05 00:00:00'

date_start = '2000-' + date_start
date_end = '2000-' + date_end
print(f'{date_start} \tstart date')
print(f'{date_end} \tend date')

file_weather = 'weather_data/FRA_Lyon.074810_IWEC.epw'
[data, meta] = dm4bem.read_epw(file_weather, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data

weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather.loc[date_start:date_end]

# Temperature sources
To = weather['temp_air']

Ti_day, Ti_night = 20, 16
Ti_sp = pd.Series(20, index=To.index)
Ti_sp = pd.Series(
    [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
    index=To.index)

# total solar irradiance
wall_out = pd.read_csv(folder_bldg + '/walls_out.csv')
w0 = wall_out[wall_out['ID'] == 'w0']

surface_orientation = {'slope': w0['β'].values[0],
                       'azimuth': w0['γ'].values[0],
                       'latitude': 45}

rad_surf = dm4bem.sol_rad_tilt_surf(
    weather, surface_orientation, w0['albedo'].values[0])

Etot = rad_surf.sum(axis=1)

# Window glass properties
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass
S_g = 9         # m2, surface area of glass

# Flow-rate sources
# solar radiation
Φo = w0['α1'].values[0] * w0['Area'].values[0] * Etot
Φi = τ_gSW * w0['α0'].values[0] * S_g * Etot
Φa = α_gSW * S_g * Etot

# auxiliary (internal) sources
Qa = pd.Series(0, index=To.index)

# Input data set
input_data_set = pd.DataFrame({'To': To, 'Ti_sp': Ti_sp,
                               'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa,
                               'Etot': Etot})

# Resample hourly data to time step dt
input_data_set = input_data_set.resample(
    str(dt) + 's').interpolate(method='linear')

# Get input from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)

# initial conditions
θ0 = 20.0                   # initial temperatures
θ_exp = pd.DataFrame(index=u.index)
θ_exp[As.columns] = θ0      # Fill θ with initial valeus θ0

# time integration
I = np.eye(As.shape[0])     # identity matrix

for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]

# outputs
y = (Cs @ θ_exp.T + Ds @  u.T).T

Kp = TC['G']['c3_q0']     # W/K, controller gain
S = 3 * 3                 # m², surface area of the toy house
q_HVAC = Kp * (u['c3_q0'] - y['c2_θ0']) / S  # W/m²

data = pd.DataFrame({'To': input_data_set['To'],
                     'θi': y['c2_θ0'],
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
axs[0].set_title(f'Time step: $dt$ = {dt:.0f} s;'
                 f'$dt_{{max}}$ = {dt_max:.0f} s')
plt.show()

print(f"Mean. outdoor temp: {weather["temp_air"].mean():.0f} ºC")
print(f'Min. indoor temperature: {data["θi"].min():.1f} °C')
print(f'Max. indoor temperature: {data["θi"].max():.1f} °C')


max_load = data['q_HVAC'].max()
max_load_index = data['q_HVAC'].idxmax()
print(f"Max. load: {max_load:.1f} W at {max_load_index}")
print(f"Energy consumption: {(data['q_HVAC'] * dt).sum() / (3.6e6):.1f} kWh")