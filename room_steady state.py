import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

v = 1.5 # m height of glass
x = 4   # m width of room
y = 5   # m depth of room
h = 2.6 # m height of room
dx = 0.9 # m width of door
dh = 2   # m height of door
ho = 25 # W/(m²⋅K)
hi= 8 # W/(m²⋅K)

ρa = 1.2 # kg/m³ air density
Ca = 1000 # J/(kg⋅K) air capacity
# ventilation flow rate
Va = x*y*h                   # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration

To = 10 # exterior temp 
Tn2 = 20 # temp neighbour w2
Tcor = 20 # temp corridor
Tn4 = 20 # temp neighbour w4
Tn5 = 20 # temp neighbour w5
Tn6 = 20 # temp neighbour w6
Tsp = 20 # set point temp
 
exttk = 0.2 # m exterior concrete wall thickness
inttk = 0.15 # m interior concrete wall thickness
gtk = 0.01 # m glass thickness
dtk = 0.1 # m door thickness
itk = 0.1 # thermal insulation thickness

λc = 1.8 # W/m °K  concrete
λg = 1.2 # W/m °K glass
λi = 0.04 # W/m °K thermal insulation
λd = 0.14 # W/m °K  door

ρc = 2200 # kg/m³
ρi = 70 # kg/m³
ρg = 2200 # kg/m³
ρd = 750 # kg/m³

Cc = 750 # J/(kg⋅K)
Ci = 1100 # J/(kg⋅K) 
Cg = 1150 # J/(kg⋅K)
Cd = 1750 # J/(kg⋅K) 

sw1 = x*(h-v) # m² surface area of wall 1
sglass = x*v  # m² surface area of glass
sw2 = y*h  # m² surface area of wall 2
sdoor = dx*dh # m² surface area of door
sw3 = x*h-sdoor # m² surface area of wall 3
sw4 = sw2 # m² surface area of wall 2
sw5 = x*y # m² surface area of wall 5
sw6 = sw5 # m² surface area of wall 6

# Incidence matrix, 30 heat flows, 20 nodes

A = np.zeros([30, 20])

A[0, 0] = 1
A[1,0] , A[1,1] = -1, 1
A[2,1] , A[2,2] = -1, 1
A[3,2] , A[3,3] = -1, 1
A[4,3] , A[4,4] = -1, 1
A[5,7] , A[5,5] = -1, 1
A[6,4] , A[6,6] = -1, 1
A[7,7] = 1
A[8,5] , A[8,6] = -1, 1
A[9,8] = 1
A[10,8] , A[10,9] = -1, 1
A[11,9] , A[11,6] = -1, 1
A[12,10] = 1
A[13,10] , A[13,11] = -1, 1
A[14,11] , A[14,6] = -1, 1
A[15,6] = 1
A[16,6] = 1
A[17,18] = 1
A[18,18] , A[18,19] = -1, 1
A[19,19] , A[19,6] = -1, 1
A[20,16] = 1
A[21,16] , A[21,17] = -1, 1
A[22,17] , A[22,6] = -1, 1
A[23,14] = 1
A[24,14] , A[24,15] = -1, 1
A[25,15] , A[25,6] = -1, 1
A[26,12] = 1
A[27,12] , A[27,13] = -1, 1
A[28,13] , A[28,6] = -1, 1
A[29,6] = 1

# Conductance matrix
G = np.zeros(30)

G[0] = ho*sw1
G[1] = λc/(exttk/2)*sw1
G[2] = λc/(exttk/2)*sw1
G[3] = λi/(itk/2)*sw1
G[4] = λi/(itk/2)*sw1
G[5] = λg/gtk*sglass
G[6] = hi*sw1
G[7] = ho*sglass
G[8] = hi*sglass
G[9] = ((λc/(inttk/2)*sw4)**-1 + (hi*sw4)**-1)**-1
G[10] = λc/(inttk/2)*sw4
G[11] = hi*sw4
G[12] = ((λc/(inttk/2)*sw5)**-1 + (hi*sw5)**-1)**-1
G[13] = λc/(inttk/2)*sw5
G[14] = hi*sw5
G[15] = ρa*Ca*Va_dot/2 #advection corridor
G[16] = 0 # set point controller
G[17] = ((λd/(dtk/2)*sdoor)**-1 + (hi*sdoor)**-1)**-1
G[18] = λd/(dtk/2)*sdoor
G[19] = hi*sdoor
G[20] = ((λc/(inttk/2)*sw3)**-1 + (hi*sw3)**-1)**-1
G[21] = λc/(inttk/2)*sw3
G[22] = hi*sw3
G[23] = ((λc/(inttk/2)*sw6)**-1 + (hi*sw6)**-1)**-1
G[24] = λc/(inttk/2)*sw6
G[25] = hi*sw6
G[26] = ((λc/(inttk/2)*sw2)**-1 + (hi*sw2)**-1)**-1
G[27] = λc/(inttk/2)*sw2
G[28] = hi*sw2
G[29] = ρa*Ca*Va_dot #exterior advection 
G = np.diag(G)

# capacities
c1 = exttk*sw1*ρc*Cc
c2 = itk *sw1*ρi*Ci
c3 = x*y*h*ρa*Ca
c4 = inttk*sw4*ρc*Cc
c5 = inttk*sw5*ρc*Cc
c6 = dtk*sdoor*ρd*Cd
c7 = inttk*sw3*ρc*Cc
c8 = inttk*sw6*ρc*Cc
c9 = inttk*sw2*ρc*Cc

# Capacity matrix
C= np.zeros(20)

C[1] = c1
C[3] = c2
C[6] = c3
C[8] = c4
C[10] = c5
C[18] = c6
C[16] = c7
C[14] = c8
C[12] = c9

C = np.diag(C)

# Temperature source vector
b = np.zeros(30)

b[0] = To
b[7] = To
b[9] = Tn4
b[12] = Tn5
b[15] = Tcor
b[16] = Tsp
b[17] = Tcor
b[20] = Tcor
b[23] = Tn6
b[26] = Tn2
b[29] = To

# Flow-rate source vector
f = np.zeros(20)
f[0] = 0
f[6] = 0 # interior accessories
f[19] = 0 #interior sun on door
f[17] = 0 #interior sun on w3

θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)

q = G @ (-A @ θ + b)

print ("Temperature inside room" ,round(θ[6],2),"ºC")