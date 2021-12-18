# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:22:42 2021

@author: John Paolo Acosta
"""

import pylab as plt
import numpy as np
import ahkab
from matplotlib.pyplot import *
get_ipython().run_line_magic('pylab', 'inline')
figsize = (15, 10)
from ahkab import circuit, printing, time_functions

rv = np.array([[20, -5, 0, -10],
               [-5, 20, -5, -10],
               [0, -5, 20, -10],
               [-10, -10, -10, 40]])
rv
irv = np.linalg.inv(rv)
irv
vs = np.array([[24], [0],[-15], [0]])
vs
print(irv.dot(vs))

#creates circuit object
fourloopcircuit = ahkab.Circuit('Four Loop Circuit')

#variable for ground fourloopcircuit
gnd = fourloopcircuit.get_ground_node()

#adds elements to four loop circuit and location such as resisitors and voltage sources
#with ground and nodes
fourloopcircuit.add_resistor('R1', 'n1', n2 = gnd, value = 10000)
fourloopcircuit.add_vsource('V1', 'n2', 'n1', dc_value = 10)
fourloopcircuit.add_resistor('R2', 'n2', n2 = gnd, value = 5000)
fourloopcircuit.add_vsource('V2', 'n3', 'n2', dc_value = 5)
fourloopcircuit.add_resistor('R3', 'n3', n2 = gnd, value = 2000)
fourloopcircuit.add_resistor('R4', 'n3', 'n4', value = 3000)
fourloopcircuit.add_vsource('V3', 'n4', n2 = gnd, dc_value = 8)
fourloopcircuit.add_resistor('R5', 'n2', 'n4', value = 3000)

opa = ahkab.new_op()
r = ahkab.run(fourloopcircuit, opa)['op']

print(r)

#creates a RLC circuit object
rlc_circuit = circuit.Circuit(title="RLC")

#variable for ground RLC circuit
gnd2 = rlc_circuit.get_ground_node()

#add elements of RLC circuit
rlc_circuit.add_resistor('R1', 'n1', 'n2', value = 500)
rlc_circuit.add_inductor('L1', 'n2', 'n3', value = 13e-3)
rlc_circuit.add_capacitor('C1', 'n3', n2 = gnd2, value = 120e-9)
rlc_circuit.add_inductor('L2', 'n3', 'n4', value = 70e-3)
rlc_circuit.add_capacitor('C2', 'n4', n2 = gnd2, value = 150e-9)
rlc_circuit.add_inductor('L3', 'n4', 'n5', value = 90e-3)
rlc_circuit.add_capacitor('C3', 'n5', n2 = gnd2, value = 170e-9)
rlc_circuit.add_resistor('R2', 'n5', n2 = gnd2, value = 1500)

voltage_step = time_functions.pulse(v1 = 0, v2 = 1, td = 500e-9,
tr = 1e-12, pw  =1, tf = 1e-12, per = 2)

rlc_circuit.add_vsource('V1', 'n1', n2 = gnd2, dc_value = 5,
ac_value = 1, function = voltage_step)

op_analysis = ahkab.new_op()
ac_analysis = ahkab.new_ac(start = 1e3, stop = 1e5, points =100)
tran_analysis = ahkab.new_tran(tstart = 0, tstop = 1.2e-3, tstep = 1e-6,
x0 = None)

q = ahkab.run(rlc_circuit, an_list = [op_analysis, ac_analysis, tran_analysis])

#plots tran simulation
fig = plt.figure()
plt.title(rlc_circuit.title + " - TRAN Simulation")
plt.plot(q['tran']['T'], q['tran']['VN1'], label = "Input Voltage")
#plt.hold(True)
plt.plot(q['tran']['T'], q['tran']['VN4'], label = "Output Voltage")
plt.legend()
#plt.hold(False)
plt.grid(True)
plt.ylim([0, 1.2])
plt.ylabel('Step Response')
plt.xlabel('Time [s]')
fig.savefig('tran_plot.png')

#plots ac simulation
fig = plt.figure()
plt.subplot(211)
plt.semilogx(q['ac']['f'], np.abs(q['ac']['Vn4']), 'o-')
plt.ylabel('abs(V(n4)) [V]')
plt.title(rlc_circuit.title + " - AC Simulation")
plt.subplot(212)
plt.grid(True)
plt.semilogx(q['ac']['f'], np.angle(q['ac']['Vn4']), 'o-')
plt.ylabel('arg(V(n4)) [rad]')
plt.xlabel('Frequency')
fig.savefig('ac_plot.png')
plt.show()


#creates object RLC poles and zeros
rlc_pz = ahkab.Circuit('RLC pole and zero')
gnd2 = rlc_pz.get_ground_node()
#add elements to the rlc_pz 
rlc_pz = ahkab.Circuit('RLC bandpass')
rlc_pz.add_inductor('L1', 'in', 'n1', 2e-6)
rlc_pz.add_capacitor('C1', 'n1', 'out', 3.4e-12)
rlc_pz.add_resistor('R1', 'out', gnd2, 20)
rlc_pz.add_vsource('V1', 'in', gnd2, dc_value=1, ac_value=1)

#print the netlist of rlc_pz
print(rlc_pz)

#results are saved in the pz_solution in object r
pza = ahkab.new_pz('V1', ('out', gnd2), x0=None, shift=1e3)
r = ahkab.run(rlc_pz, pza)['pz']
r.keys()

#prints the poles and zeros 
print('Singularities:')
for x, _ in r:
    print ("* %s = %+g %+gj Hz" % (x, np.real(r[x]), np.imag(r[x])))

figure(figsize=figsize)
# zeroes 
for x, v in r:
    plot(np.real(v), np.imag(v), 'bo'*(x[0]=='z')+'rx'*(x[0]=='p'))
# set axis limits and print some thin axes
xm = 1e6
xlim(-xm*10., xm*10.)
plot(xlim(), [0,0], 'k', alpha=.5, lw=.5)
plot([0,0], ylim(), 'k', alpha=.5, lw=.5)
# plot the distance from the origin of p0 and p1
plot([np.real(r['p0']), 0], [np.imag(r['p0']), 0], 'k--', alpha=.5)
plot([np.real(r['p1']), 0], [np.imag(r['p1']), 0], 'k--', alpha=.5)
# print the distance between p0 and p1
plot([np.real(r['p1']), np.real(r['p0'])], [np.imag(r['p1']), np.imag(r['p0'])], 'k-', alpha=.5, lw=.5)
# label the singularities
text(np.real(r['p1']), np.imag(r['p1'])*1.1, '$p_1$', ha='center', fontsize=20)
text(.4e6, .4e7, '$z_0$', ha='center', fontsize=20)
text(np.real(r['p0']), np.imag(r['p0'])*1.2, '$p_0$', ha='center', va='bottom', fontsize=20)
xlabel('Real [Hz]'); ylabel('Imag [Hz]'); title('Singularities');

#Calculation of Resonance Frequancy
C = 3.4e-12
L = 2e-6
f0 = 1./(2*np.pi*np.sqrt(L*C))
print ('Resonance frequency from analytic calculations: %g Hz' %f0)

#AC analysis
alpha = (-r['p0']-r['p1'])/2
a1 = np.real(abs(r['p0'] - r['p1']))/2
f0 = np.sqrt(a1**2 - alpha**2)
f0 = np.real_if_close(f0)
print ('Resonance frequency from PZ analysis: %g Hz' %f0)

aca = ahkab.new_ac(start=1e7, stop=5e9, points=5e2, x0=None)
rac = ahkab.run(rlc_pz, aca)['ac']

import sympy
sympy.init_printing()

#plot is dB
def dB20(x):
    return 20*np.log10(x)

from sympy.abc import w
from sympy import I
p0, p1, z0 = sympy.symbols('p0, p1, z0')
k = 20/2e-6 # constant term, can be calculated to be R/L
H = 20/2e-6*(I*w + z0*6.28)/(I*w +p0*6.28)/(I*w + p1*6.28)
Hl = sympy.lambdify(w, H.subs({p0:r['p0'], z0:abs(r['z0']), p1:r['p1']}))

figure(figsize=figsize)
semilogx(rac.get_x()/2/np.pi, dB20(abs(rac['vout'])), label='TF from AC analysis')
legend(); xlabel('Frequency [Hz]'); ylabel('|H(w)| [dB]'); xlim(1e6, 3e8); ylim(-50, 1);

import sympy
sympy.init_printing()

#Symbolic Analysis
from sympy.abc import w
from sympy import I
p0, p1, z0 = sympy.symbols('p0, p1, z0')
k = 20/2e-6 # constant term, can be calculated to be R/L
H = 20/2e-6*(I*w + z0*6.28)/(I*w +p0*6.28)/(I*w + p1*6.28)

Hl = sympy.lambdify(w, H.subs({p0:r['p0'], z0:abs(r['z0']), p1:r['p1']}))

def dB20(x):
    return 20*np.log10(x)

figure(figsize=figsize)
semilogx(rac.get_x()/2/np.pi, dB20(abs(Hl(rac.get_x()))), 'o', ms=4, label='TF from PZ analysis')
legend(); xlabel('Frequency [Hz]'); ylabel('|H(w)| [dB]'); xlim(1e7, 1e9); ylim(-50, 1);

symba = ahkab.new_symbolic(source='V1')
rs, tfs = ahkab.run(rlc_pz, symba)['symbolic']

#gets transfer function
print(rs)
print (tfs)
tfs['VOUT/V1']
Hs = tfs['VOUT/V1']['gain']
s, C1, R1, L1 = rs.as_symbols('s C1 R1 L1')
HS = sympy.lambdify(w, Hs.subs({s:I*w, C1:3.4e-12, R1:20., L1:2e-6}))

np.allclose(dB20(abs(HS(rac.get_x()))), dB20(abs(Hl(rac.get_x()))), atol=1)

figure(figsize=figsize);  title('Series RLC passband: TFs compared')
semilogx(rac.get_x()/2/np.pi, dB20(abs(rac['vout'])), label='TF from AC analysis')
semilogx(rac.get_x()/2/np.pi, dB20(abs(Hl(rac.get_x()))), 'o', ms=4, label='TF from PZ analysis')
semilogx(rac.get_x()/2/np.pi, dB20(abs(HS(rac.get_x()))), '-', lw=10, alpha=.2, label='TF from symbolic analysis')
vlines(1.07297e+08, *gca().get_ylim(), alpha=.4)
text(7e8/2/np.pi, -45, '$f_d = 107.297\\, \\mathrm{MHz}$', fontsize=20)
legend(); xlabel('Frequency [Hz]'); ylabel('|H(w)| [dB]'); xlim(1e6, 1e9); ylim(-50, 1);