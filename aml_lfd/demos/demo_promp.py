import numpy as np
import matplotlib.pyplot as plt
from aml_lfd.promp.discrete_promp_shell import DiscretePROMPShell

p_shell = DiscretePROMPShell()

plt.figure("DiscretePROMPShell")

# Generate and plot trajectory Data
x = np.arange(0,1.01,0.01)           # time points for trajectories
nrTraj=30                            # number of trajectoreis for training
sigmaNoise=0.02                      # noise on training trajectories
A = np.array([.2, .2, .01, -.05])
X = np.vstack( (np.sin(5*x), x**2, x, np.ones((1,len(x))) ))

Y = np.zeros( (nrTraj,len(x)) )
for traj in range(0, nrTraj):
    sample = np.dot(A + sigmaNoise * np.random.randn(1,4), X)[0]
    label = 'training' if traj==0 else ''

    plt.plot(x, sample, 'b', label=label)
    p_shell.add_demo_traj(sample)


p_shell.train()

# p_shell.add_viapoint(0.7, 5)


p_shell.set_start(0)
p_shell.set_goal(0.2)

for i in np.arange(0,10):
    label = 'output' if i==0 else ''
    plt.plot(p_shell._x, p_shell.generate_trajectory(tau=1.), 'r', label=label)

plt.legend()
plt.show()