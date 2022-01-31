# Libraries
import numpy as np
import matplotlib.pyplot as plt

####################################################################################
# set width of bars
barWidth = 0.4

# set heights of bars
# NO_SP, SP, LIPO, TAT
crosstrain = [0.7594118455798342, 0.636152257336045, 0.540813602107522, 0.3495166319442282]
test = [0.7846751543789857, 0.643289986162721, 0.5854979742269555, 0.5467246498886504]


# Set position of bar on X axis
r1 = np.arange(len(crosstrain))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

crosstrain_error = [0.016505129161257178, 0.01823541140679819, 0.04045275924547164, 0.14550379634493096]
test_error = [0.022064922998550916, 0.034377549964073835, 0.05043969641076096, 0.09549838178328399]

# Make the plot
plt.bar(r1, crosstrain, color='dodgerblue', width=barWidth, edgecolor='white', label='crosstrain', yerr=crosstrain_error, capsize=8)
plt.bar(r2, test, color='royalblue', width=barWidth, edgecolor='white', label='test', yerr=test_error, capsize=8)
#plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='ACC')

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(crosstrain))], ['NO_SP', 'SP', 'LIPO', 'TAT'])
plt.ylabel('MCC')

#plt.title('MCC per sp-type', fontweight='bold')
plt.ylim(0,1)

# Create legend & Show graphic
plt.legend()
plt.savefig('crosstrain-test.png', dpi=240)

plt.show()