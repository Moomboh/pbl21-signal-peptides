# Libraries
import numpy as np
import matplotlib.pyplot as plt

####################################################################################
SP_TYPES = ['NO_SP', 'SP', 'LIPO', 'TAT']
y_pos = np.arange(len(SP_TYPES))
performance = [0.7597037475796905, 0.637187192942397, 0.5415494667519204, 0.34979892755731556]
crosstrain_error = [0.016748739846423014, 0.018951570311385873, 0.04004876143582493, 0.14663033623497615]


plt.bar(y_pos, performance, align='center', yerr=crosstrain_error, capsize=8, width=0.5, color='dodgerblue', label='crosstrain')
plt.xticks(y_pos, SP_TYPES)
plt.ylim(0,1)

plt.ylabel('scores')
plt.title('MCC per sp-type', fontweight='bold')
plt.legend()

plt.savefig('mcc_per_type.png', dpi=240)
plt.show()