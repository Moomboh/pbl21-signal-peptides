import pandas
import matplotlib.pyplot as plt
import numpy as np

df = pandas.DataFrame(dict(graph=['MCC', 'Recall', 'Precision'],
                           n=[0, 0.25004488375512135, 0.25004488375512135], m=[0.667079947433487, 0.7176881503232695, 0.5912574653626224]))
# random prec:0.25004488375512135	rec:0.25004488375512135
# crosstrain MCC: 0.667079947433487 prec: 0.5912574653626224 rec: 0.7176881503232695

ind = np.arange(len(df))
width = 0.4


crosstrain_error = [0.012945666545119323, 0.038196153957134, 0.02802633102123152]
random_error = [0, 0.0058774104690544355, 0.005872928514439003]

fig, ax = plt.subplots()
ax.barh(ind, df.n, width, color='lightsteelblue', label='random baseline', xerr=random_error, capsize=8)
ax.barh(ind + width, df.m, width, color='dodgerblue', label='crosstrain', xerr=crosstrain_error, capsize=8)

ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])

#plt.title('crosstrain scores compared to random baseline scores', fontweight='bold')
plt.xlim(0,1)

ax.legend()

plt.savefig('crosstrain-random.png', dpi=240)
plt.show()