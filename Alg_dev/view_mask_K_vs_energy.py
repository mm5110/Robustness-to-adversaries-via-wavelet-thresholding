import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

img_path = 'car.png'
step_alpha = 0.05
steps = 5000
levels = 4


loaded_data = np.load('mask_sparsity_vs_energy.npy')
# loaded_params = np.load('mask_sparsity_vs_energy_params.npy')

K_span = loaded_data[0]
mask_energies = loaded_data[1]
attack_success_record = loaded_data[2]



colors =[]
for i in range(len(K_span)):
	if attack_success_record[i] == 0:
		colors.append('red')
	else:
		colors.append('blue')


plt.scatter(K_span, mask_energies*100, c=colors, cmap=matplotlib.colors.ListedColormap(colors))
plt.suptitle('Energy vs sparsity of mask that resulted in misclassification', fontsize=14, fontweight='bold')
plt.title('Levels = ' + str(levels) + ', Image = ' + str(img_path) + ', Step size = ' + str(step_alpha) + ', Max number of steps = ' + str(steps))
plt.xlabel('Sparsity K')
plt.ylabel('l2 Energy ratio Mask/Original (%)')
plt.show()