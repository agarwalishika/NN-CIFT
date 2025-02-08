import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(data, fname, title):
    x1, x2 = [], []
    for d in data:
        vals = d.split(',')
        x1.append(float(vals[0]))
        x2.append(float(vals[1]))
    x1_set = list(set(x1))
    x2_set = list(set(x2))

    performance = np.zeros((len(x1_set), len(x2_set)))
    for d in data:
        vals = d.split(',')
        a = float(vals[0])
        b = float(vals[1])
        y = float(vals[2].strip())
        performance[x1_set.index(a), x2_set.index(b)] = y
    # import pdb; pdb.set_trace()
    
    x1, x2 = np.meshgrid(x1_set, x2_set)  # Create a grid for both hyperparameters

    # x1 = np.array(x1)
    # x2 = np.array(x2)
    performance = np.array(performance)

    # Heatmap
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x1, x2, performance, cmap='viridis', shading='auto')  # Filled contour plot
    plt.colorbar(label='Performance')
    plt.xlabel('u')
    plt.ylabel('v')
    # plt.title('Heatmap of % of NN training and model training samples versus performance')
    plt.title(title)
    plt.savefig('cache/hyperparam_figs/' + fname, bbox_inches="tight")

    # # 3D Surface Plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(x1, x2, y, cmap='viridis', edgecolor='k', alpha=0.8)
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Y Variable')
    # ax.set_xlabel('Hyperparameter 1')
    # ax.set_ylabel('Hyperparameter 2')
    # ax.set_zlabel('Y Variable')
    # ax.set_title('3D Surface Plot of Hyperparameters vs. Y')
    # plt.show()


with open('cache/hyperparameter_study_results.txt', 'r') as f:
    lines = f.readlines()

peft_delift_data, peft_nn_data, icl_delift_data, icl_nn_data = [], [], [], []
for line in lines:
    vals = line.split(',')
    data = f'{vals[1]},{vals[2]},{vals[4]}'
    if vals[0] == "peft":
        if vals[3] == "True":
            peft_delift_data.append(data)
        else:
            peft_nn_data.append(data)
    else:
        if vals[3] == "True":
            icl_delift_data.append(data)
        else:
            icl_nn_data.append(data)


visualize(peft_delift_data, "icl_delift", "DELIFT (ICL)")
visualize(peft_nn_data, "icl_nn", "NN-CIFT + DELIFT (ICL)")
visualize(icl_delift_data, "peft_delift", "DELIFT (QLoRA)")
visualize(icl_nn_data, "peft_nn", "NN-CIFT + DELIFT (QLoRA)")
