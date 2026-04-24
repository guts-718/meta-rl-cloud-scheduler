import matplotlib.pyplot as plt
import numpy as np

def plot_training_curve(rewards):
    plt.figure()
    plt.plot(rewards, label="RL Reward")

    # smooth curve
    window = 10
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(len(smoothed)), smoothed, label="Smoothed", linewidth=2)

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("RL Training Curve")
    plt.legend()
    plt.grid()

    plt.savefig("training_curve.png")
    plt.show()