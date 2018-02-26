import matplotlib.pyplot as plt


def log_reward(reward):
    with open("rewards.txt", "a") as reward_file:
        reward_file.write(str(reward) + "\n")


def show_graph():
    rewards = []

    with open("rewards.txt", "r") as reward_file:
        for line in reward_file:
            line = line.strip()
            rewards.append(float(line))

    epochs = range(0, len(rewards))

    plt.figure()
    axes = plt.axes()
    axes.plot(epochs, rewards)

    plt.title("Rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")

