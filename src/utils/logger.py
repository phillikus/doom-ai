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

    epochs, rewards = avg_rewards(rewards)
    axes.plot(epochs, rewards)

    plt.title("Rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")


def avg_rewards(rewards):
    epochs = []
    new_rewards = []
    for i in range(0, 40):
        start_index = i*100
        epochs.append(start_index)
        new_rewards.append(sum(rewards[start_index: start_index+100]) / 100)

    return epochs, new_rewards