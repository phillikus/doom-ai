class Parameters():
    def __init__(self):
        self.scenario = 'Basic'
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 4
        self.num_steps = 20
        self.max_episode_length = 10000
        self.num_actions = 3
        self.model = 'a3c'
        self.num_updates = 100