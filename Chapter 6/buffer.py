import random

class ExperienceReplay():
    def __init__(self):
        self.buffer = []
        self.buffer_size = 1000

    def add(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.buffer_size: self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, size):
        return random.sample(self.buffer, size)

    #         buffer = sorted(self.buffer, key=lambda replay: abs(replay[3]) > 0, reverse=True)
    #         p = np.array([0.99 ** i for i in range(len(buffer))])
    #         p = p / sum(p)
    #         sample_idxs = np.random.choice(np.arange(len(buffer)),size=size, p=p)
    #         sample_output = [buffer[idx] for idx in sample_idxs]
    # #         print(sample_output)
    # #         sample_output = np.reshape(sample_output,(size,-1))
    #         return sample_output

    def __len__(self):
        return len(self.buffer)