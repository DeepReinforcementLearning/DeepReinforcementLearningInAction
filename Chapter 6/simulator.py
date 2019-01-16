import torch
from torch.nn import functional as F

class SimulatorState(torch.nn.Module):
    def __init__(self):
        super(SimulatorState, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.s_fc1 = torch.nn.Linear(512, 99)

        self.action_fc1 = torch.nn.Linear(4, 1)

        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 16)

        #         self.s_fc1 = torch.nn.Linear(50, 16)

        self.reward_fc1 = torch.nn.Linear(50, 30)
        self.reward_fc2 = torch.nn.Linear(30, 20)
        self.reward_fc3 = torch.nn.Linear(20, 3)

    def forward(self, x):
        state = x[:, :64]
        a_x = x[:, 64:]

        num_batch = state[0]

        s_x = state.reshape(-1, 4, 4, 4)

        s_x = F.relu(self.conv1(s_x))
        s_x = F.relu(self.conv2(s_x))
        s_x = F.relu(self.conv3(s_x))
        s_x = s_x.view(-1, 8 * 64)
        s_x = self.s_fc1(s_x)

        a_x = self.action_fc1(a_x)

        x = torch.cat((s_x, a_x), dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #         state_copy = state.reshape(-1,4,16).clone()
        #         state_copy[np.arange(num_batch)][0][:] = 0
        #         state_copy[np.arange(num_batch),s_x] = 1

        #         r_x = F.relu(self.reward_fc1(state_copy))
        #         r_x = F.relu(self.reward_fc2(r_x))
        #         r_x = self.reward_fc3(r_x)

        return F.softmax(x)


class SimulatorReward(torch.nn.Module):
    def __init__(self):
        super(SimulatorReward, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = torch.nn.Linear(512, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 3)

    def forward(self, x):
        x = x.reshape(-1, 4, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        x = x.view(-1, 512)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x)