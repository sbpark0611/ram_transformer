import math

import torch
import torch.nn as nn


class EPN(nn.Module):
    def __init__(
        self,
        num_labels: int,
        num_actions: int,
        embedding_size: int = 16,
        num_heads: int = 1,
        hidden_size: int = 64,
        num_iterations: int = 1,
    ):
        super(EPN, self).__init__()
        self.state_embedding = nn.Embedding(num_labels + 1, embedding_size)
        self.action_embedding = nn.Embedding(num_actions + 1, embedding_size)
        self.planner = Planner(embedding_size, num_heads, num_iterations)
        self.planner_mlp = nn.Sequential(
            nn.Linear(
                self.planner.embedding_size + embedding_size, self.planner.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(self.planner.hidden_size, self.planner.embedding_size),
        )

        self.feature = nn.Sequential(
            nn.Linear(embedding_size * 2 + self.planner.embedding_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, memory, obs):
        states = self.state_embedding(memory["position"])
        _, fixed_num_steps, _ = states.size()
        goal = self.state_embedding(obs["goal"])
        goals = goal.unsqueeze(1).expand(-1, fixed_num_steps, -1)
        actions = self.action_embedding(memory["prev_action"])
        prev_states = self.state_embedding(memory["prev_position"])

        episodic_storage = torch.concat((states, actions, prev_states, goals), dim=-1)

        belief_state = self.planner(episodic_storage)

        current_state = self.state_embedding(obs["position"])
        current_states = current_state.unsqueeze(1).expand(-1, fixed_num_steps, -1)
        belief_state = torch.cat((belief_state, current_states), dim=2)
        belief_state = self.planner_mlp(belief_state)
        # print(belief_state.max(dim=1)[0].shape)
        planner_output = torch.max(belief_state, dim=1)[0]

        state_goal_embedding = torch.concat((current_state, goal), dim=-1)
        combined_embedding = torch.cat((planner_output, state_goal_embedding), dim=1)
        combined_embedding = self.feature(combined_embedding)
        return combined_embedding


class Planner(nn.Module):
    def __init__(
        self, 
        embedding_size, 
        num_heads, 
        num_iterations, 
        ffn_act_ftn="nmda", 
        alpha=1.0, 
        beta=1.0, 
        dropout=0
    ):
        super(Planner, self).__init__()
        self.embedding_size = 4 * embedding_size
        self.num_heads = num_heads
        self.hidden_size = 16 * embedding_size
        self.num_iterations = num_iterations
        self.ln_1 = nn.LayerNorm(self.embedding_size)
        self.self_attn = nn.MultiheadAttention(
            self.embedding_size, num_heads, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(self.embedding_size)
        if ffn_act_ftn != "nmda":
            act_nn = _act_fns[ffn_act_ftn]
        else:
            act_nn = NMDA(alpha, beta)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size, bias=False),
            act_nn,
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.embedding_size, bias=False),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(self.embedding_size)

    def forward(self, x):
        for _ in range(self.num_iterations):
            _x = self.ln_1(x)
            attn_output, _ = self.self_attn(_x, _x, _x)
            x = x + self.dropout1(attn_output)
            x = x + self.dropout2(self.mlp(self.ln_2(x)))
        return self.ln_f(x)


class NMDA(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(NMDA, self).__init__()
        self.alpha = alpha
        if alpha <= 0:
            self.a = None
        else:
            self.a = math.log(self.alpha)
        self.beta = beta

    def forward(self, x):
        if self.a is None:
            return x
        else:
            return x * torch.sigmoid(self.beta * x - self.a)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


_act_fns = {
    "gelu": nn.GELU(),
    "quick_gelu": QuickGELU(),
    "relu": nn.ReLU(inplace=True),
    "swish": nn.SiLU(),
    "linear": nn.Identity(),
}