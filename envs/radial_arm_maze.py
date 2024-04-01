# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory & Planning Game environment."""
import string
from collections import OrderedDict

import networkx as nx
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class RadialArmMaze(gym.Env):
    """Memory & Planning Game environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        maze_size=5,
        num_maze=10,
        max_episode_steps=100,
        target_reward=1.0,
        per_step_reward=0.0,
        num_labels=10,
        num_arms=2,
        render_mode=None,
        maps=None,
        epn=False,
        pretrain=False,
        no_duplication=True,
        seed=None,
        **kwargs,
    ):
        super(RadialArmMaze, self).__init__()

        self._maze_size = maze_size
        self._num_maze = num_maze
        self._num_labels = num_labels        
        
        self._num_arms = num_arms

        self._graph = nx.grid_2d_graph(self._num_arms + 1, self._maze_size, periodic=True)

        self._max_episode_steps = max_episode_steps
        self._target_reward = target_reward
        self._per_step_reward = per_step_reward
        self._oracle_min_num_actions = _oracle_min_actions(self._maze_size)
        self._pretrain_mode = pretrain
        self._test = False
        self._reverse_mode = 0
        self._no_duplication = no_duplication

        if self._no_duplication:
            self._num_labels = self._maze_size * (self._num_arms + 1)

        self.pos2idx = {n: i for i, n in enumerate(self._graph.nodes())}

        self.action_space = spaces.Discrete(self._num_arms + 1)
        self.epn = epn
        if self.epn:
            self.observation_space = spaces.Dict(
                {
                    "position": spaces.Discrete(self._num_labels + 1),
                    "prev_position": spaces.Discrete(self._num_labels + 1),
                    "goal": spaces.Discrete(self._num_labels + 1),
                    "prev_action": spaces.Discrete(self._num_arms + 2),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "position": spaces.Discrete(self._num_labels),
                    "goal": spaces.Discrete(self._num_labels),
                    "prev_action": spaces.Discrete(self._num_arms + 2),
                    "reward": spaces.Discrete(2),
                }
            )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.window_size = 512
        self.clock = None
        self._is_respawn = False
        self.num_steps = []
        if maps is None:
            self._init_world(seed)
        else:
            self.labels = maps["labels"]
            self.goals = maps["goals"]

    def _init_world(self, seed):
        super().reset(seed=seed)
        if self._num_maze > 0:
            self.labels = self.np_random.integers(
                0,
                self._num_labels,
                size=(self._num_maze, self._num_arms + 1, self._maze_size),
            )  # obs
            # Initialize fixed goals
            self.goals = []
            for _ in range(self._num_maze):
                fixed_goal = (
                    self.np_random.integers(self._maze_size),
                    self.np_random.integers(self._maze_size),
                )
                self.goals.append(fixed_goal)

    def _get_obs(self):
        if self.epn:
            return OrderedDict(
                [
                    ("position", self.position + 1),
                    ("prev_position", self.prev_position + 1),
                    ("goal", self.goal + 1),
                    ("prev_action", self.previous_action + 1),
                ]
            )
        else:
            return OrderedDict(
                [
                    ("position", self.position),
                    ("goal", self.goal),
                    ("prev_action", self.previous_action),
                    ("reward", 1 if self.is_respawn else 0),
                ]
            )

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                np.array(self._position) - np.array(self._goal), ord=1
            ),
            "respawn": self.is_respawn,
            "max_reward": self._max_episode_steps / self.oracle_min_num_actions,
            "prev_pos": self.pos2idx[self._prev_position]
            if isinstance(self._prev_position, tuple)
            else -1,
            "pos": self.pos2idx[self._position],
            "steps": self.num_steps,
        }
    
    def get_next_position(self, position, action):
        if position[1] == self._maze_size - 1: # pick one arm
            return tuple([action, self._maze_size - 2])
        if position[0] == 0: # first raw
            if action == 1: # forward
                return tuple([position[0], position[1] + 1])
            if action == 2: # backward
                return tuple([position[0], max(position[1] - 1, 0)])
            return position
        else:
            if action == 1: # forward
                return tuple([position[0], max(position[1] - 1, 0)])
            if action == 2: # backward
                return tuple([position[0], position[1] + 1])
            return position

    def step(self, action):
        # for x in range(self._num_arms+1):
        #     for y in range(self._maze_size):
        #         p = tuple([x, y])
        #         if p == self._goal:
        #             print('g', end=' ')
        #         elif p == self._position:
        #             print('o', end=' ')
        #         else:
        #             print(self._labels[p], end=' ')
        #     print()
        # print('---------------------')
        self._episode_steps += 1
        self._steps2goal += 1
        self._prev_position = self._position
        self._position = self.get_next_position(self._position, action)
        self._prev_action = action
        if (
            self._position == self._goal
            and action == 0
            and not self._pretrain_mode
        ):
            reward = self._target_reward
            self._set_new_goal()
            self._position = tuple([0, 0])
        else:
            reward = self._per_step_reward
            self._is_respawn = False
        self._episode_reward += reward
        done = self._episode_steps >= self._max_episode_steps

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, done, False, self._get_info()

    def reset(self, seed=None):
        super().reset(seed=seed)

        if self.epn:
            self._prev_action = -1
            self._prev_position = -1
        else:
            self._prev_action = 5
            self._prev_position = None
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._steps2goal = 0
        self.num_steps = []
        if self._test or self._num_maze <= 0:
            if self._no_duplication:
                sequence = np.arange(
                    self._num_labels
                )  # Create a sequence from 0 to n-1
                random_labels = np.random.permutation(sequence)
            else:
                random_labels = self.np_random.integers(
                    0, self._num_labels, size=(self._maze_size * (self._num_arms + 1),)
                )
            self._respawn()
            self._set_new_goal(start=True)
            self._labels = {
                n: random_labels[i] for i, n in enumerate(self._graph.nodes())
            }
        else:
            self._env_idx = self.np_random.integers(self._num_maze)
            random_labels = self.labels[self._env_idx].reshape(-1)
            self._labels = {
                n: random_labels[i] for i, n in enumerate(self._graph.nodes())
            }
            self._respawn()
            self._goal = self.goals[self._env_idx]

        if self.render_mode == "human":
            self._render_frame()

        self.switched = False

        return self._get_obs(), self._get_info()

    def _respawn(self):
        self._position = tuple([0, 0])

    def _set_opposite_position(self, pos):
        mirror_arm = pos[0] % self._num_arms + 1
        return tuple([mirror_arm, pos[1]])

    def _set_new_goal(self, start=False):
        self._goal = tuple([self.np_random.integers(self._num_arms) + 1, 0])
        self.switched = False
        if start:
            self._is_respawn = False
            self._steps2goal = 0
        else:
            self._is_respawn = True
            self.num_steps.append(self._steps2goal)
            self._steps2goal = 0

    def set_pretrain(self, pretrain=True):
        self._pretrain_mode = pretrain

    def test_mode(self):
        self._test = True

    def reverse_mode(self, case=1):
        self._reverse_mode = case

    def pos_idx(self):
        return self.pos2idx[self._position]

    @property
    def position(self):
        pos = self._labels[self._position]

        if self._position[1] == self._maze_size - 1:
            pos = self._labels[0, self._maze_size - 1]

        if self._episode_steps >= self._max_episode_steps // 2:
            # 1. Position of goal and the label of position move to the opposite site
            if self._reverse_mode == 1 and self._position in [self._goal, self._set_opposite_position(self._goal)]:
                pos = self._labels[self._set_opposite_position(self._position)]
            elif self._reverse_mode == 3 and self._position in [self._goal, self._set_opposite_position(self._goal)]:
                pos = self._labels[self._set_opposite_position(self._position)]
            # 2. labels of each site are fixed but the goal position move to the opposite site
            # 3. label of goal position is exchanged with the label of the opposite site. but the goal position is fixed
        return pos
    
    @property
    def prev_position(self):
        if isinstance(self._prev_position, int):
            return -1
        else:
            return self._labels[self._prev_position]

    @property
    def goal(self):
        goal = self._labels[self._goal]
        if self._episode_steps >= self._max_episode_steps // 2 and self._reverse_mode in [1, 2]:
            # 1. Position of goal and the label of position move to the opposite site
            # 2. labels of each site are fixed but the goal position move to the opposite site
            # 3. label of goal position is exchanged with the label of the opposite site. but the goal position is fixed
            goal = self._labels[self._set_opposite_position(self._goal)]
        return goal

    @property
    def previous_action(self):
        return self._prev_action

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def is_respawn(self):
        return self._is_respawn

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def maze_size(self):
        return self._maze_size

    @property
    def oracle_min_num_actions(self):
        return self._maze_size * 2 - 1
        # return self._oracle_min_num_actions

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pygame.font.init()
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + self.window_size / self._maze_size + 50)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create a separate surface for the grid and the title.
        grid_canvas = pygame.Surface((self.window_size, self.window_size + self.window_size / self._maze_size))
        grid_canvas.fill((255, 255, 255))

        title_canvas = pygame.Surface((self.window_size, 50))
        title_canvas.fill((255, 255, 255))

        pix_square_size = (
            self.window_size / self._maze_size
        )  # The size of a single grid square in pixels

        letters = string.ascii_uppercase + string.ascii_lowercase
        labels = {n: letters[self._labels[n]] for n in self._graph.nodes()}

        node_list = list(self._graph.nodes())

        font = pygame.font.Font(
            None, int(pix_square_size)
        )  # font size depends on square size

        for n in node_list:
            color = (0, 0, 0)  # default color
            if n == self._position:
                color = (173, 216, 230)  # lightblue
            elif n == self._goal:
                color = (144, 238, 144)  # lightgreen
            else:
                color = (255, 105, 180)  # pink

            pygame.draw.rect(
                grid_canvas,
                color,
                pygame.Rect(
                    int(pix_square_size * n[1]),
                    int(pix_square_size * n[0]),
                    int(pix_square_size),
                    int(pix_square_size),
                ),
            )

            # Draw label
            label_surface = font.render(
                labels[n], True, (0, 0, 0)
            )  # Black color for text
            grid_canvas.blit(
                label_surface, (pix_square_size * n[1], pix_square_size * n[0])
            )

        font = pygame.font.Font(None, 24)  # Adjust size as needed
        text = font.render(
            "Action: {}, Reward: {}, Step: {}".format(self.previous_action, self.episode_reward, self._episode_steps),
            True,
            (0, 0, 0),
        )

        # Get the width of the text
        text_width = text.get_width()

        # Blit the text onto the title_canvas at the right side
        title_canvas.blit(
            text, (self.window_size - text_width - 10, 10)
        )  # 10px padding from the right

        if self.render_mode == "human":
            # Blit both surfaces onto the window.
            self.window.blit(title_canvas, (0, 0))
            self.window.blit(grid_canvas, (0, 50))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def generate_worlds(num_maze, maze_size, num_arms, seed=None, **kwargs):
        rng = np.random.default_rng(seed)
        sequence = np.arange((num_arms + 1) * maze_size)
        labels = []
        for _ in range(num_maze):
            labels.append(rng.permutation(sequence))

        # Initialize fixed goals
        goals = []
        for i in range(num_arms):
            fixed_goal = tuple([i + 1, 0])
            goals.append(fixed_goal)
        return {"labels": labels, "goals": goals}


def _min_actions(x1, y1, x2, y2, K):
    dx = min(abs(x1 - x2), K - abs(x1 - x2))
    dy = min(abs(y1 - y2), K - abs(y1 - y2))
    min_steps = dx + dy
    return min_steps + 1


def _sum_min_actions(K):
    total_min_actions = 0
    for x1 in range(K):
        for y1 in range(K):
            for x2 in range(K):
                for y2 in range(K):
                    if x1 != x2 or y1 != y2:
                        total_min_actions += _min_actions(x1, y1, x2, y2, K)
    return total_min_actions


def _oracle_min_actions(K):
    sum_min_actions_for_all_combinations = _sum_min_actions(K)
    return sum_min_actions_for_all_combinations / (K**2 * (K**2 - 1))
