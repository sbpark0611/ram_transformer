from typing import Callable

import torch
import numpy as np


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def eval_memory_correct(logits, target, pos, start, stop, mlen):
    visited_pos = pos[:, max(0, start - mlen) : stop]
    c = visited_pos[..., None, None] == pos[:, stop : stop + 1]
    _visited_before = c.any(1).squeeze(-1).diagonal()
    _unvisited_before = (~c).all(1).squeeze(-1).diagonal()
    visited_correct = torch.sum(
        (
            torch.argmax(logits[_visited_before], dim=-1) == target[_visited_before]
        ).float()
    ).item()
    unvisited_correct = torch.sum(
        (
            torch.argmax(logits[_unvisited_before], dim=-1) == target[_unvisited_before]
        ).float()
    ).item()
    correct = torch.sum((torch.argmax(logits, dim=-1) == target).float()).item()
    return {
        "visited_correct": visited_correct,
        "unvisited_correct": unvisited_correct,
        "correct": correct,
        "visited_tot": _visited_before.sum(),
        "unvisited_tot": _unvisited_before.sum(),
    }


def levy_walk(n_steps, batch_size, mu=1.1):
    # Define the actions
    actions = np.array([1, 2, 3, 4])  # 1:Right, 2:Down, 3:Up, 4:Left
    stay = 0  # Stay

    # Initialize an empty array for the action sequences
    action_sequences = np.empty((batch_size, n_steps), dtype=int)

    for i in range(batch_size):
        # Generate a random sequence of actions
        action_sequence = []
        previous_action = None
        while len(action_sequence) < n_steps:  # Generate n_steps steps
            # Choose a random action
            action = np.random.choice(actions)

            # If the action changes, add a "stay" action
            if action != previous_action and previous_action is not None:
                action_sequence.append(stay)

            # Choose a random step length from the Lévy distribution
            step_length = int(np.random.pareto(mu) + 1)

            # Add the action to the sequence
            for _ in range(step_length):
                action_sequence.append(action)

            # Update the previous action
            previous_action = action

        # Store the action sequence in the action sequences array
        action_sequences[i] = action_sequence[:n_steps]

    return action_sequences
