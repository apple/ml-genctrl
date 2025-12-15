# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import logging
import math
from collections.abc import Iterable
from typing import Optional

import numpy as np
from torch.distributions.distribution import Distribution

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Controllability:
    def __init__(
        self,
        reachability_verifier,
        partially_controllable: float = 0.99,
        confidence: float = 0.05,
        error: float = 0.05,
    ) -> None:
        """
        Initialize the controllability class with a reachability verifier.
        :param reachability_verifier: An instance of a reachability verifier.
        :param partially_controllable: Partial controllability parameter. "I'm satisfied if /alpha/ propotion of states can reach the target set."
        :param confidence: With probability 1 - confidence, the controllable set is epsilon away from the true set.

        The reachability verifier should have the control system as an attribute.
        It should also have the pmin, quantization, and confidence specified.

        Reachability:
            attributes:
                system: ControlSystem,
                input_distribution: Distribution for inputs
                num_samples: number of samples for reachable set.

            methods:
                get_reachable_set(x0, T, optional(inputs))
        """
        self.reachability_verifier = reachability_verifier
        self.partially_controllable = partially_controllable
        self.confidence = (
            confidence  # The overall target confidence, NOT the delta term in Thm 3.19.
        )
        self.error = error

        # Calculate the number of samples needed to ensure the controllability condition is met
        self.optimal_sample_budget = self.__find_optimal_sample_budget(
            target_confidence=self.confidence,
            error=self.error,
            partially_controllable=self.partially_controllable,
            p_min=self.reachability_verifier.p_min,
            N=self.reachability_verifier.covering_number,
        )

        self.num_samples = self.optimal_sample_budget[
            "num_initial_state_samples"
        ]  # k in Thm 3.19

        # Set the confidence for the reachability verifier (it automatically adjusts the number of samples needed)
        self.reachability_verifier.set_confidence(
            self.optimal_sample_budget["delta_reachability"]
        )

        logger.info(
            f"Number of init. state samples required for controllability: {self.num_samples}"
        )
        logger.info(
            f"Number of total samples required for controllability: {self.optimal_sample_budget['total_sample_complexity']}"
        )

    def __find_optimal_sample_budget(
        self, target_confidence, error, partially_controllable, p_min, N
    ):
        """
        Find the optimal sample budget for controllability based on the target confidence, error, and reachability parameters.
        :param target_confidence: Target confidence level for the controllable set.
        :param error: Error margin for the controllable set.
        :param partially_controllable: Partial controllability parameter.
        :param p_min: Minimum probability of reaching a bin in the quantized target set.
        :param N: Covering number of the reachable set.
        :return: A dictionary with optimal sample budget parameters.
        """
        # Set up the search grid
        delta_C_vals = np.geomspace(
            1e-6, 1 - 1e-3, 300
        )  # confidence from the initial states
        delta_R_vals = np.geomspace(
            1e-6, 1 - 1e-3, 300
        )  # confidence from the reachable set of a single state

        # Log the best number of samples
        best_n = float("inf")
        best_tuple = None

        log1_alpha = math.log(1 - partially_controllable)
        log1_pmin = math.log(1 - p_min)

        for delta_C in delta_C_vals:
            for delta_R in delta_R_vals:
                # Compute k (must be integer ≥ formula)
                try:
                    log_eps_delta = math.log(error * delta_C)
                    k = math.ceil(log_eps_delta / log1_alpha)
                except (ValueError, ZeroDivisionError):
                    continue  # invalid log domain

                # Check joint confidence constraint
                joint_conf = (1 - delta_R) ** k * (1 - delta_C)
                if joint_conf < 1 - target_confidence:
                    continue  # doesn't satisfy constraint

                # Compute m
                try:
                    log_term = math.log(delta_R / N)
                    m = math.ceil(log_term / log1_pmin)
                except (ValueError, ZeroDivisionError):
                    continue  # invalid log domain

                n = m * k
                if n < best_n:
                    best_n = n
                    best_tuple = {
                        "delta_controllability": delta_C,
                        "delta_reachability": delta_R,
                        "num_initial_state_samples": k,
                        "num_reachable_samples": m,
                        "total_sample_complexity": n,
                    }

        if best_tuple is None:
            raise ValueError("No valid sample budget found. Check the parameters.")

        return best_tuple

    # Public methods=========================================================================================================

    @classmethod
    def from_reachability_problem(
        cls,
        reachability_verifier,
        sampling_budget: int = None,
        partially_controllable: float = 0.1,
        confidence: float = 0.05,
        error: float = 0.05,
    ):
        """
        Initialize the controllability class from a target confidence.
        :param reachability_verifier: An instance of a reachability verifier.
        :param sampling_budget: The number of samples available for the controllability check.
        :param partially_controllable: Partial controllability parameter.
        :param confidence: Confidence level for the controllable set.
        :param error: Error margin for the controllable set.
        """
        return cls(reachability_verifier, partially_controllable, confidence, error)

    def sample_initial_states_exhaustive(
        self,
        initial_state_space: Iterable,
        time_horizon: int,
        input_distribution: Distribution,
    ):
        """
        This helper function is only used if the number of initial states is less than the number of samples.
        It exhaustively samples all initial states from the initial state space.
        :param initial_state_space: Iterable of initial states.
        :param time_horizon: Time horizon for the reachability check.
        :param input_distribution: Input distribution to use.
        :return: A tuple of (reachable_tubes, reachable_sets, inputs_used_all, points_reached_all, states, stop_conditions)
        """
        (
            reachable_tubes,
            reachable_sets,
            inputs_used_all,
            points_reached_all,
            states,
            stop_conditions,
        ) = (
            {},
            {},
            {},
            {},
            {},
            {},
        )  # dict: {initial_state: reached}

        # Collect all the reachable sets for each initial state
        for x0 in initial_state_space:
            (
                reachable_tube_trajectories,
                reachable_set_trajectories,
                inputs_used_trajectories,
                points_reached_trajectories,
                state_trajectories,
                stop_trajectories,
            ) = self.reachability_verifier.get_reachable_set(
                x0, time_horizon, input_distribution
            )

            if x0 not in points_reached_all:
                # Add the state to the log
                reachable_tubes[x0] = []  # eventual length = t
                reachable_sets[x0] = []
                inputs_used_all[x0] = []
                states[x0] = []
                stop_conditions[x0] = []
                points_reached_all[x0] = []

                # Log each timestep
                for t in range(time_horizon + 1) if time_horizon > 1 else [0]:
                    reachable_tubes[x0].append(reachable_tube_trajectories[t])
                    reachable_sets[x0].append(reachable_set_trajectories[t])
                    points_reached_all[x0].append(points_reached_trajectories[t])
                    inputs_used_all[x0].append(inputs_used_trajectories[t])
                    states[x0].append(state_trajectories[t])
                    stop_conditions[x0].append(stop_trajectories[t])

            # We hit the same x0 twice
            elif x0 in points_reached_all:
                for t in range(time_horizon + 1) if time_horizon > 1 else [0]:
                    points_reached_all[x0][t].extend(points_reached_trajectories[t])
                    inputs_used_all[x0][t].extend(inputs_used_trajectories[t])
                    reachable_sets[x0][t] = self.reachability_verifier.union(
                        [reachable_sets[x0][t], reachable_set_trajectories[t]]
                    )
                    reachable_tubes[x0][t] = self.reachability_verifier.union(
                        [reachable_tubes[x0][t], reachable_tube_trajectories[t]]
                    )
                    states[x0][t].extend(state_trajectories[t])
                    stop_conditions[x0][t].extend(stop_trajectories[t])

        return (
            reachable_tubes,
            reachable_sets,
            inputs_used_all,
            points_reached_all,
            states,
            stop_conditions,
        )

    def sample_initial_states_random(
        self,
        initial_state_distribution: Distribution,
        time_horizon: int,
        input_distribution: Distribution,
    ):
        """
        Sample initial states from a distribution and compute the reachable sets.
        :param initial_state_distribution: A distribution to sample initial states from.
        :param time_horizon: Time horizon for the reachability check.
        :param input_distribution: Input distribution to use.
        :return: A tuple of (reachable_tubes, reachable_sets, inputs_used_all, points_reached_all, states, stop_conditions)
        """
        (
            reachable_tubes,
            reachable_sets,
            inputs_used_all,
            points_reached_all,
            states,
            stop_conditions,
        ) = (
            {},
            {},
            {},
            {},
            {},
            {},
        )  # dict: {initial_state: reached}

        # Check if the distribution has finite support
        support = initial_state_distribution.enumerate_support()
        if support is not None and len(support) < self.num_samples:
            logger.warning(
                f"Initial state distribution has only {len(support)} states but "
                f"{self.num_samples} samples requested. Will collect all {len(support)} unique states."
            )

        for _ in range(self.num_samples):
            # Check if we've exhausted all available states from finite support
            if support is not None and len(points_reached_all) >= len(support):
                logger.info(f"Exhausted all {len(support)} unique initial states. Stopping early.")
                break

            x0 = (
                initial_state_distribution.sample()
            )  # Sample an initial state from the distribution

            # Sample until we get a new initial state
            while x0 in points_reached_all:
                x0 = initial_state_distribution.sample()

            (
                reachable_tube_trajectories,
                reachable_set_trajectories,
                inputs_used_trajectories,
                points_reached_trajectories,
                state_trajectories,
                stop_trajectories,
            ) = self.reachability_verifier.get_reachable_set(
                x0, time_horizon, input_distribution
            )

            if x0 not in points_reached_all:
                # Add the state to the log
                reachable_tubes[x0] = []  # eventual length = t
                reachable_sets[x0] = []
                inputs_used_all[x0] = []
                states[x0] = []
                stop_conditions[x0] = []
                points_reached_all[x0] = []

                # Log each timestep
                for t in range(time_horizon + 1) if time_horizon > 1 else [0]:
                    reachable_tubes[x0].append(reachable_tube_trajectories[t])
                    reachable_sets[x0].append(reachable_set_trajectories[t])
                    points_reached_all[x0].append(points_reached_trajectories[t])
                    inputs_used_all[x0].append(inputs_used_trajectories[t])
                    states[x0].append(state_trajectories[t])
                    stop_conditions[x0].append(stop_trajectories[t])

            # We hit the same x0 twice
            elif x0 in points_reached_all:
                for t in range(time_horizon + 1) if time_horizon > 1 else [0]:
                    points_reached_all[x0][t].extend(points_reached_trajectories[t])
                    inputs_used_all[x0][t].extend(inputs_used_trajectories[t])

                    reachable_sets[x0][t] = self.reachability_verifier.union(
                        [reachable_sets[x0][t], reachable_set_trajectories[t]]
                    )
                    reachable_tubes[x0][t] = self.reachability_verifier.union(
                        [reachable_tubes[x0][t], reachable_tube_trajectories[t]]
                    )
                    states[x0][t].extend(state_trajectories[t])
                    stop_conditions[x0][t].extend(stop_trajectories[t])

        return (
            reachable_tubes,
            reachable_sets,
            inputs_used_all,
            points_reached_all,
            states,
            stop_conditions,
        )

    def get_controllable_set(
        self,
        initial_state_distribution: Distribution,
        T: int,
        input_distribution: Optional[Distribution] = None,
    ) -> tuple[dict, Iterable, dict, dict]:
        """
        Compute the controllable set from the initial state distribution over time horizon T.
        :param initial_state_distribution: Distribution over initial states.
        :param T: Time horizon.
        :param input_distribution: Optional input distribution to consider.
        :return: A tuple of (controllable_tubes, reachable_tubes, reachable_sets, inputs_used, points_reached, states, stop_conditions)
        """
        # Get the size of the initial state space. Initial states are string-valued, so we can take the length
        initial_state_space = initial_state_distribution.enumerate_support()

        if initial_state_space is not None:
            initial_state_space_size = len(initial_state_space)
        else:
            initial_state_space_size = (
                np.inf
            )  # the initial state distn is LLM valued, not categorical.

        # If the number of initial states is less than the number of samples, we can exhaustively sample all initial states
        if initial_state_space_size <= self.num_samples:
            (
                reachable_tubes,
                reachable_sets,
                inputs_used_all,
                points_reached_all,
                states,
                stop_conditions,
            ) = self.sample_initial_states_exhaustive(
                initial_state_space, T, input_distribution
            )
        else:
            # Otherwise sample initial states from the distribution
            (
                reachable_tubes,
                reachable_sets,
                inputs_used_all,
                points_reached_all,
                states,
                stop_conditions,
            ) = self.sample_initial_states_random(
                initial_state_distribution, T, input_distribution
            )

        # Take the intersection to get the controllable set (the implementation depends on the problem-- discrete vs quantized)
        controllable_tubes = []
        for t in range(T):
            reached_tubes_t = [reachable_tubes[x0][t] for x0 in reachable_tubes]
            controllable_tube = self.reachability_verifier.intersection(reached_tubes_t)
            controllable_tubes.append(controllable_tube)

        return (
            controllable_tubes,
            reachable_tubes,
            reachable_sets,
            inputs_used_all,
            points_reached_all,
            states,
            stop_conditions,
        )
