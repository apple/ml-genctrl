# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import logging
import math
from abc import abstractmethod
from collections.abc import Iterable
from pprint import pprint
from typing import Optional

import numpy as np
from torch.distributions.distribution import Distribution

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Local imports
from genctrl.systems.control_system import ControlSystem
from genctrl.utils.utils import (
    intersect_two_lists,
    propagate_reachable_trajectory,
)


class Reachability:
    def __init__(
        self, system: ControlSystem, input_distribution: Optional[Distribution]
    ) -> None:
        """
        Initialize the Reachability class.
        """
        self.system = system

        if input_distribution is None:
            # If no input space is provided, use the system's whole input space
            self.input_distribution = system.input_distribution
        else:
            self.input_distribution = input_distribution

        self.num_samples = (
            None  # This needs to be populated for each type of reachability problem.
        )

    def __str__(self):
        """
        String representation of the Reachability class.
        """
        string = f"Reachability class for\nControl System: {str(self.system)}\nInput space: {str(self.input_distribution)}"
        return string

    def __sample_u(
        self,
        override_inputs: Optional[Distribution] = None,
        timestep: int = 0,
        **extra_variables,
    ):
        """
        Sample the input space uniformly.
            :return: Sampled input
        """
        # Sample the input space uniformly
        input_distribution = (
            self.input_distribution if override_inputs is None else override_inputs
        )
        return input_distribution.sample(timestep=timestep, **extra_variables)

    def __MC_single_step(self, x0, override_inputs: Optional[Distribution] = None):
        """
        Monte Carlo simulation to check reachability.
            :param x0: Initial state
            :param override_inputs: Optional input distribution override
            :return: Next state and output
        """
        # Sample the input space uniformly for a single step
        Y = set()  # report the output space hits
        X_next = set()  # report the next state hits

        # Sample the input space
        Us = [self.__sample_u(override_inputs) for _ in range(self.num_samples)]

        # System dynamics, single step
        X_next, Y = self.system.forward([x0] * self.num_samples, Us)

        return X_next, Y

    @abstractmethod
    def _calculate_number_of_samples(self) -> int:
        """
        Calculate the number of samples needed for the reachability check.
        Each subclass should implement this method based on the appropriate sample complexity
        bound for the reachability problem type.
        """
        pass

    # Public methods=========================================================================================================

    @classmethod
    def from_problem_type(
        cls,
        problem_type: str,
        system: ControlSystem,
        input_distribution: Optional[Distribution] = None,
        **kwargs,
    ) -> "Reachability":
        """
        Factory method to create a Reachability instance based on the problem type.
            :param problem_type: Type of reachability problem (e.g., "quantized", "discrete", "continuous")
            :param system: ControlSystem instance
            :param input_distribution: Optional input distribution
            :return: Reachability instance
        """
        if problem_type == "quantized":
            return QuantizedReachability(system, input_distribution, **kwargs)
        elif problem_type == "discrete":
            return DiscreteReachability(system, input_distribution, **kwargs)
        elif problem_type == "continuous":
            return ContinuousReachability(system, input_distribution, **kwargs)
        else:
            raise ValueError(f"Unknown reachability problem type: {problem_type}")

    def set_confidence(self, target_confidence: float):
        """
        Set the confidence parameter for the reachability check and updates the number of samples needed.
            :param target_confidence: Target confidence level
        """
        self.confidence = target_confidence
        self.num_samples = type(self)._calculate_number_of_samples(self)
        logger.info(
            f"Updated number of samples needed for reachability confidence {target_confidence}: {self.num_samples}"
        )

    def get_reachable_set(
        self, x0, T: int, input_distribution: Optional[Distribution] = None
    ) -> tuple[list, list, list, list]:
        """
        Get the reachable set from the initial state x0 after T time steps.
            :param x0: Initial state.
            :param T: Time horizon
            :param input_distribution: Optional input distribution.
            :return: Reachable set and the inputs used to get them.
        """
        # Check if the system is reachable from one state to another

        # If the system is deterministic and the input space is small, exhaustively sample the input space.
        exhaustive_sample = False
        input_distribution = (
            input_distribution
            if input_distribution is not None
            else self.input_distribution
        )
        initial_input_space = input_distribution.enumerate_support(current_time_step=0)

        # Enumerate initial input space
        if isinstance(initial_input_space, Iterable):
            exhaustive_sample_size = len(initial_input_space)

            # If we're in feedback mode and the feedback is stochastic, sample.
            if T > 1:
                if any(
                    [
                        input_distribution.is_stochastic(current_time_step=t)
                        for t in range(1, T)
                    ]
                ):
                    # If the feedback is stochastic at any point
                    exhaustive_sample_size = np.inf

        # Unenumerable input space
        elif initial_input_space is None:
            exhaustive_sample_size = np.inf  # Continuous

        # If small input space and deterministic, exhaustively sample the initial input space
        exhaustive_sample = (
            exhaustive_sample_size <= self.num_samples and not self.system.stochastic
        )
        t = 0
        if exhaustive_sample:
            num_samples = exhaustive_sample_size
            Us = [(inp, False) for inp in initial_input_space]
            logger.info(
                "Deterministic system with small input space, exhaustively sampling the input space with "
                + str(num_samples)
                + " samples."
            )
        else:
            # Sample the input space uniformly
            num_samples = self.num_samples
            Us = [
                (self.__sample_u(input_distribution, timestep=t), False)
                for _ in range(num_samples)
            ]

        # System dynamics, T steps
        X_next = [x0] * num_samples
        U0, stop_next = zip(*Us)  # avoid overwriting
        U0, stop_next = list(U0), list(stop_next)

        # Collect conversation traces and start dynamics
        Y_trajectories = [[None] * num_samples]
        U_trajectories = [U0]
        X_trajectories = [X_next.copy()]
        stop_trajectories = [stop_next]

        U_next = U0

        for t in range(T):
            X_next, Y = self.system.forward(X_next, U_next, t=t)
            # Update the state and next Us

            if T > 1:
                # Define extra_variables for feedback
                extra_variables = {"initial_inputs": U0, "last_outputs": Y}

                # Get the relevant variables
                feedback = self.__sample_u(
                    input_distribution, timestep=t + 1, **extra_variables
                )
                U_next, stops = list(zip(*feedback))
                U_next = list(U_next)
                stops = list(stops)

                # Collect
                stop_trajectories.append(stops)
                U_trajectories.append(U_next.copy())
                X_trajectories.append(X_next.copy())
                Y_trajectories.append(Y.copy())
            else:
                Y_trajectories[0] = Y

        return Y_trajectories, U_trajectories, X_trajectories, stop_trajectories

    def reachable(self, x0, to: set, T: int, U: Optional[Distribution]) -> bool:
        """
        Check if the system is reachable from one state to another.
            :param x0: Initial state.
            :param to: Target set of states.
            :param T: Time horizon
            :param U: Input space (set of inputs)
            :return: True if reachable, False otherwise
        """
        reachable_set, *_ = self.get_reachable_set(x0, T, U)
        return to.issubset(reachable_set)

    def to_dict(self):
        """
        Convert the Reachability class to a dictionary.
            :return: Dictionary representation of the Reachability class
        """
        return {
            "input_space": (  # Key name kept as 'input_space' for backward compatibility
                str(self.input_distribution)
                if self.input_distribution is not None
                else None
            ),
            "m": self.num_samples,  # Number of samples needed for the reachability check
        }


class QuantizedReachability(Reachability):
    def __init__(
        self,
        system: ControlSystem,
        input_distribution: Optional[Distribution],
        gamma: float,
        p_min: Optional[float] = 0.01,
        confidence: Optional[float] = 0.01,
        **kwargs,
    ) -> None:
        """
        Initialize the QuantizedReachability class.
            :param system: ControlSystem
            :param input_distribution: Input distribution
            :param gamma: Quantization parameter (gamma-ball)
            :param p_min: User-specified threshold for nontrivial probability of a bin.
            :param confidence: Confidence parameter (with confidence 1-confidence, the reachable set is epsilon-accurate) (delta)
        """
        super().__init__(system, input_distribution)

        # Parameters (Emily's Theorem 3.11, quantization)
        self.gamma = gamma
        self.p_min = (
            p_min  # User-specified threshold for nontrivial probability of a bin.
        )
        self.covering_number = self.__compute_covering_number()  # If we have access to the whole output range, then we can compute the covering number. Otherwise, fill it in when computing the reachable set.
        self.confidence = confidence  # Confidence parameter (with confidence 1-confidence, the reachable set is epsilon-accurate) (delta)

        # Calculate sample complexity (Emily's Theorem 3.11)
        self.num_samples = (
            self._calculate_number_of_samples()
        )  # m is the number of samples needed for the reachability check

        logger.info("Number of samples needed:" + str(self.num_samples))

    # Private methods==================================================================================
    def __compute_covering_number(self) -> int:
        """
        Compute the covering number for the output space.
        Currently, only 1-dimensional output ranges are supported! # TODO: Extend to multi-dimensional output ranges.
            :return: Gamma covering number
        """
        # Checks on output_range
        output_range = self.system.output_space  # Assuming the system has an output_space attribute that is a collection of intervals.
        assert len(output_range) > 0, "Output range must not be empty."
        assert all(
            isinstance(interval, Iterable) and len(interval) == 2
            for interval in output_range
        ), (
            "Output range must be a collection of ordered pairs denoting compact intervals."
        )
        assert all(interval[0] < interval[1] for interval in output_range), (
            "Output range intervals must be ordered pairs with the first element less than the second."
        )

        # Compute the covering number for the output space
        # The covering number is the sum of the lengths of the intervals divided by gamma.
        return int(
            sum(
                [
                    math.ceil((interval[1] - interval[0]) / self.gamma)
                    for interval in output_range
                ]
            )
        )

    def _calculate_number_of_samples(self) -> int:
        """
        Calculate the number of samples needed for the reachability check.
            :return: Number of samples needed
        """
        # Use Emily's Theorem 3.11 bound for overapproximating interval
        m = max(
            self.covering_number,
            1
            / np.log(1 - self.p_min)
            * (np.log(self.confidence) - np.log(self.covering_number)),
        )
        return int(np.ceil(m))

    # Public methods=========================================================================================================
    @staticmethod
    def compute_gamma_cover(points_reached: Iterable, gamma: float) -> Iterable:
        intervals = [[point - gamma, point + gamma] for point in points_reached]

        # Put the gamma cover in order by first element
        intervals.sort(key=lambda x: x[0])

        # Take the union of the intervals
        merged = []
        for interval in intervals:
            if not len(merged) or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # Overlap: merge current interval with the last one in merged
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

    @staticmethod
    def union(*args) -> Iterable:
        """
        Compute the union of multiple reachable sets.
        Intervals are lists of [[start, end], [start, end], ...]
            :param args: reachable sets to union. Each reachable set is a list of length-2 intervals.
            :return: Union of the sets
        """
        # Accept either a list of intervals or multiple interval args
        if len(args) == 1 and isinstance(args[0], Iterable):
            reachable_sets = list(args[0])
        else:
            reachable_sets = list(args)

        if not reachable_sets:
            return []

        # We are given a list of lists of intervals, flatten them and then union them
        intervals_flattened = []
        for reachable_set in reachable_sets:
            if reachable_set is not None:
                if type(reachable_set) == list:
                    intervals_flattened.extend(reachable_set)

        intervals = intervals_flattened

        # Sort intervals by start
        if not len(intervals):
            return []
        print(intervals)
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]

        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # overlap or touch
                # merge them
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current)

        return merged

    @staticmethod
    def intersection(*args):
        """
        Compute the intersection of multiple sets.
            :param args: Sets to intersect
            :return: Intersection of the sets
        """
        if len(args) == 1 and isinstance(args[0], Iterable):
            reachable_intervals = args[0]
        else:
            reachable_intervals = args

        current = reachable_intervals[0]

        for other in reachable_intervals[1:]:
            current = intersect_two_lists(current, other)
            if not current:
                break
        return current

    def get_reachable_set(
        self, x0, T: int, input_distribution: Optional[Distribution] = None
    ) -> tuple[Iterable, Iterable, Iterable, Iterable, Iterable, Iterable]:
        """
        Get the reachable set from the initial state x0 after T time steps.
            :param x0: Initial state.
            :param T: Time horizon
            :param input_distribution: Optional input distribution.

            :return: Reachable set, inputs used to get them, and points reached.
        """
        (
            points_reached_trajectories,
            inputs_used_trajectories,
            X_trajectories,
            stop_trajectories,
        ) = super().get_reachable_set(x0, T, input_distribution)

        # Edit the points reached trajectories to take into account stopping criteria
        points_reached_edited_trajectories = propagate_reachable_trajectory(
            points_reached_trajectories, stop_trajectories
        )

        # Now compute the gamma-cover of points reached
        gamma_reachable_sets = []

        for points_reached_t in points_reached_edited_trajectories:
            points_reached_t = [
                point for point in points_reached_t if point is not None
            ]  # Remove None values-- None only occurs at the first time step.
            gamma_reachable_set_t = QuantizedReachability.compute_gamma_cover(
                points_reached_t, self.gamma
            )
            gamma_reachable_sets.append(gamma_reachable_set_t)

        # Now we compute the running reachable set because that's what we're after
        if T > 1:
            reachable_tubes = [gamma_reachable_sets[0], gamma_reachable_sets[1]]
            for i in range(2, len(gamma_reachable_sets)):
                gamma_reachable_set_t = gamma_reachable_sets[i]
                if not len(reachable_tubes):
                    # First reachable set
                    reachable_tubes.append(gamma_reachable_set_t)
                else:
                    reachable_tubes.append(
                        QuantizedReachability.union(
                            [gamma_reachable_set_t, reachable_tubes[-1]]
                        )
                    )
        else:
            reachable_tubes = [gamma_reachable_sets[0]]

        return (
            reachable_tubes,
            gamma_reachable_sets,
            inputs_used_trajectories,
            points_reached_trajectories,
            X_trajectories,
            stop_trajectories,
        )


class DiscreteReachability(Reachability):
    def __init__(
        self,
        system: ControlSystem,
        input_distribution: Optional[Distribution],
        p_min: Optional[float] = 0.01,
        confidence: Optional[float] = 0.01,
        **kwargs,
    ) -> None:
        """
        Initialize the DiscreteReachability class.
            :param system: ControlSystem
            :param input_distribution: Input distribution
            :param p_min: User-specified threshold for nontrivial probability of a bin.
            :param confidence: Confidence parameter (with confidence 1-confidence, the reachable set is epsilon-accurate) (delta)
        """
        super().__init__(system, input_distribution)

        # Parameters (Emily's Theorem 3.11, quantization)
        self.p_min = (
            p_min  # User-specified threshold for nontrivial probability of a bin.
        )
        self.covering_number = self.__compute_covering_number()  # If we have access to the whole output range, then we can compute the covering number. Otherwise, fill it in when computing the reachable set.
        self.confidence = confidence  # Confidence parameter (with confidence 1-confidence, the reachable set is epsilon-accurate) (delta)

        # Calculate sample complexity (Emily's Theorem 3.11)
        self.num_samples = (
            self._calculate_number_of_samples()
        )  # m is the number of samples needed for the reachability check
        logger.info("Number of samples needed:" + str(self.num_samples))

    # Private methods==================================================================================
    def __str__(self):
        """
        String representation of the DiscreteReachability class.
        """
        string = f"DiscreteReachability class for\nControl System: {str(self.system)}\nInput distribution: {str(self.input_distribution)}"
        return string

    def _calculate_number_of_samples(self) -> int:
        """
        Calculate the number of samples needed for the reachability check.
            :return: Number of samples needed
        """
        # Use Emily's Theorem 3.11 bound for overapproximating interval
        m = max(
            self.covering_number,
            1
            / np.log(1 - self.p_min)
            * (np.log(self.confidence) - np.log(self.covering_number)),
        )
        return int(np.ceil(m))

    def __compute_covering_number(self) -> int:
        """
        Compute the covering number for the output space, where the output space is
        a discrete set of categories.
            :return: Covering number (the length of the output range)
        """
        # Compute the covering number for the output space
        output_range = self.system.output_space  # Assuming the system has an output_space that is a set of discrete categories.
        assert len(output_range) > 0, "Output range must not be empty."
        if isinstance(output_range, set):
            output_range = list(output_range)
        if isinstance(output_range[0], list):
            output_range = map(tuple, output_range)
        return len(set(output_range))

    @staticmethod
    def union(sets):
        """
        Compute the union of multiple sets.
            :param sets: Sets to union
            :return: Union of the sets
        """
        sets = [set(s) for s in sets]
        combined = list(set.union(*sets))

        # Check if numeric, if so, sort.
        if (
            combined  # Check not empty first
            and isinstance(combined[0], (int, float))
            and not isinstance(combined[0], bool)
        ):
            return sorted(combined)
        else:
            return combined

    @staticmethod
    def intersection(sets):
        """
        Compute the intersection of multiple sets.
            :param sets: Sets to intersect
            :return: Intersection of the sets
        """
        sets = [set(s) for s in sets]
        return list(set.intersection(*sets))

    def get_reachable_set(
        self, x0, T: int, input_distribution: Optional[Distribution] = None
    ) -> tuple[Iterable, Iterable, Iterable, Iterable, Iterable, Iterable]:
        """
        Get the reachable set from the initial state x0 after T time steps.
            :param x0: Initial state.
            :param T: Time horizon
            :param input_distribution: Optional input distribution.

            :return: Reachable set, inputs used to get them, and points reached.
        """
        (
            points_reached_trajectories,
            inputs_used_trajectories,
            X_trajectories,
            stop_trajectories,
        ) = super().get_reachable_set(x0, T, input_distribution)

        # Edit the points reached trajectories to take into account stopping criteria
        points_reached_edited_trajectories = propagate_reachable_trajectory(
            points_reached_trajectories, stop_trajectories
        )

        # Construct the time-indexed reachable sets
        reachable_sets = [
            set(points_reached_t)
            for points_reached_t in points_reached_edited_trajectories
        ]
        for i, _ in enumerate(reachable_sets):
            reachable_sets[i].discard(None)  # get rid of empty

        # Construct the reachable tube
        reachable_tubes = [reachable_sets[0]]
        for t in range(1, len(reachable_sets)):
            reachable_set_t = reachable_sets[t]
            reachable_tubes.append(
                set(DiscreteReachability.union([reachable_set_t, reachable_tubes[-1]]))
            )

        return (
            reachable_tubes,
            reachable_sets,
            inputs_used_trajectories,
            points_reached_trajectories,
            X_trajectories,
            stop_trajectories,
        )  # points reached are the same as reachable set in discrete reachability


class ContinuousReachability(Reachability):
    def __init__(
        self,
        system: ControlSystem,
        input_distribution: Optional[Distribution],
        epsilon: Optional[float],
        delta: float,
        output_dimensionality: int,
    ) -> None:
        """
        Initialize the ContinuousReachability class.
        :param system: ControlSystem
        :param input_distribution: Input distribution
        :param epsilon: Epsilon-accurate parameter for the reachability check
        :param delta: Confidence parameter (with confidence 1-delta, the reachable set is epsilon-accurate)
        :param output_dimensionality: Dimensionality of the state space (output space) (n)
        """
        super().__init__(system, input_distribution)

        # Parameters (DA19)
        self.eps = epsilon
        self.delta = delta
        self.output_dimensionality = output_dimensionality  # dimensionality of the state space (output space) (n)

        # Calculate number of samples needed per Devonport and Arcak, 2019
        self.num_samples = (
            self.__calculate_number_of_samples()
        )  # m is the number of samples needed for the reachability check
        logger.info("Number of samples needed:" + str(self.num_samples))

    # Private methods==================================================================================

    def __calculate_number_of_samples_deterministic(self) -> int:
        """
        Calculate the number of samples needed for the reachability check.
            :return: Number of samples needed
        """
        # Use Devonport and Arcak, 2019 bound for overapproximating interval
        m = (
            2
            * self.output_dimensionality
            / self.eps
            * np.log(2 * self.output_dimensionality / self.delta)
        )
        return int(np.ceil(m))

    def __calculate_number_of_samples_stochastic(self) -> int:
        """
        Calculate the number of samples needed for stochastic systems.
            :return: Number of samples needed
        """
        logger.error("Method not implemented")
        raise NotImplementedError(
            "Sample complexity for stochastic continuous reachability is not yet implemented"
        )

    def __calculate_number_of_samples(self) -> int:
        """
        Calculate the number of samples needed for the reachability check.
            :return: Number of samples needed
        """
        # Use Devonport and Arcak, 2019 bound for overapproximating interval
        if self.system.stochastic:
            m = self.__calculate_number_of_samples_stochastic()
        else:
            m = self.__calculate_number_of_samples_deterministic()

        return m

    # Public methods=========================================================================================================
    def get_reachable_set(
        self, x0, T: int, input_distribution: Optional[Distribution] = None
    ) -> tuple[Iterable, Iterable, Iterable]:
        """
        Get the reachable set from the initial state x0 after T time steps.
            :param x0: Initial state.
            :param T: Time horizon
            :param input_distribution: Optional input distribution.
            :return: Reachable set
        """
        # Run the sampling
        (
            points_reached_trajectories,
            inputs_used_trajectories,
            X_trajectories,
            stop_trajectories,
        ) = super().get_reachable_set(x0, T, input_distribution)

        try:
            reachable_set = [
                [min(points_reached_trajectories), max(points_reached_trajectories)]
            ]
        except TypeError:
            raise TypeError(
                "points_reached_trajectories might contain mixed types, cannot compute the reachable set."
            )

        # Take interval overapproximation of the reachable set
        return reachable_set, inputs_used_trajectories, points_reached_trajectories

    @staticmethod
    def intersection(*args):
        """
        Compute the intersection of multiple sets.
            :param args: Sets to intersect
            :return: Intersection of the sets
        """
        if len(args) == 1 and isinstance(args[0], Iterable):
            reachable_intervals = args[0]
        else:
            reachable_intervals = args

        current = reachable_intervals[0]

        for other in reachable_intervals[1:]:
            current = intersect_two_lists(current, other)
            if not current:
                break
        return current


if __name__ == "__main__":
    # Build the control system by defining the model and output mapping
    def even(numbers: Iterable[int]) -> list[bool]:
        return [number % 2 == 0 for number in numbers]

    y = even
    U = set(range(1, 10))  # input space
    X_out = set(range(1, 20))  # naive output space
    A = [True, False]  # output (attribute) space
    model = (
        lambda x, u: (
            x,
            np.random.choice(list(X_out), size=len(x), replace=True),
        )
    )  # model: (X_t, U) -> X_{t+1} (stochastically) maps the current state + input to the next state.
    system = ControlSystem(model, y)

    # Define the reachability verifier
    T = 1  # time horizon
    epsilon = 0.1  # confidence parameter
    delta = 0.1  # confidence parameter
    n = 1  # dimensionality of the state space (output space)
    verifier = Reachability(system, U, T, epsilon, delta, n)

    print("Testing verifier=======================")
    x0 = 1  # initial state
    x1, y1_dist = verifier.MC_single_step(1)

    print("Initial state: ", x0)
    print(f"With confidence p={delta}:")
    print(f"T={T} {epsilon}-reachable set includes ", set(y1_dist))
    print("Surjective? ", set(A) == set(y1_dist))
