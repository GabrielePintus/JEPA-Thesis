"""
Gymnasium Robotics Maze Environment Classes

This module defines the core classes for creating goal-conditioned maze environments
within the Gymnasium Robotics framework, specifically designed to integrate a discrete
maze map into a MuJoCo simulation. It includes the `Maze` class for handling
maze geometry and the `MazeEnv`/`PointMazeEnv` classes for implementing the
GoalEnv logic with a Point Mass agent.

"""

from os import path
import tempfile
import time
import math
from typing import Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.maze import maps
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.envs.maze.maps import COMBINED, GOAL, RESET, U_MAZE
import xml.etree.ElementTree as ET
from gymnasium_robotics.core import GoalEnv


class Maze:
    r"""This class creates and holds information about the maze in the MuJoCo simulation.

    The accessible attributes are the following:
    - :attr:`maze_map` - The maze discrete data structure (list of lists).
    - :attr:`maze_size_scaling` - Scaling factor for continuous coordinates in MuJoCo.
    - :attr:`maze_height` - The height of the walls in the MuJoCo simulation.
    - :attr:`unique_goal_locations` - All the `(x,y)` goal coordinates.
    - :attr:`unique_reset_locations` - All the `(x,y)` agent initialization coordinates.
    - :attr:`combined_locations` - All the `(x,y)` cell coordinates for goal and reset locations.
    - :attr:`map_length` - Number of rows (i index) in the maze map.
    - :attr:`map_width` - Number of columns (j index) in the maze map.
    - :attr:`x_map_center` - The x coordinate of the map's center.
    - :attr:`y_map_center` - The y coordinate of the map's center.

    The Maze class also presents methods to convert between cell indices and `(x,y)` coordinates:
    - :meth:`cell_rowcol_to_xy` - Convert from discrete cell `(i,j)` to continuous MuJoCo `(x,y)`.
    - :meth:`cell_xy_to_rowcol` - Convert from continuous MuJoCo `(x,y)` to discrete cell `(i,j)`.

    ### Version History
    * v4: Refactor compute_terminated into a pure function compute_terminated and a new function update_goal which resets the goal position. Bug fix: missing maze_size_scaling factor added in generate_reset_pos() -- only affects AntMaze.
    * v3: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v2 & v1: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]],
        maze_size_scaling: float,
        maze_height: float,
    ):
        """Initializes the Maze structure and calculates its center and dimensions."""

        self._maze_map = maze_map
        self._maze_size_scaling = maze_size_scaling
        self._maze_height = maze_height

        # Stores continuous MuJoCo (x,y) coordinates for goal/reset locations
        self._unique_goal_locations = []
        self._unique_reset_locations = []
        self._combined_locations = []

        # Get the center cell Cartesian position of the maze. This will be the origin
        self._map_length = len(maze_map)
        self._map_width = len(maze_map[0])
        self._x_map_center = self.map_width / 2 * maze_size_scaling
        self._y_map_center = self.map_length / 2 * maze_size_scaling

    @property
    def maze_map(self) -> List[List[Union[str, int]]]:
        """Returns the list[list] data structure of the maze."""
        return self._maze_map

    @property
    def maze_size_scaling(self) -> float:
        """Returns the scaling value used for continuous coordinates."""
        return self._maze_size_scaling

    @property
    def maze_height(self) -> float:
        """Returns the wall height in the MuJoCo simulation."""
        return self._maze_height

    @property
    def unique_goal_locations(self) -> List[np.ndarray]:
        """Returns all possible goal locations in continuous (x,y) coordinates."""
        return self._unique_goal_locations

    @property
    def unique_reset_locations(self) -> List[np.ndarray]:
        """Returns all possible agent reset locations in continuous (x,y) coordinates."""
        return self._unique_reset_locations

    @property
    def combined_locations(self) -> List[np.ndarray]:
        """Returns all possible goal/reset locations in continuous (x,y) coordinates."""
        return self._combined_locations

    @property
    def map_length(self) -> int:
        """Returns the length (number of rows i) of the maze."""
        return self._map_length

    @property
    def map_width(self) -> int:
        """Returns the width (number of columns j) of the maze."""
        return self._map_width

    @property
    def x_map_center(self) -> float:
        """Returns the x coordinate of the center of the maze in MuJoCo."""
        return self._x_map_center

    @property
    def y_map_center(self) -> float:
        """Returns the y coordinate of the center of the maze in MuJoCo."""
        return self._y_map_center

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        """Converts a cell index `(i,j)` to x and y coordinates in the MuJoCo simulation."""
        x = (rowcol_pos[1] + 0.5) * self.maze_size_scaling - self.x_map_center
        y = self.y_map_center - (rowcol_pos[0] + 0.5) * self.maze_size_scaling

        return np.array([x, y])

    def cell_xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        """Converts a continuous `(x,y)` coordinate to a discrete cell index `(i,j)`."""
        i = math.floor((self.y_map_center - xy_pos[1]) / self.maze_size_scaling)
        j = math.floor((xy_pos[0] + self.x_map_center) / self.maze_size_scaling)
        return np.array([i, j])

    @classmethod
    def make_maze(
        cls,
        agent_xml_path: str,
        maze_map: list,
        maze_size_scaling: float,
        maze_height: float,
        render_goal: bool = True,
    ):
        """Class method that returns a Maze instance and the path to a new MJCF XML file
        with the maze geometry included.

        Args:
            agent_xml_path (str): Path to the base agent XML file.
            maze_map (list[list[str,int]]): The discrete maze structure.
            maze_size_scaling (float): Scaling factor for the maze.
            maze_height (float): Height of the maze walls.
            render_goal (bool): Whether to render the goal site.

        Returns:
            tuple[Maze, str]: The initialized Maze object and the temporary XML file path.
        """
        tree = ET.parse(agent_xml_path)
        worldbody = tree.find(".//worldbody")

        maze = cls(maze_map, maze_size_scaling, maze_height)
        empty_locations = []
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Calculate cell center coordinates in simulation global Cartesian coordinates
                x = (j + 0.5) * maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze_size_scaling
                if struct == 1:  # Wall block.
                    # Add wall geometry to the MuJoCo worldbody
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                        size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {maze_height / 2 * maze_size_scaling}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.7 0.5 0.3 .99",
                    )

                elif struct == RESET:
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif struct == GOAL:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    maze._combined_locations.append(np.array([x, y]))
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))

        # Add target site for visualization
        goal_rgba = "1 0 0 0.99" if render_goal else "1 0 0 0.0"
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 {maze_height / 2 * maze_size_scaling}",
            size=f"{0.2 * maze_size_scaling}",
            rgba=goal_rgba,
            type="sphere",
        )

        # Handle cases where reset/goal/combined locations are not explicitly marked
        if (
            not maze._unique_goal_locations
            and not maze._unique_reset_locations
            and not maze._combined_locations
        ):
            # If no special cells, all empty cells are combined locations
            maze._combined_locations = empty_locations
        elif not maze._unique_reset_locations and not maze._combined_locations:
            # If no reset/combined cells, all empty cells are reset locations
            maze._unique_reset_locations = empty_locations
        elif not maze._unique_goal_locations and not maze._combined_locations:
            # If no goal/combined cells, all empty cells are goal locations
            maze._unique_goal_locations = empty_locations

        # Combined locations are valid for both goal and reset
        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_name = f"ant_maze{str(time.time())}.xml"
            temp_xml_path = path.join(path.dirname(tmp_dir), temp_xml_name)
            tree.write(temp_xml_path)

        return maze, temp_xml_path


class MazeEnv(GoalEnv):
    """
    Base class for maze environments, handling maze creation, goal generation,
    reset logic, and goal-conditioned reward/termination.
    """
    def __init__(
        self,
        agent_xml_path: str,
        reward_type: str = "dense",
        continuing_task: bool = True,
        reset_target: bool = True,
        maze_map: List[List[Union[int, str]]] = U_MAZE,
        maze_size_scaling: float = 1.0,
        maze_height: float = 0.5,
        position_noise_range: float = 0.25,
        render_goal: bool = True,
        **kwargs,
    ):

        self.reward_type = reward_type
        self.continuing_task = continuing_task
        self.reset_target = reset_target
        # Create maze geometry and get temp xml path
        self.maze, self.tmp_xml_file_path = Maze.make_maze(
            agent_xml_path, maze_map, maze_size_scaling, maze_height, render_goal=render_goal
        )

        self.position_noise_range = position_noise_range

    def generate_target_goal(self) -> np.ndarray:
        """Samples a goal position from the available unique goal locations."""
        assert len(self.maze.unique_goal_locations) > 0
        goal_index = self.np_random.integers(
            low=0, high=len(self.maze.unique_goal_locations)
        )
        goal = self.maze.unique_goal_locations[goal_index].copy()
        return goal

    def generate_reset_pos(self) -> np.ndarray:
        """Samples a reset position, ensuring it's not too close to the current goal."""
        assert len(self.maze.unique_reset_locations) > 0

        # While reset position is close to goal position
        reset_pos = self.goal.copy()
        while (
            np.linalg.norm(reset_pos - self.goal) <= 0.5 * self.maze.maze_size_scaling
        ):
            reset_index = self.np_random.integers(
                low=0, high=len(self.maze.unique_reset_locations)
            )
            reset_pos = self.maze.unique_reset_locations[reset_index].copy()

        return reset_pos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):
        """Reset the maze simulation.

        Args:
            options (dict): Can contain "goal_cell" and "reset_cell" to manually set
                            initial goal and reset locations (i,j).
        """
        super().reset(seed=seed)

        if options is None:
            goal = self.generate_target_goal()
            self.goal = self.add_xy_position_noise(goal)
            reset_pos = self.generate_reset_pos()
        else:
            if "goal_cell" in options and options["goal_cell"] is not None:
                # Validate goal cell is within bounds and not a wall
                assert self.maze.map_length > options["goal_cell"][0]
                assert self.maze.map_width > options["goal_cell"][1]
                assert (
                    self.maze.maze_map[options["goal_cell"][0]][options["goal_cell"][1]]
                    != 1
                ), f"Goal can't be placed in a wall cell, {options['goal_cell']}"

                goal = self.maze.cell_rowcol_to_xy(options["goal_cell"])

            else:
                goal = self.generate_target_goal()

            # Apply noise to goal position
            self.goal = self.add_xy_position_noise(goal)

            if "reset_cell" in options and options["reset_cell"] is not None:
                # Validate reset cell is within bounds and not a wall
                assert self.maze.map_length > options["reset_cell"][0]
                assert self.maze.map_width > options["reset_cell"][1]
                assert (
                    self.maze.maze_map[options["reset_cell"][0]][
                        options["reset_cell"][1]
                    ]
                    != 1
                ), f"Reset can't be placed in a wall cell, {options['reset_cell']}"

                reset_pos = self.maze.cell_rowcol_to_xy(options["reset_cell"])

            else:
                reset_pos = self.generate_reset_pos()

        # Apply noise to reset position
        self.reset_pos = self.add_xy_position_noise(reset_pos)

        # Update the position of the target site for visualization
        self.update_target_site_pos()

    def add_xy_position_noise(self, xy_pos: np.ndarray) -> np.ndarray:
        """Adds uniform noise to the x and y coordinates of a position."""
        noise_x = (
            self.np_random.uniform(
                low=-self.position_noise_range, high=self.position_noise_range
            )
            * self.maze.maze_size_scaling
        )
        noise_y = (
            self.np_random.uniform(
                low=-self.position_noise_range, high=self.position_noise_range
            )
            * self.maze.maze_size_scaling
        )
        xy_pos[0] += noise_x
        xy_pos[1] += noise_y

        return xy_pos

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        """Computes the reward based on the distance to the desired goal."""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "dense":
            return np.exp(-distance)
        elif self.reward_type == "sparse":
            # Reward is 1.0 if distance is less than or equal to 0.45
            return (distance <= 0.45).astype(np.float64)

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        """Determines if the episode is terminated (only for episodic tasks)."""
        if not self.continuing_task:
            # Terminate the episode when the goal is reached
            return bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.45)
        else:
            # Continuing tasks only terminate via truncation (time limit)
            return False

    def update_goal(self, achieved_goal: np.ndarray) -> None:
        """If task is continuing and the goal is reached, selects a new goal."""

        if (
            self.continuing_task
            and self.reset_target
            and bool(np.linalg.norm(achieved_goal - self.goal) <= 0.45)
            and len(self.maze.unique_goal_locations) > 1
        ):
            # Generate a new goal, ensuring it is not too close to the achieved_goal
            while np.linalg.norm(achieved_goal - self.goal) <= 0.45:
                # Generate another goal
                goal = self.generate_target_goal()
                # Add noise to goal position
                self.goal = self.add_xy_position_noise(goal)

            # Update the position of the target site for visualization
            self.update_target_site_pos()

    def compute_truncated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        """Always returns False, truncation is handled by time limit wrappers."""
        return False

    def update_target_site_pos(self):
        """Must be implemented by child class to update the goal site in the MuJoCo simulation."""
        raise NotImplementedError


class PointMazeEnv(MazeEnv, EzPickle):
    """
    Unified Point Maze environment using the MuJoCo Point model.

    Features:
    - Fully functional PointMaze environment using MuJoCo.
    - Adjustable render resolution (height, width).
    - Configurable top-down camera (elevation, distance, azimuth, lookat).
    - Direct control over position and velocity (set_state, set_position).
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]] = maps.U_MAZE,
        render_mode: Optional[str] = "rgb_array",
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        render_img_size: tuple = (224, 224),
        render_goal: bool = True,
        camera_elevation: float = 90.0,
        camera_distance: float = 5.0,
        camera_azimuth: float = 0,  # xy angle of rotation
        camera_lookat: np.ndarray = np.array([0.0, 0.0, 0.0]),
        **kwargs,
    ):
        # Path to the base XML file for the point mass agent
        point_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "assets/point.xml"
        )
        super().__init__(
            agent_xml_path=point_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=1,
            maze_height=0.01,  # Reduced height to hide walls' thickness
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            render_goal=render_goal,
            **kwargs,
        )

        maze_length = len(maze_map)
        # Adjust default camera distance based on maze size
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        # Create PointEnv, which handles the MuJoCo simulation
        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape = self.point_env.observation_space.shape
        # Define the observation space as a dictionary for goal-conditioned tasks
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype="float64"),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode
        self.render_goal = render_goal

        # Set renderer image size
        self.point_env.mujoco_renderer.height, self.point_env.mujoco_renderer.width = render_img_size
        self.viewer = None

        # Camera configuration parameters
        self._camera_elevation = camera_elevation
        self._camera_distance = camera_distance
        self._camera_azimuth = camera_azimuth
        self._camera_lookat = camera_lookat

        # Initialize EzPickle for serialization
        EzPickle.__init__(
            self,
            maze_map,
            render_mode,
            reward_type,
            continuing_task,
            reset_target,
            render_img_size,
            render_goal,
            camera_elevation,
            camera_distance,
            camera_azimuth,
            camera_lookat,
            **kwargs,
        )

    # ===============================================================
    # Core Environment Methods
    # ===============================================================

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        """Resets the environment, setting the agent and goal positions."""
        super().reset(seed=seed, **kwargs)
        # Set the agent's initial position in the MuJoCo data
        self.point_env.init_qpos[:2] = self.reset_pos

        obs, info = self.point_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        # Add 'success' indicator to the info dictionary
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        return obs_dict, info

    def step(self, action):
        """Steps the simulation forward given an action."""
        # The PointEnv step returns observation, reward, terminated, truncated, info
        obs, _, _, _, info = self.point_env.step(action)
        obs_dict = self._get_obs(obs)

        reward = self.compute_reward(obs_dict["achieved_goal"], self.goal, info)
        terminated = self.compute_terminated(obs_dict["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs_dict["achieved_goal"], self.goal, info)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        # Update goal if the continuing task goal has been reached
        self.update_goal(obs_dict["achieved_goal"])
        return obs_dict, reward, terminated, truncated, info

    def _get_obs(self, point_obs) -> Dict[str, np.ndarray]:
        """Converts the raw PointEnv observation into the GoalEnv observation dictionary."""
        achieved_goal = point_obs[:2]
        return {
            "observation": point_obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    # ===============================================================
    # Rendering & Camera Control
    # ===============================================================

    def _get_viewer(self):
        """Configures and returns the MuJoCo viewer with custom camera settings."""
        viewer = self.point_env.mujoco_renderer._get_viewer(render_mode="rgb_array")
        viewer.cam.elevation = self._camera_elevation
        viewer.cam.distance = self._camera_distance
        viewer.cam.azimuth = self._camera_azimuth
        viewer.cam.lookat[:] = self._camera_lookat
        return viewer

    def render(self):
        """Renders the environment."""
        if self.viewer is None:
            self.viewer = self._get_viewer()
        return self.viewer.render(render_mode="rgb_array")

    # ===============================================================
    # Low-Level State Control
    # ===============================================================

    def set_state(self, position: np.ndarray, velocity: np.ndarray):
        """Directly sets the internal MuJoCo state (qpos and qvel)."""
        self.point_env.data.qpos[:] = position
        self.point_env.data.qvel[:] = velocity
        # Perform a step to propagate the change
        return self.step(np.zeros(self.action_space.shape))

    def set_position(self, position: np.ndarray):
        """Sets position, keeping zero velocity."""
        return self.set_state(position, np.zeros_like(position))

    # ===============================================================
    # Utility and Cleanup
    # ===============================================================

    def update_target_site_pos(self):
        """Updates the position of the visual target site in the MuJoCo model."""
        # Site pos is (x, y, z). Z is half the wall height.
        self.point_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def close(self):
        """Closes the PointEnv and the base GoalEnv."""
        super().close()
        self.point_env.close()

    @property
    def model(self):
        """Returns the MuJoCo model."""
        return self.point_env.model

    @property
    def data(self):
        """Returns the MuJoCo simulation data."""
        return self.point_env.data