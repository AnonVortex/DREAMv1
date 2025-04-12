import unittest
import numpy as np
from environment.environment_manager import EnvironmentManager, EnvironmentConfig, PhysicsBackend
from environment.task_environment import TaskEnvironment, TaskConfig, TaskType

class TestEnvironment(unittest.TestCase):
    """Test cases for the environment implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.env_config = EnvironmentConfig(
            physics_backend=PhysicsBackend.PYBULLET,
            enable_gui=False,
            max_steps=100
        )
        self.task_config = TaskConfig(
            task_type=TaskType.NAVIGATION,
            time_limit=60.0,
            success_threshold=0.95,
            reward_shaping=True
        )
        self.env = TaskEnvironment(self.env_config, self.task_config)
    
    def tearDown(self):
        """Clean up after tests."""
        if self.env:
            self.env.close()
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env)
        self.assertIsNotNone(self.env.env_manager)
        self.assertTrue(self.env.env_manager.is_initialized)
        
        # Check spaces
        self.assertEqual(self.env.action_space.shape, (6,))
        self.assertEqual(self.env.observation_space.shape, (10,))
    
    def test_reset(self):
        """Test environment reset."""
        observation = self.env.reset()
        
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (10,))
        self.assertEqual(len(self.env.active_agents), 1)
        self.assertEqual(len(self.env.goals), 1)
        
        # Check agent position is within bounds
        agent_pos = observation[:3]
        self.assertTrue(np.all(np.abs(agent_pos) <= 5))
    
    def test_step(self):
        """Test environment stepping."""
        self.env.reset()
        
        # Take a random action
        action = self.env.action_space.sample()
        observation, reward, done, info = self.env.step(action)
        
        # Check return values
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (10,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # Check info dictionary
        self.assertIn("current_step", info)
        self.assertIn("episode_reward", info)
        self.assertIn("task_completion", info)
    
    def test_reward_computation(self):
        """Test reward computation."""
        self.env.reset()
        
        # Get initial observation
        obs = self.env._get_observation()
        initial_distance = obs[-1]  # Last element is distance to goal
        
        # Take a zero action (should not move)
        action = np.zeros(6)
        _, reward, _, _ = self.env.step(action)
        
        # Check reward is negative (proportional to distance)
        self.assertLess(reward, 0)
        self.assertAlmostEqual(
            reward,
            -initial_distance / self.task_config.exploration_area_size
        )
    
    def test_manipulation_task(self):
        """Test manipulation task environment."""
        task_config = TaskConfig(
            task_type=TaskType.MANIPULATION,
            time_limit=60.0,
            success_threshold=0.95,
            reward_shaping=True
        )
        env = TaskEnvironment(self.env_config, task_config)
        
        # Test reset
        observation = env.reset()
        self.assertEqual(observation.shape, (16,))
        
        # Test step
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        self.assertEqual(observation.shape, (16,))
        self.assertIsInstance(reward, float)
        
        env.close()
    
    def test_cooperation_task(self):
        """Test cooperation task environment."""
        task_config = TaskConfig(
            task_type=TaskType.COOPERATION,
            time_limit=60.0,
            success_threshold=0.95,
            reward_shaping=True,
            min_agents_for_cooperation=2,
            max_agents_for_cooperation=3
        )
        env = TaskEnvironment(self.env_config, task_config)
        
        # Test reset
        observation = env.reset()
        self.assertEqual(observation.shape, (30,))  # 10 states * 3 max agents
        
        # Verify number of agents
        num_agents = len(env.active_agents)
        self.assertGreaterEqual(num_agents, task_config.min_agents_for_cooperation)
        self.assertLessEqual(num_agents, task_config.max_agents_for_cooperation)
        
        # Test step
        action = np.random.uniform(-1, 1, (num_agents * 6,))  # 6 actions per agent
        observation, reward, done, info = env.step(action)
        
        self.assertEqual(observation.shape, (30,))
        self.assertIsInstance(reward, float)
        
        env.close()
    
    def test_exploration_task(self):
        """Test exploration task environment."""
        task_config = TaskConfig(
            task_type=TaskType.EXPLORATION,
            time_limit=60.0,
            success_threshold=0.95,
            reward_shaping=True,
            exploration_area_size=10.0
        )
        env = TaskEnvironment(self.env_config, task_config)
        
        # Test reset
        observation = env.reset()
        self.assertIsInstance(observation, dict)
        self.assertIn("agent_state", observation)
        self.assertIn("exploration_map", observation)
        
        # Check map size
        map_size = int(task_config.exploration_area_size)
        self.assertEqual(
            observation["exploration_map"].shape,
            (map_size, map_size)
        )
        
        # Test step
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        self.assertIsInstance(observation, dict)
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, 0)
        self.assertLessEqual(reward, 1)
        
        env.close()
    
    def test_action_noise(self):
        """Test action noise application."""
        task_config = TaskConfig(
            task_type=TaskType.NAVIGATION,
            action_noise=0.1
        )
        env = TaskEnvironment(self.env_config, task_config)
        
        env.reset()
        original_action = np.ones(6)
        
        # Take multiple steps and verify actions are noisy
        actions_are_different = False
        for _ in range(10):
            observation, _, _, _ = env.step(original_action.copy())
            if not np.allclose(observation[3:6], original_action[:3]):
                actions_are_different = True
                break
        
        self.assertTrue(actions_are_different)
        env.close()
    
    def test_observation_noise(self):
        """Test observation noise application."""
        task_config = TaskConfig(
            task_type=TaskType.NAVIGATION,
            observation_noise=0.1
        )
        env = TaskEnvironment(self.env_config, task_config)
        
        # Take multiple observations and verify they're noisy
        env.reset()
        action = np.zeros(6)  # Should not move the agent
        
        first_obs = env._get_observation()
        observations_are_different = False
        
        for _ in range(10):
            env.step(action)
            new_obs = env._get_observation()
            if not np.allclose(first_obs, new_obs):
                observations_are_different = True
                break
        
        self.assertTrue(observations_are_different)
        env.close()

if __name__ == "__main__":
    unittest.main() 