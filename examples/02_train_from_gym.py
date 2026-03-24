"""Train a world model from Gymnasium environment recordings.

Requires: pip install worldkit[envs]
"""

print("Example: Training from Gym environment")
print("=" * 50)
print()
print("To run this example, install gym extras:")
print("  pip install worldkit[envs]")
print()
print("Then use:")
print("  from worldkit.data import Recorder")
print("  import gymnasium as gym")
print("  env = gym.make('CartPole-v1', render_mode='rgb_array')")
print("  recorder = Recorder(env, output='cartpole_data.h5')")
print("  recorder.record(episodes=100)")
print()
print("  from worldkit import WorldModel")
print("  model = WorldModel.train(data='cartpole_data.h5', config='nano')")
print("  model.save('cartpole_model.wk')")
