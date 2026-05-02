print("=" * 40)
print("Testing all imports...")
print("=" * 40)

try:
    import gymnasium
    print("✓ gymnasium:", gymnasium.__version__)
except Exception as e:
    print("✗ gymnasium FAILED:", e)

try:
    import stable_baselines3
    print("✓ stable-baselines3:", stable_baselines3.__version__)
except Exception as e:
    print("✗ stable-baselines3 FAILED:", e)

try:
    import numpy as np
    print("✓ numpy:", np.__version__)
except Exception as e:
    print("✗ numpy FAILED:", e)

try:
    import matplotlib
    print("✓ matplotlib:", matplotlib.__version__)
except Exception as e:
    print("✗ matplotlib FAILED:", e)

try:
    import pandas as pd
    print("✓ pandas:", pd.__version__)
except Exception as e:
    print("✗ pandas FAILED:", e)

try:
    import gl_gym
    print("✓ gl_gym (GreenLight-Gym): imported successfully")
except Exception as e:
    print("✗ gl_gym FAILED:", e)

print("=" * 40)
print("All done!")
print("=" * 40)