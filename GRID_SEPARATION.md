# Environment and Grid Separation

## Overview

The codebase has been updated to separate the environment name from grid selection, making it easier to switch between different training grids without changing the core environment configuration.

## Changes Made

### Before
```python
ENV_NAME = "ReferenceModel-3-1"  # Both environment type AND specific grid
CP_TRAINED_ON_ENV_NAME = "ReferenceModel-3-1"

env_setup = {
    "env_name": ENV_NAME,  # Used for both environment type and grid selection
    # ... other config
}
```

### After
```python
ENV_NAME = "ReferenceModelMultiAgent"  # Generic environment type
GRID_NAME = "ReferenceModel-3-1"      # Specific grid to use
CP_TRAINED_ON_GRID_NAME = "ReferenceModel-3-1"  # Grid the checkpoint was trained on

env_setup = {
    "env_name": ENV_NAME,    # Environment type for RLlib
    "grid_name": GRID_NAME,  # Which grid layout to use
    # ... other config
}
```

## Benefits

1. **Easy Grid Switching**: Change `GRID_NAME` to switch between different grids
2. **Clear Separation**: Environment type vs. grid layout are now separate concerns
3. **Checkpoint Compatibility**: System knows which grid a model was trained on
4. **Backward Compatibility**: Old configurations still work

## Available Grids

- `ReferenceModel-1-1`: 2x9 grid
- `ReferenceModel-1-2`: 3x10 grid  
- `ReferenceModel-2-1`: 10x20 grid
- `ReferenceModel-2-2`: 15x21 grid
- `ReferenceModel-3-1`: 20x30 grid

## Usage Examples

### Training on Different Grids
```python
# Train on large grid
GRID_NAME = "ReferenceModel-3-1"

# Switch to smaller grid for testing
GRID_NAME = "ReferenceModel-1-2"
```

### Loading Checkpoints
The system automatically uses the correct grid for checkpoint loading:
```python
CP_TRAINED_ON_GRID_NAME = "ReferenceModel-3-1"  # Grid used for training
# When loading checkpoint, it will use this grid regardless of current GRID_NAME
```

### Programmatic Grid Selection
```python
from main import env_setup, env_creator

# Create environment with specific grid
test_config = env_setup.copy()
test_config["grid_name"] = "ReferenceModel-2-1"
env = env_creator(env_config=test_config)
```