## Overview
This project explores reinforcement learning (RL) and the use of vision-language models (VLMs) and large language models (LLMs) to automate reward function modeling.

## Project Structure

```
├── agents.py                    # Core agents implementation
├── CLIP.py                      # Leveraging CLIP for reward function modeling
├── model_final.pth              # Trained model weights
├── Molmo.py                     # Additional module for reward modeling
├── networks.py                  # Neural network architectures
├── project_main.py              # Main script for the project
├── project_main_pointmaze.py    # Script for the PointMaze environment
RL/
├── rollouts.py                  # Rollout implementation for training and evaluation
├── sac.py                       # Soft Actor-Critic implementation
├── utils.py                     # Utility functions for the project
├── video.py                     # Script for video generation and evaluation
├── vide_eval.mp4                # Example video demonstrating evaluation results
├── .gitignore                   # Git ignore file
```

## Key Features
- **LLMs and VLMs Integration**: Uses CLIP and related models for automated reward function design.
- **Video Outputs**: Includes scripts to generate videos for visual evaluation of RL performance.


### Running the Main Script
To train and evaluate the RL model, run:
```bash
python project_main.py
```

### Testing in PointMaze
To execute the script tailored for the PointMaze environment:
```bash
python project_main_pointmaze.py
```

### Video Generation
To generate evaluation videos:
```bash
python video.py
```
