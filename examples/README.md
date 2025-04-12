# H-MAS Examples

This directory contains example notebooks and scripts demonstrating how to use the H-MAS framework.

## Setup

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Start Jupyter:
```bash
jupyter notebook
```

## Available Examples

### 1. Learning Agent Demo (`learning_agent_demo.ipynb`)

A comprehensive demonstration of the `LearningAgent` class in a practical scenario. This notebook shows:

- How to create and initialize a learning agent
- Training the agent in a grid world environment
- Using different learning types (reinforcement, meta-learning)
- Analyzing agent performance and strategy adaptation
- Visualizing learning progress
- Testing the trained agent

The example implements:
- A simple grid world environment
- Complete training loop with strategy adaptation
- Performance visualization
- Agent analysis tools

### 2. Classification and Clustering Demo (`learning_agent_classification_demo.ipynb`)

A demonstration of supervised and unsupervised learning capabilities. This notebook shows:

- Using the agent for binary classification
- Implementing clustering with the same agent
- Comparing different learning approaches
- Strategy adaptation for each learning type
- Performance visualization and analysis

The example implements:
- Synthetic dataset generation
- Batch training for classification
- Iterative clustering
- Performance comparison between learning types
- Strategy analysis and adaptation

Key concepts demonstrated in both examples:
- Agent initialization and setup
- Processing different types of learning inputs
- Meta-learning and strategy adaptation
- Performance analysis and visualization
- Agent reflection and state examination

### Running the Examples

1. Make sure you have all requirements installed
2. Open the desired notebook in Jupyter
3. Run all cells in sequence
4. Experiment with parameters and modifications

### Tips

- The notebooks are designed to be educational and modifiable
- Each cell contains comments explaining the code
- Feel free to modify parameters and observe the effects
- Use the visualization tools to understand agent behavior
- Check the agent's reflection data to understand its learning process

### Troubleshooting

If you encounter issues:

1. Ensure all requirements are installed correctly
2. Check that the H-MAS package is installed and importable
3. Verify that you're running the cells in sequence
4. Make sure you have Python 3.8+ installed

For more help, check the main documentation or open an issue on the repository. 