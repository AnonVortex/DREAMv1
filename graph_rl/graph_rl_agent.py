# by Alexis Soto-Yanez
"""
Graph RL Agent Module for HMAS Prototype
Author: Alexis Soto-Yanez
Date: 2025
Description:
    A production-level implementation of a Graph Reinforcement Learning (RL) agent using an actor-critic framework.
    This module leverages Graph Neural Networks (GNNs) (using PyTorch Geometric) to represent and process graph-structured data.
    The agent learns a policy to optimize combinatorial tasks (e.g., routing, scheduling) while outputting both action probabilities (policy)
    and value estimates (for critic feedback). All internal details are encapsulated behind a high-level API.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import numpy as np

# -----------------------------
# GraphRLAgent: Actor-Critic Network
# -----------------------------
class GraphRLAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, policy_output_dim):
        """
        Initialize the Graph RL Agent with a GNN-based actor-critic network.
        
        Parameters:
            input_dim (int): Dimensionality of node features.
            hidden_dim (int): Dimensionality of hidden representations.
            policy_output_dim (int): Size of the action space (number of discrete actions).
        """
        super(GraphRLAgent, self).__init__()
        # Use Graph Attention Network (GAT) layers for adaptive neighbor aggregation.
        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)  # Output: 2*hidden_dim
        self.gat2 = GATConv(2 * hidden_dim, hidden_dim, heads=1, concat=False)  # Output: hidden_dim
        # Actor head: outputs logits for action probabilities.
        self.policy_head = nn.Linear(hidden_dim, policy_output_dim)
        # Critic head: outputs a scalar value estimation.
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        """
        Forward pass for the Graph RL agent.
        Parameters:
            data (torch_geometric.data.Data): Data object containing node features, edge indices, and batch info.
        Returns:
            policy_logits (Tensor): Logits for each action.
            state_value (Tensor): Estimated value for the current state.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        # Aggregate node features into a graph-level representation.
        x_pool = global_mean_pool(x, batch)
        policy_logits = self.policy_head(x_pool)
        state_value = self.value_head(x_pool)
        return policy_logits, state_value

# -----------------------------
# GraphRLAgentWrapper: High-Level API
# -----------------------------
class GraphRLAgentWrapper:
    def __init__(self, input_dim=10, hidden_dim=64, policy_output_dim=5, lr=0.001, gamma=0.99):
        """
        Wrapper class that encapsulates the Graph RL Agent and provides utility functions for:
          - Converting raw data to graph representations.
          - Running training steps with an actor-critic method.
          - Exposing a high-level API to solve graph-based combinatorial tasks.
        
        Parameters:
            input_dim (int): Dimension of node features.
            hidden_dim (int): Hidden dimension size.
            policy_output_dim (int): Number of possible discrete actions.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GraphRLAgent(input_dim, hidden_dim, policy_output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def build_graph(self, raw_data):
        """
        Build a NetworkX graph from raw data.
        
        Parameters:
            raw_data (dict): Should have keys 'nodes' and 'edges'.
                - 'nodes' is a list of dicts with keys: 'id' and 'features' (list of floats).
                - 'edges' is a list of tuples (source, target).
        
        Returns:
            G (networkx.Graph): Constructed graph with node features.
        """
        G = nx.Graph()
        for node in raw_data.get("nodes", []):
            features = node.get("features", [0.0] * 10)
            G.add_node(node["id"], features=features)
        for edge in raw_data.get("edges", []):
            G.add_edge(edge[0], edge[1])
        return G

    def graph_to_data(self, G):
        """
        Convert a NetworkX graph into a PyTorch Geometric Data object.
        
        Parameters:
            G (networkx.Graph): Graph with node attributes 'features'.
        
        Returns:
            data (torch_geometric.data.Data): Data object for the GNN.
        """
        # Extract node features in order.
        features = []
        for n in G.nodes():
            feat = G.nodes[n].get("features", [0.0] * 10)
            features.append(feat)
        x = torch.tensor(features, dtype=torch.float)
        # Convert edges to tensor.
        edges = list(G.edges())
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        # Create a batch vector (all nodes belong to a single graph).
        batch = torch.zeros(x.size(0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return data

    def select_action(self, data):
        """
        Given a PyG Data object, compute the action probabilities and sample an action.
        
        Parameters:
            data (torch_geometric.data.Data): Input graph data.
        
        Returns:
            action (int): The selected discrete action.
            state_value (float): The critic's value estimate.
            log_prob (Tensor): Log probability of the chosen action.
        """
        self.model.eval()
        with torch.no_grad():
            policy_logits, state_value = self.model(data.to(self.device))
            action_probs = F.softmax(policy_logits, dim=-1)
            # Create a categorical distribution and sample an action.
            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample()
            return action.item(), state_value.item(), distribution.log_prob(action)
    
    def train_step(self, data, target_reward):
        """
        Perform a single training step using the actor-critic method.
        
        Parameters:
            data (torch_geometric.data.Data): Input graph data.
            target_reward (float): The reward signal obtained from the environment.
        
        Returns:
            loss (float): The computed loss for this step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        policy_logits, state_value = self.model(data.to(self.device))
        # For simplicity, we assume a single scalar reward for the whole graph.
        # Compute advantage.
        advantage = target_reward - state_value
        # Actor loss: policy gradient (negative log likelihood weighted by advantage)
        action_probs = F.softmax(policy_logits, dim=-1)
        distribution = torch.distributions.Categorical(action_probs)
        # In this simplified version, we assume the chosen action is the one with max probability.
        action = torch.argmax(action_probs, dim=-1)
        log_prob = distribution.log_prob(action)
        actor_loss = -log_prob * advantage.detach()
        # Critic loss: Mean Squared Error between estimated value and reward.
        critic_loss = F.mse_loss(state_value, torch.tensor([[target_reward]], dtype=torch.float).to(self.device))
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_episode(self, raw_data, target_reward, steps=5):
        """
        Train the Graph RL agent on a single episode for a given raw graph task.
        
        Parameters:
            raw_data (dict): Raw data to build the graph.
            target_reward (float): Reward signal for the task.
            steps (int): Number of training steps (or episodes) to perform.
        
        Returns:
            avg_loss (float): Average loss over the episode.
        """
        G = self.build_graph(raw_data)
        data = self.graph_to_data(G)
        data = data.to(self.device)
        total_loss = 0.0
        for step in range(steps):
            loss = self.train_step(data, target_reward)
            total_loss += loss
        avg_loss = total_loss / steps
        return avg_loss

    def solve_task(self, raw_data, target_reward, training_steps=20):
        """
        High-level API to solve a combinatorial optimization task using Graph RL.
        This function trains the agent for a specified number of episodes and returns the final decision.
        
        Parameters:
            raw_data (dict): Raw input data to build the graph.
            target_reward (float): The reward signal used for training (placeholder for actual environment reward).
            training_steps (int): Number of training episodes.
        
        Returns:
            final_action (int): The final selected action after training.
            final_state_value (float): The final value estimate for the state.
        """
        # Optionally, train the agent for a number of episodes.
        for epoch in range(training_steps):
            avg_loss = self.train_episode(raw_data, target_reward, steps=5)
            print(f"Epoch {epoch+1}/{training_steps} - Avg Loss: {avg_loss:.4f}")
        # After training, build the final graph and select an action.
        G = self.build_graph(raw_data)
        data = self.graph_to_data(G)
        data = data.to(self.device)
        final_action, final_state_value, _ = self.select_action(data)
        return final_action, final_state_value

# If executed as a script, run the full task solution process.
if __name__ == "__main__":
    # Define raw data for a production-level graph combinatorial task.
    raw_data = {
        "nodes": [
            {"id": 0, "features": [0.8]*10},
            {"id": 1, "features": [0.3]*10},
            {"id": 2, "features": [0.5]*10},
            {"id": 3, "features": [0.9]*10},
            {"id": 4, "features": [0.4]*10},
        ],
        "edges": [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (0, 2),
            (1, 3)
        ]
    }
    
    # In a real scenario, target_reward is provided by an environment. Here we use a dummy value.
    target_reward = 1.0
    
    # Initialize the agent wrapper.
    agent_wrapper = GraphRLAgentWrapper(input_dim=10, hidden_dim=64, policy_output_dim=5, lr=0.001, gamma=0.99)
    
    print("Training Graph RL Agent on Task...")
    final_action, final_value = agent_wrapper.solve_task(raw_data, target_reward, training_steps=20)
    print("\nFinal Selected Action:", final_action)
    print("Final State Value Estimate:", final_value)
