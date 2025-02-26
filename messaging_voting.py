# by Alexis Soto-Yanez
"""
Messaging Layer and Voting Mechanism Prototype

This script demonstrates:
  - Asynchronous messaging using asyncio: agents send messages to a central coordinator.
  - A simple voting mechanism: agents propose outputs with confidence scores,
    and the coordinator aggregates these proposals using weighted voting.
  
Each agent sends a message with its output and confidence score.
The coordinator collects messages and computes a weighted consensus output.
"""

import asyncio
import random
from dataclasses import dataclass

# Define a message structure for proposals
@dataclass
class Proposal:
    agent_id: str
    output: str
    confidence: float

# Agent class simulating asynchronous proposal generation
class Agent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def send_proposal(self, message_queue: asyncio.Queue):
        # Simulate processing delay
        await asyncio.sleep(random.uniform(0.5, 2.0))
        # Create a proposal with random confidence and a simulated output
        output = f"{self.agent_id}_output"
        confidence = random.uniform(0.5, 1.0)  # confidence between 0.5 and 1.0
        proposal = Proposal(agent_id=self.agent_id, output=output, confidence=confidence)
        print(f"[{self.agent_id}] Sending proposal: {proposal}")
        await message_queue.put(proposal)

# Coordinator class to aggregate proposals and perform voting
class VotingCoordinator:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.proposals = []

    async def collect_proposals(self, message_queue: asyncio.Queue):
        while len(self.proposals) < self.num_agents:
            proposal = await message_queue.get()
            self.proposals.append(proposal)
            print(f"[Coordinator] Collected proposal from {proposal.agent_id}")
        return self.proposals

    def perform_voting(self, proposals):
        # For simplicity, let's assume each proposal's "vote" is weighted by its confidence.
        # We select the output with the highest total weighted vote.
        vote_counts = {}
        for prop in proposals:
            vote_counts.setdefault(prop.output, 0)
            vote_counts[prop.output] += prop.confidence
        
        print(f"[Coordinator] Vote counts: {vote_counts}")
        # Determine the winning output
        winning_output = max(vote_counts, key=vote_counts.get)
        return winning_output, vote_counts[winning_output]

# Main asynchronous function to run the messaging and voting process
async def main():
    # Create a shared message queue for proposals
    message_queue = asyncio.Queue()
    
    # Create several agents (for example, 4 agents)
    agent_ids = ["Agent1", "Agent2", "Agent3", "Agent4"]
    agents = [Agent(agent_id) for agent_id in agent_ids]
    
    # Create tasks for each agent to send their proposals asynchronously
    tasks = [agent.send_proposal(message_queue) for agent in agents]
    
    # Create a voting coordinator
    coordinator = VotingCoordinator(num_agents=len(agents))
    
    # Run agent tasks concurrently and collect proposals
    await asyncio.gather(*tasks)
    proposals = await coordinator.collect_proposals(message_queue)
    
    # Perform voting based on the collected proposals
    winning_output, winning_weight = coordinator.perform_voting(proposals)
    print(f"\n[Coordinator] Winning Output: {winning_output} with weighted vote {winning_weight:.2f}")

# Entry point for the async event loop
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
    