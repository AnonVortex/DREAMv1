# by Alexis Soto-Yanez
"""
Dynamic Negotiation Prototype with Multiple Rounds

This script extends the messaging and voting mechanism by implementing a
multi-round negotiation loop. Agents update their proposals over several rounds,
adjusting their confidence scores based on negotiation feedback, and then a final
weighted vote determines the consensus output.
"""

import asyncio
import random
from dataclasses import dataclass

@dataclass
class Proposal:
    agent_id: str
    output: str
    confidence: float

class Agent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.proposal = None

    async def send_initial_proposal(self, message_queue: asyncio.Queue):
        # Simulate processing delay
        await asyncio.sleep(random.uniform(0.5, 1.5))
        output = f"{self.agent_id}_output"
        confidence = random.uniform(0.5, 1.0)
        self.proposal = Proposal(agent_id=self.agent_id, output=output, confidence=confidence)
        print(f"[{self.agent_id}] Initial proposal: {self.proposal}")
        await message_queue.put(self.proposal)

    async def update_proposal(self, message_queue: asyncio.Queue, round_number: int):
        # Simulate processing delay
        await asyncio.sleep(random.uniform(0.5, 1.5))
        if self.proposal:
            # Adjust confidence by a small random amount to simulate negotiation.
            adjustment = random.uniform(-0.1, 0.1)
            new_confidence = max(0.0, min(1.0, self.proposal.confidence + adjustment))
            # For simplicity, the output remains unchanged.
            self.proposal = Proposal(agent_id=self.agent_id, output=self.proposal.output, confidence=new_confidence)
            print(f"[{self.agent_id}] Updated proposal in round {round_number}: {self.proposal}")
            await message_queue.put(self.proposal)

class NegotiationCoordinator:
    def __init__(self, num_agents: int, num_rounds: int):
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.proposals = []

    async def collect_proposals(self, message_queue: asyncio.Queue):
        # Collect proposals until we have proposals from all agents.
        while len({p.agent_id for p in self.proposals}) < self.num_agents:
            proposal = await message_queue.get()
            # Update the proposal for the agent.
            self.proposals = [p for p in self.proposals if p.agent_id != proposal.agent_id]
            self.proposals.append(proposal)
            print(f"[Coordinator] Collected proposal from {proposal.agent_id}")
        return self.proposals

    def perform_voting(self, proposals):
        # Weight each proposal by its confidence.
        vote_counts = {}
        for prop in proposals:
            vote_counts.setdefault(prop.output, 0)
            vote_counts[prop.output] += prop.confidence
        print(f"[Coordinator] Vote counts: {vote_counts}")
        winning_output = max(vote_counts, key=vote_counts.get)
        return winning_output, vote_counts[winning_output]

async def negotiation_rounds(num_rounds, agents, coordinator, message_queue):
    for round_number in range(1, num_rounds + 1):
        print(f"\n--- Negotiation Round {round_number} ---")
        # Each agent updates its proposal.
        update_tasks = [agent.update_proposal(message_queue, round_number) for agent in agents]
        await asyncio.gather(*update_tasks)
        # Collect the updated proposals.
        proposals = await coordinator.collect_proposals(message_queue)
        # Intermediate voting (for demonstration, we just print the result).
        winning_output, winning_weight = coordinator.perform_voting(proposals)
        print(f"[Coordinator] Round {round_number} Winning Output: {winning_output} with weighted vote {winning_weight:.2f}")
    return proposals

async def main():
    num_agents = 4
    num_rounds = 3  # Number of negotiation rounds.
    message_queue = asyncio.Queue()
    
    # Initialize agents.
    agents = [Agent(f"Agent{i+1}") for i in range(num_agents)]
    
    # Each agent sends its initial proposal.
    initial_tasks = [agent.send_initial_proposal(message_queue) for agent in agents]
    await asyncio.gather(*initial_tasks)
    
    coordinator = NegotiationCoordinator(num_agents=num_agents, num_rounds=num_rounds)
    
    # Collect initial proposals.
    proposals = await coordinator.collect_proposals(message_queue)
    winning_output, winning_weight = coordinator.perform_voting(proposals)
    print(f"\n[Coordinator] Initial Winning Output: {winning_output} with weighted vote {winning_weight:.2f}")
    
    # Perform multiple negotiation rounds.
    proposals = await negotiation_rounds(num_rounds, agents, coordinator, message_queue)
    
    # Final voting after all negotiation rounds.
    final_winning_output, final_winning_weight = coordinator.perform_voting(proposals)
    print(f"\n[Coordinator] Final Winning Output: {final_winning_output} with weighted vote {final_winning_weight:.2f}")

if __name__ == "__main__":
    # For interactive environments like Spyder, we patch the event loop.
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())

