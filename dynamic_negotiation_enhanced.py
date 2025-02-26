#by Alexis Soto-Yanez
"""
Enhanced Dynamic Negotiation Prototype with Justifications

This script extends the previous dynamic negotiation prototype by adding a 'justification'
field to each proposal. Agents generate and update justifications along with their outputs
and confidence scores over multiple negotiation rounds. The coordinator aggregates the proposals,
performs weighted voting, and outputs both the winning output and the associated justifications.
"""

import asyncio
import random
from dataclasses import dataclass

@dataclass
class Proposal:
    agent_id: str
    output: str
    confidence: float
    justification: str

class Agent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.proposal = None

    async def send_initial_proposal(self, message_queue: asyncio.Queue):
        await asyncio.sleep(random.uniform(0.5, 1.5))
        output = f"{self.agent_id}_output"
        confidence = random.uniform(0.5, 1.0)
        justification = f"Initial confidence {confidence:.2f} based on preliminary analysis."
        self.proposal = Proposal(agent_id=self.agent_id, output=output, confidence=confidence, justification=justification)
        print(f"[{self.agent_id}] Initial proposal: {self.proposal}")
        await message_queue.put(self.proposal)

    async def update_proposal(self, message_queue: asyncio.Queue, round_number: int):
        await asyncio.sleep(random.uniform(0.5, 1.5))
        if self.proposal:
            # Adjust confidence by a small random amount
            adjustment = random.uniform(-0.1, 0.1)
            new_confidence = max(0.0, min(1.0, self.proposal.confidence + adjustment))
            # Update justification to include round information and the adjustment made.
            justification = (f"Round {round_number}: Adjusted by {adjustment:+.2f}. "
                             f"New confidence {new_confidence:.2f}.")
            self.proposal = Proposal(
                agent_id=self.agent_id,
                output=self.proposal.output,  # keeping output unchanged for simplicity
                confidence=new_confidence,
                justification=justification
            )
            print(f"[{self.agent_id}] Updated proposal in round {round_number}: {self.proposal}")
            await message_queue.put(self.proposal)

class NegotiationCoordinator:
    def __init__(self, num_agents: int, num_rounds: int):
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.proposals = []

    async def collect_proposals(self, message_queue: asyncio.Queue):
        while len({p.agent_id for p in self.proposals}) < self.num_agents:
            proposal = await message_queue.get()
            # Replace any existing proposal from this agent
            self.proposals = [p for p in self.proposals if p.agent_id != proposal.agent_id]
            self.proposals.append(proposal)
            print(f"[Coordinator] Collected proposal from {proposal.agent_id}")
        return self.proposals

    def perform_voting(self, proposals):
        vote_counts = {}
        justifications = {}
        for prop in proposals:
            vote_counts.setdefault(prop.output, 0)
            vote_counts[prop.output] += prop.confidence
            justifications.setdefault(prop.output, []).append(f"{prop.agent_id}: {prop.justification}")
        print(f"[Coordinator] Vote counts: {vote_counts}")
        winning_output = max(vote_counts, key=vote_counts.get)
        return winning_output, vote_counts[winning_output], justifications[winning_output]

async def negotiation_rounds(num_rounds, agents, coordinator, message_queue):
    for round_number in range(1, num_rounds + 1):
        print(f"\n--- Negotiation Round {round_number} ---")
        update_tasks = [agent.update_proposal(message_queue, round_number) for agent in agents]
        await asyncio.gather(*update_tasks)
        proposals = await coordinator.collect_proposals(message_queue)
        winning_output, winning_weight, justifs = coordinator.perform_voting(proposals)
        print(f"[Coordinator] Round {round_number} Winning Output: {winning_output} with weighted vote {winning_weight:.2f}")
        print(f"[Coordinator] Justifications: {justifs}")
    return proposals

async def main():
    num_agents = 4
    num_rounds = 3
    message_queue = asyncio.Queue()
    
    agents = [Agent(f"Agent{i+1}") for i in range(num_agents)]
    initial_tasks = [agent.send_initial_proposal(message_queue) for agent in agents]
    await asyncio.gather(*initial_tasks)
    
    coordinator = NegotiationCoordinator(num_agents=num_agents, num_rounds=num_rounds)
    
    proposals = await coordinator.collect_proposals(message_queue)
    winning_output, winning_weight, justifs = coordinator.perform_voting(proposals)
    print(f"\n[Coordinator] Initial Winning Output: {winning_output} with weighted vote {winning_weight:.2f}")
    print(f"[Coordinator] Initial Justifications: {justifs}")
    
    proposals = await negotiation_rounds(num_rounds, agents, coordinator, message_queue)
    
    final_winning_output, final_winning_weight, final_justifs = coordinator.perform_voting(proposals)
    print(f"\n[Coordinator] Final Winning Output: {final_winning_output} with weighted vote {final_winning_weight:.2f}")
    print(f"[Coordinator] Final Justifications: {final_justifs}")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
