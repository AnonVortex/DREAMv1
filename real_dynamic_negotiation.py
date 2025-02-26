# by Alexis Soto-Yanez
"""
Real Dynamic Negotiation Prototype

This script implements a robust dynamic negotiation mechanism for a team of agents.
Each agent sends an initial proposal (including output, confidence, and justification),
then repeatedly updates its proposal based on the aggregated proposals of its peers.
Negotiation rounds continue until consensus is reached or a maximum number of rounds is exceeded.
"""

import asyncio
import random
from dataclasses import dataclass
import statistics
import nest_asyncio

@dataclass
class Proposal:
    agent_id: str
    output: str
    confidence: float
    justification: str

class RealAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.proposal = None

    async def send_initial_proposal(self, message_queue: asyncio.Queue):
        # Simulate a processing delay.
        await asyncio.sleep(random.uniform(0.2, 0.5))
        output = f"{self.agent_id}_output"
        confidence = random.uniform(0.6, 1.0)
        justification = f"Initial confidence {confidence:.2f} based on internal assessment."
        self.proposal = Proposal(self.agent_id, output, confidence, justification)
        print(f"[{self.agent_id}] Initial proposal: {self.proposal}")
        await message_queue.put(self.proposal)

    async def update_proposal(self, aggregated_proposals: list, round_number: int):
        # Simulate processing delay.
        await asyncio.sleep(random.uniform(0.2, 0.5))
        # Determine the majority output from the aggregated proposals.
        outputs = [p.output for p in aggregated_proposals]
        freq = {}
        for out in outputs:
            freq[out] = freq.get(out, 0) + 1
        majority_output = max(freq, key=freq.get)
        
        # Update confidence: if my output agrees with the majority, boost confidence; otherwise, lower it.
        if self.proposal.output == majority_output:
            # Increase confidence slightly.
            adjustment = random.uniform(0.02, 0.05)
            note = "Agreed with majority; increasing confidence."
        else:
            # Decrease confidence if not in agreement.
            adjustment = -random.uniform(0.05, 0.1)
            note = "Disagreed with majority; decreasing confidence."
            # With a probability, consider switching to the majority output.
            if random.random() < 0.3:
                note += " Switched output to majority."
                self.proposal.output = majority_output

        new_confidence = max(0.0, min(1.0, self.proposal.confidence + adjustment))
        justification = f"Round {round_number}: {note} New confidence {new_confidence:.2f}."
        self.proposal = Proposal(self.agent_id, self.proposal.output, new_confidence, justification)
        print(f"[{self.agent_id}] Updated proposal in round {round_number}: {self.proposal}")
        return self.proposal

class RealNegotiationCoordinator:
    def __init__(self, agents_count: int, max_rounds: int = 5, consensus_threshold: float = 0.1):
        self.agents_count = agents_count
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.proposals = []

    def aggregate_votes(self, proposals: list):
        vote_counts = {}
        for p in proposals:
            vote_counts.setdefault(p.output, 0)
            vote_counts[p.output] += p.confidence
        return vote_counts

    def get_majority_output(self, proposals: list):
        freq = {}
        for p in proposals:
            freq[p.output] = freq.get(p.output, 0) + 1
        majority_output = max(freq, key=freq.get)
        majority_fraction = freq[majority_output] / len(proposals)
        return majority_output, majority_fraction

    def consensus_reached(self, proposals: list):
        # Consensus if all agents propose the same output.
        outputs = [p.output for p in proposals]
        if len(set(outputs)) == 1:
            return True
        # Or if confidence variation is very low.
        confidences = [p.confidence for p in proposals]
        if statistics.pstdev(confidences) < self.consensus_threshold:
            return True
        return False

async def negotiation_round(coordinator: RealNegotiationCoordinator, agents: list, round_number: int):
    proposals = []
    for agent in agents:
        updated_proposal = await agent.update_proposal(coordinator.proposals, round_number)
        proposals.append(updated_proposal)
    coordinator.proposals = proposals
    vote_counts = coordinator.aggregate_votes(proposals)
    majority_output, majority_fraction = coordinator.get_majority_output(proposals)
    print(f"[Coordinator] Round {round_number} Vote counts: {vote_counts}")
    print(f"[Coordinator] Round {round_number} Majority output: {majority_output} ({majority_fraction*100:.1f}% agreement)")
    return proposals

async def run_real_negotiation():
    num_agents = 4
    max_rounds = 5
    message_queue = asyncio.Queue()
    agents = [RealAgent(f"Agent{i+1}") for i in range(num_agents)]
    coordinator = RealNegotiationCoordinator(agents_count=num_agents, max_rounds=max_rounds)

    # Send initial proposals concurrently.
    initial_tasks = [agent.send_initial_proposal(message_queue) for agent in agents]
    await asyncio.gather(*initial_tasks)

    proposals = []
    for _ in range(num_agents):
        p = await message_queue.get()
        proposals.append(p)
    coordinator.proposals = proposals
    vote_counts = coordinator.aggregate_votes(proposals)
    majority_output, majority_fraction = coordinator.get_majority_output(proposals)
    print(f"\n[Coordinator] Initial Vote counts: {vote_counts}")
    print(f"[Coordinator] Initial Majority output: {majority_output} ({majority_fraction*100:.1f}% agreement)")

    # Perform negotiation rounds until consensus or max rounds reached.
    round_number = 1
    while round_number <= max_rounds and not coordinator.consensus_reached(coordinator.proposals):
        print(f"\n--- Negotiation Round {round_number} ---")
        await negotiation_round(coordinator, agents, round_number)
        round_number += 1

    final_vote_counts = coordinator.aggregate_votes(coordinator.proposals)
    final_majority_output, _ = coordinator.get_majority_output(coordinator.proposals)
    print(f"\n[Coordinator] Final Vote counts: {final_vote_counts}")
    print(f"[Coordinator] Final Consensus Output: {final_majority_output}")
    return final_majority_output

async def main():
    final_output = await run_real_negotiation()
    print(f"\nFinal Negotiated Output: {final_output}")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
