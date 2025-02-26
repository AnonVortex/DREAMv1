# by Alexis Soto-Yanez
"""
Federation Collaboration for HMAS (AGI Prototype)

This script extends the hierarchical collaboration model to include multiple organizations
that work together to produce a single final output.
"""

# Reusing the BaseAgent, Team, and Organization classes from earlier.

class BaseAgent:
    def __init__(self, name):
        self.name = name

    def execute(self, data):
        result = f"{self.name} processed [{data}]"
        return result

class Team:
    def __init__(self, name, agents):
        self.name = name
        self.agents = agents

    def collaborate(self, data):
        team_results = {}
        for agent in self.agents:
            result = agent.execute(data)
            team_results[agent.name] = result
        aggregated_result = f"Team {self.name} aggregated: " + " | ".join(team_results.values())
        print(f"[Team: {self.name}] Collaboration complete.")
        return aggregated_result

class Organization:
    def __init__(self, name, teams):
        self.name = name
        self.teams = teams

    def coordinate(self, data):
        org_results = {}
        for team in self.teams:
            result = team.collaborate(data)
            org_results[team.name] = result
        aggregated_org_result = f"Organization {self.name} output: " + " || ".join(org_results.values())
        print(f"[Organization: {self.name}] Coordination complete.")
        return aggregated_org_result

# New: Federation class that aggregates multiple organizations.
class Federation:
    def __init__(self, organizations):
        """
        :param organizations: List of Organization instances.
        """
        self.organizations = organizations

    def collaborate(self, data):
        federation_results = {}
        for org in self.organizations:
            result = org.coordinate(data)
            federation_results[org.name] = result
        aggregated_federation_output = "Federation Final Output: " + " ### ".join(
            f"{org}: {result}" for org, result in federation_results.items()
        )
        print("[Federation] Collaboration among organizations complete.")
        return aggregated_federation_output

# ----- Example Usage -----
if __name__ == "__main__":
    # Create agents for the first organization.
    agent1 = BaseAgent("Agent1_Vision")
    agent2 = BaseAgent("Agent2_Audition")
    agent3 = BaseAgent("Agent3_Smell")
    agent4 = BaseAgent("Agent4_Touch")
    agent5 = BaseAgent("Agent5_Taste")
    
    # Form teams for the first organization.
    perception_team = Team("PerceptionTeam", [agent1, agent2])
    sensor_team = Team("SensorTeam", [agent3, agent4, agent5])
    organization1 = Organization("Organization1", [perception_team, sensor_team])
    
    # Create agents for the second organization.
    agent6 = BaseAgent("Agent6_Vision")
    agent7 = BaseAgent("Agent7_Audition")
    agent8 = BaseAgent("Agent8_Smell")
    agent9 = BaseAgent("Agent9_Touch")
    agent10 = BaseAgent("Agent10_Taste")
    
    # Form teams for the second organization.
    perception_team2 = Team("PerceptionTeam2", [agent6, agent7])
    sensor_team2 = Team("SensorTeam2", [agent8, agent9, agent10])
    organization2 = Organization("Organization2", [perception_team2, sensor_team2])
    
    # Create a federation of the two organizations.
    federation = Federation([organization1, organization2])
    
    # Simulate an input that the federation needs to process.
    input_data = "raw_multimodal_input"
    
    # The federation coordinates the collaboration of both organizations.
    final_output = federation.collaborate(input_data)
    
    print("\nFederation Final Output:")
    print(final_output)
