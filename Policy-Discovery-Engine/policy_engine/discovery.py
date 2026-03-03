import authzed.api.v1 as zed
from authzed.api.v1 import Client, Relationship, ObjectRef, SubjectReference

class GraphDiscoveryEngine:
    def __init__(self, token="somerandomkey", endpoint="localhost:50051"):
        self.client = Client(endpoint, token, insecure=True)

    def learn_edge(self, agent_role: str, tool_name: str, capability: str = "can_execute"):
        """
        Observes a successful tool call and writes the 'Happy Path' to the Graph.
        """
        print(f"[Discovery] Learning: Role '{agent_role}' needs access to '{tool_name}'")
        
        # Construct the Tuple: tool:name # capability @ role:name
        rel = Relationship(
            resource=ObjectRef(object_type="tool", object_id=tool_name),
            relation=capability,
            subject=SubjectReference(object=ObjectRef(object_type="role", object_id=agent_role))
        )
        
        # Write to SpiceDB
        self.client.WriteRelationships(updates=[rel])

    def define_risk(self, repo_id: str, score: int):
        """
        Defines the risk level of a data source.
        """
        # Mapping logic would go here (Resource -> Risk Node)
        pass
