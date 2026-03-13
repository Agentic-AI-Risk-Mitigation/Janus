"""
Graph discovery engine -- learns tool-access edges from observed behaviour.

Watches successful tool calls and writes the "happy path" relationships
to SpiceDB so they can be enforced later.
"""

from authzed.api.v1 import Client, ObjectReference, Relationship, SubjectReference
from grpcutil import insecure_bearer_token_credentials


class GraphDiscoveryEngine:
    def __init__(self, token: str = "somerandomkey", endpoint: str = "localhost:50051"):
        self.client = Client(endpoint, insecure_bearer_token_credentials(token))

    def learn_edge(
        self,
        agent_role: str,
        tool_name: str,
        capability: str = "can_execute",
    ) -> None:
        """Observes a successful tool call and writes the relationship to the graph."""
        print(f"[Discovery] Learning: Role '{agent_role}' needs access to '{tool_name}'")

        rel = Relationship(
            resource=ObjectReference(object_type="tool", object_id=tool_name),
            relation=capability,
            subject=SubjectReference(
                object=ObjectReference(object_type="role", object_id=agent_role),
            ),
        )

        self.client.WriteRelationships(updates=[rel])

    def define_risk(self, repo_id: str, score: int) -> None:
        """Defines the risk level of a data source (stub)."""
