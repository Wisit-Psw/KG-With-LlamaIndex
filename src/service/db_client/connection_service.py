import os
from neo4j import GraphDatabase
from llama_index.graph_stores.neo4j import Neo4jPGStore

NEO4J_URI = os.getenv("NEO4J_HOST")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASS")

class NeoConnectionService:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(
                NEO4J_USER, 
                NEO4J_PASSWORD
            )
        )

        self.llamaDriver = Neo4jPGStore(
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                url=NEO4J_URI,
            )

neoConnection = NeoConnectionService()