from typing import List
from typing import TypeVar

from llama_index.core import Document
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from graph_config.graph_config import GraphConfiguration
from service.db_client.connection_service import neoConnection
from service.model_client_conn.model_client_conn import modelClientConnService

IndexType = TypeVar("IndexType", bound="RangeIndex") # type: ignore


entities = GraphConfiguration.entities
relations = GraphConfiguration.relations
validation_schema = GraphConfiguration.validation_schema

similarity_threshold = 0.9
word_edit_distance = 5

class DataExtractor:
    kg_extractor = SchemaLLMPathExtractor(
        llm=modelClientConnService.lamaModelClient,
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=validation_schema,
        num_workers=10,
        strict=True,
    )

    index:PropertyGraphIndex = PropertyGraphIndex.from_existing(
        kg_extractors=[kg_extractor],
            llm=modelClientConnService.lamaModelClient,
            embed_model=modelClientConnService.embedModel,
            property_graph_store=neoConnection.llamaDriver,
            show_progress=True,
    )


    @staticmethod
    def extractData(documents:List[Document]):
        
        DataExtractor.index = PropertyGraphIndex.from_documents(
            documents,
            kg_extractors=[DataExtractor.kg_extractor],
            llm=modelClientConnService.lamaModelClient,
            embed_model=modelClientConnService.embedModel,
            property_graph_store=neoConnection.llamaDriver,
            show_progress=True,
        )