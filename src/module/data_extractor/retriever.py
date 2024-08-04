from pydantic import BaseModel
from typing import Optional, List

from typing import Any, Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import CustomPGRetriever, VectorContextRetriever
from llama_index.core.vector_stores.types import VectorStore
from llama_index.program.openai import OpenAIPydanticProgram


class Entities(BaseModel):
    """List of named entities in the text such as names of people, organizations, concepts, and locations"""
    names: Optional[List[str]]


prompt_template_entities = """
Extract all named entities such as names of people, organizations, concepts, and locations
from the following text:
{text}
"""


class Retriever(CustomPGRetriever):
    """Custom retriever with cohere reranking."""

    def init(
        self,
        ## vector context retriever params
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        similarity_top_k: int = 4,
        path_depth: int = 1,
        include_text: bool = True,
        **kwargs: Any,
    ) -> None:
        """Uses any kwargs passed in from class constructor."""
        self.entity_extraction = OpenAIPydanticProgram.from_defaults(
            output_cls=Entities, prompt_template_str=prompt_template_entities
        )
        self.vector_retriever = VectorContextRetriever(
            self.graph_store,
            include_text=self.include_text,
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
            path_depth=path_depth,
        )

    def custom_retrieve(self, query_str: str) -> str:
        """Define custom retriever with reranking.

        Could return `str`, `TextNode`, `NodeWithScore`, or a list of those.
        """
        entities = self.entity_extraction(text=query_str).names
        result_nodes = []
        if entities:
            print(f"Detected entities: {entities}")
            for entity in entities:
                result_nodes.extend(self.vector_retriever.retrieve(entity))
        else:
            result_nodes.extend(self.vector_retriever.retrieve(query_str))
        ## TMP: please change
        final_text = "\n\n".join(
            [n.get_content(metadata_mode="llm") for n in result_nodes]
        )
        return final_text
