from llama_index.core.graph_stores.types import PropertyGraphStore
from pydantic import BaseModel
from typing import Optional, List,Any

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import CustomPGRetriever, VectorContextRetriever
from llama_index.core.vector_stores.types import VectorStore
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.query_engine import RetrieverQueryEngine

from module.data_extractor.data_extractor import DataExtractor,modelClientConnService

from llama_index.core.schema import (NodeWithScore)


class Entities(BaseModel):
    """รายชื่อของ named entities ในข้อความ เช่น ชนิดต้นไม้ ลักษณะดิน เนื้อหาย่อย และสถานที่"""
    ตำรับที่: Optional[List[str]]
    ชื่อตำรับ: Optional[List[str]]
    สรรพคุณของตำรับ: Optional[List[str]]
    ส่วนประกอบของตำรับ: Optional[List[str]]
    วิธีปรุงยา: Optional[List[str]]
    รูปแบบยา: Optional[List[str]]
    วิธีใช้: Optional[List[str]]



class Retriever(CustomPGRetriever):
    """Custom retriever with cohere reranking."""

    def __init__(self, graph_store: PropertyGraphStore, include_text: bool = False, **kwargs: Any) -> None:
        super().__init__(graph_store, include_text, **kwargs)

    prompt_template_entities = """
    แยก named entities ทั้งหมด เช่น ตำรับที่ ชื่อตำรับ สรรพคุณของตำรับ ส่วนประกอบของตำรับ วิธีปรุงยา รูปแบบยา วิธีใช้
    จากข้อความต่อไปนี้:
    {text}
    """

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
            output_cls=Entities, prompt_template_str=self.prompt_template_entities,
            llm=modelClientConnService.lamaModelClient
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
        entities = self.entity_extraction(text=query_str)
        result_nodes:List[NodeWithScore] = []

        all_entities = []
        if entities.ตำรับที่:
            all_entities.extend(entities.ตำรับที่)
        if entities.ชื่อตำรับ:
            all_entities.extend(entities.ชื่อตำรับ)
        if entities.สรรพคุณของตำรับ:
            all_entities.extend(entities.สรรพคุณของตำรับ)
        if entities.ส่วนประกอบของตำรับ:
            all_entities.extend(entities.ส่วนประกอบของตำรับ)
        if entities.วิธีปรุงยา:
            all_entities.extend(entities.วิธีปรุงยา)
        if entities.รูปแบบยา:
            all_entities.extend(entities.รูปแบบยา)
        if entities.วิธีใช้:
            all_entities.extend(entities.วิธีใช้)
            
        if all_entities:
            for entity in all_entities:
                result_nodes.extend(self.vector_retriever.retrieve(entity))
        else:
            result_nodes.extend(self.vector_retriever.retrieve(query_str))
        
        final_text = "\n\n".join(
            [n.get_content(metadata_mode="llm") for n in result_nodes]
        )
        return final_text


class kgRetriever:

    def __init__(self) -> None:
        self.kgRetrieverClient = Retriever(
            DataExtractor.index.property_graph_store,
            include_text=True,
            vector_store=DataExtractor.index.vector_store,
            embed_model=modelClientConnService.embedModel
        )

        self.queryEngine = RetrieverQueryEngine.from_args(
            DataExtractor.index.as_retriever(sub_retrievers=[self.kgRetrieverClient]), llm=modelClientConnService.lamaModelClient
        )


