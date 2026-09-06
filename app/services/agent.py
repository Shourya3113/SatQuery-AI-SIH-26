import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Any
from app.services.data_validator import ValidationResult

class RouterExecutionTrace(BaseModel):
    selected_tool: str = Field(description="The specialist tool selected: 'single_image_vqa_tool', 'bitemporal_change_tool', or 'optical_sar_fusion_tool'.")
    reasoning: str = Field(description="Explanation of why this tool was selected based on the query and image metadata.")
    inputs: List[str] = Field(description="List of filenames to be passed as inputs to the tool.")

def get_agent_router_trace(query: str, validation_result: ValidationResult) -> dict:
    """
    Uses LangChain to determine which specialist tool to use based on the user query and image metadata.
    """
    
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        # Using Gemini 1.5 Flash which has a generous free tier!
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    except Exception as e:
        return {
            "selected_tool": "error",
            "reasoning": f"Failed to initialize LLM (e.g., missing API key): {str(e)}",
            "inputs": []
        }
    
    parser = JsonOutputParser(pydantic_object=RouterExecutionTrace)
    
    prompt = PromptTemplate(
        template="""You are the agentic router for SatQuery AI, a multi-modal vision-language assistant for remote sensing.
Your job is to read the user query and the geospatial image metadata, and decide which specialist tool to execute.

Available Tools:
1. single_image_vqa_tool: For optical/SAR captioning, text-guided grounding, and queries involving a single image.
2. bitemporal_change_tool: For comparing two images of the same area over time (requires exactly 2 co-registered images).
3. optical_sar_fusion_tool: For cross-modal joint reasoning (typically requires 2 co-registered images of different modalities, e.g., one optical, one SAR).

Image Metadata:
{metadata}

User Query:
{query}

Determine the best tool to answer the user query.
{format_instructions}
""",
        input_variables=["query", "metadata"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    # Convert metadata to a JSON string for the prompt
    metadata_str = json.dumps([m.model_dump() for m in validation_result.metadata], indent=2)
    
    try:
        # In a real async setup, we'd use chain.ainvoke
        result = chain.invoke({"query": query, "metadata": metadata_str})
        return result
    except Exception as e:
        return {
            "selected_tool": "error",
            "reasoning": f"Failed to parse or execute LLM routing: {str(e)}",
            "inputs": []
        }
