# ========== Imports 

# Util
from typing_extensions import TypedDict
from typing import Annotated
from typing import Annotated, Literal
import os
from dotenv import load_dotenv, find_dotenv
import re
import sqlite3
from src.rag_agent import get_retriever
from pathlib import Path
import sqlite3
import os
# langchain
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END

#from langgraph.checkpoint.postgres import PostgresSaver

from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# GCP
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import Image
from google.cloud import aiplatform

# ========== Start

load_dotenv(dotenv_path=find_dotenv('.env.development'))

project_id = os.getenv('PROJECT_ID')
location = os.getenv('LOCATION')
endpoint_id = os.getenv('ENDPOINT_ID')
endpoint_id2 = os.getenv('ENDPOINT_ID2')

memory = MemorySaver()

BASE_DIR = Path(__file__).resolve().parent.parent  # back/
STATE_DIR = BASE_DIR / "state_db"
STATE_DIR.mkdir(parents=True, exist_ok=True)      # crea la carpeta si no existe
DB_PATH = STATE_DIR / "example.db"   
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
sqlite_saver = SqliteSaver(conn)

llm = ChatOpenAI(model="gpt-4o")

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    userQuestion: str
    localQuestion: str
    hasVisitedInvestigator: bool
    hasVisitedCreator: bool
    hasVisitedEvaluator: bool
    nextNode: Literal["investigator", "creator", "evaluator", "asr", "unifier"]
    imagePath1: str
    imagePath2: str
    endMessage: str
    hasVisitedASR: bool 
    mermaidCode: str
    turn_messages:list

class AgentState(TypedDict):
    messages: list
    userQuestion: str
    localQuestion: str
    imagePath1: str
    imagePath2: str
    
builder = StateGraph(GraphState)

class supervisorResponse(TypedDict):
    localQuestion: Annotated[str, ..., "What is the question for the worker node?"]
    nextNode: Literal["investigator", "creator", "evaluator", "asr", "unifier"]

supervisorSchema = {
    "title": "SupervisorResponse",
    "description": "Response from the supervisor indicating the next node and the setup question.",
    "type": "object",
    "properties": {
        "localQuestion": {
            "type": "string",
            "description": "What is the question for the worker node?"
        },
        "nextNode": {
            "type": "string",
            "description": "The next node to act.",
            "enum": ["investigator", "creator", "evaluator", "unifier", "asr"]
        }
    },
    "required": ["localQuestion", "nextNode"]
}

class evaluatorResponse(TypedDict):
    positiveAspects: Annotated[str, ..., "What are the positive aspects of the user's idea?"]
    negativeAspects: Annotated[str, ..., "What are the negative aspects of the user's idea?"]
    suggestions: Annotated[str, ..., "What are the suggestions for improvement?"]

evaluatorSchema = {
    "title": "EvaluatorResponse",
    "description": "Response from the evaluator indicating the positive and negative aspects of the user's idea.",
    "type": "object",
    "properties": {
        "positiveAspects": {
            "type": "string",
            "description": "What are the positive aspects of the user's idea?"
        },
        "negativeAspects": {
            "type": "string",
            "description": "What are the negative aspects of the user's idea?"
        },
        "suggestions": {
            "type": "string",
            "description": "What are the suggestions for improvement?"
        }
    },
    "required": ["positiveAspects", "negativeAspects", "suggestions"]
}

class investigatorResponse(TypedDict):
    definition: Annotated[str, ..., "What is the definition of the concept?"]
    useCases: Annotated[str, ..., "What are the use cases of the concept?"]
    examples: Annotated[str, ..., "What are the examples of the concept?"]

investigatorSchema = {
    "title": "InvestigatorResponse",
    "description": "Response from the investigator indicating the definition, use cases, and examples of the concept.",
    "type": "object",
    "properties": {
        "definition": {
            "type": "string",
            "description": "What is the definition of the concept?"
        },
        "useCases": {
            "type": "string",
            "description": "What are the use cases of the concept?"
        },
        "examples": {
            "type": "string",
            "description": "What are the examples of the concept?"
        }
    },
    "required": ["definition", "useCases", "examples"]
}

retriever = get_retriever()

# ========== Prompts 

# ===== Nodes

def makeSupervisorPrompt(state: GraphState) -> str:
    visited_nodes = []
    if state["hasVisitedInvestigator"]:
        visited_nodes.append("investigator")
    if state["hasVisitedCreator"]:
        visited_nodes.append("creator")
    if state["hasVisitedEvaluator"]:
        visited_nodes.append("evaluator")
    if state.get("hasVisitedASR", False):
        visited_nodes.append("asr")

    visited_nodes_str = ", ".join(visited_nodes) if visited_nodes else "none"

    supervisorPrompt = f"""You are a supervisor tasked with managing a conversation between the following workers: investigator, 
    diagrams, creator, evaluator, and ASR advisor. Given the following user request, respond with the worker to act next. 
    Each worker will perform a task and respond with their results and status.
    
    Important flows:
    - If the user just asks about ADD, use the investigator.
    - If the user wants to extract or describe a diagram, first classify or extract the diagram elements.
    - If the user provides an ASR and limitations (and optionally an image of an implementation), route to the ASR node.
    - If the user wants a diagram, route to the creator node.
    
    Visited nodes so far: {visited_nodes_str}.
    
    This is the user question: {state["userQuestion"]}
    
    The possible outputs: ['investigator', 'creator', 'evaluator', 'asr', 'unifier'].

    In case there is nothing else to do go to unifier.
    
    You also need to define a specific question for the node you send:
      - For the investigator node: ask for concepts or patterns in the user diagrams.
      - For the creator node: ask to generate a diagram or code example.
      - **For the ASR node: always prioritize it if the user provides an ASR and limitations in the question.**
        - If no diagram is provided, ask for recommendations on how to implement the ASR given the limitations.  
        - If a diagram is provided, ask to evaluate whether the implementation meets the ASR and adheres to the limitations.
      - For the evaluator node: ask to evaluate the user's ideas, especially if two diagrams are provided.
    
    NOTE: YOU CANNOT GO DIRECTLY TO THE UNIFIER NODE; you must go to at least one worker node before.
    """
    return supervisorPrompt

prompt_researcher = """You are an expert in software architecture, specializing in Attribute Driven Design (ADD) and tactics related to availability 
    and performance. Your task is to analyze the user's question and provide an accurate and well-explained response based on your expertise. 
    You have access to two tools to assist you in answering:

    - 'LLM': A powerful large language model fine-tuned for software architecture-related queries. Use this tool when you need 
      a detailed explanation, best practices, or general knowledge about architecture principles.
    - 'LLMWithImages': A large language model with image support, allowing you to analyze diagrams, patterns, and visual representations    
    - 'local_RAG': A local Retrieval-Augmented Generation (RAG) system with access to a curated knowledge base of software architecture 
      documentation, use this tool when you need to retrieve information about performance and scalability
"""

prompt_creator = """you are an expert in mermade diagrams and IT architecture, you will be given a prompt 
and you will generate the mermade diagram for it."""

def getEvaluatorPrompt(image_path1: str, image_path2) -> str:
    image1 = ""
    image2 = ""

    if image_path1:
        image1 = "this is the first image path: " + image_path1
    if image_path2:
        image2 = "this is the second image path: " + image_path2

    evaluatorPrompt = f"""You are an expert in software architecture evaluation, specializing in assessing project feasibility and analyzing 
        the strengths and weaknesses of proposed strategies. Your role is to critically evaluate the user's request and provide a well-informed 
        assessment based on two specialized tools:

        - `Theory Tool` for correctness checks.
        - `Viability Tool` for feasibility assessment.
        - `Needs Tool` for requirement alignment.
        - `Analyze Tool` for comparing two diagrams.
        {image1}
        {image2}
        """
    return evaluatorPrompt

def _push_turn(state: GraphState, role: str, name: str, content: str) -> None:
    """Guarda una línea en el log del turno actual (modal de 'Mensajes Internos')."""
    line = {"role": role, "name": name, "content": content}
    state["turn_messages"] = state.get("turn_messages", []) + [line]

def _reset_turn(state: GraphState) -> GraphState:
    """Reinicia el buffer del turno (se llama al inicio de la ejecución)."""
    return {**state, "turn_messages": []}


# ===== Tools

llm_prompt = "Retrieve general software architecture knowledge. Answer concisely and focus on key concepts: fill the 3 sections of the anwser allways: [definition, useCases, examples]"

llmWithImages_prompt = """Analyze the diagram and provide a detailed explanation of the software architecture tactics found in the image. 
    Focus on performance and availability tactics."""

rag_prompt = "retrieve information about scalability and performance. There is information about the definition, patterns and tactics with examples."

theory_prompt = "Analyze the theoretical correctness of this architecture diagram. Follow best practices."

viability_prompt = "Evaluate the feasibility of the user's ideas. Provide a detailed analysis of the viability of the proposed strategies."

needs_prompt = "Analyze the user's requirements and check if they align with the proposed architecture. Focus on the user's needs."

analyze_prompt = """Analyze the following pair of diagrams:
    A class diagram representing the implementation of a component,
    A component diagram that places this component within the architectural context.
    Evaluate whether the component's implementation (class diagram) is properly designed to support quality attributes such as scalability and 
    performance, among others.
    Provide a detailed assessment highlighting strengths, deficiencies, and improvement suggestions."""

# ========== Tools 

# ===== Investigator

@tool
def LLM(prompt: str) -> str:
    """This researcher is able of answering questions only about Attribute Driven Design, also known as ADD or ADD 3.0.
    Remember the context is software architecture, don't confuse Attribute Driven Design with Attention-Deficit Disorder."""
    response = llm.with_structured_output(investigatorSchema).invoke(prompt)
    return response

@tool
def LLMWithImages(image_path: str) -> str:
    """This researcher is able of answering questions about software architecture diagrams, patterns, and visual representations.
    Remember to focus on performance and availability tactics, and always use the image as a reference."""
    image = Image.load_from_file(image_path)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    response = generative_multimodal_model.generate_content([
        "What software architecture tactics can you see in this diagram? "
        "If it is a class diagram, analyze and evaluate it by identifying classes, attributes, methods, relationships, "
        "Object-Oriented Design principles, and design patterns.", 
        image
    ])
    return response

# ===== Investigator tools (reemplaza SOLO local_RAG) =====
@tool
def local_RAG(prompt: str) -> str:
    """This researcher answers about performance & scalability using local documents.
    It returns a short synthesis followed by a SOURCES block for UI consumption."""
    docs = retriever.invoke(prompt)
    if not docs:
        return "No local documents were retrieved for this query."

    # pequeño resumen con los primeros documentos (evita respuestas larguísimas)
    preview_chunks = []
    for i, d in enumerate(docs[:3], start=1):
        snippet = (d.page_content or "")[:600].strip()
        preview_chunks.append(f"[{i}] {snippet}")

    # fuentes abreviadas y normalizadas
    src_lines = []
    for d in docs:
        title = d.metadata.get("title") or Path(d.metadata.get("source_path", "")).stem or "doc"
        page = d.metadata.get("page_label") or d.metadata.get("page")
        src = d.metadata.get("source_path") or d.metadata.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str} — {src}")

    body = "\n\n".join(preview_chunks)
    sources = "\n".join(src_lines)
    return f"""{body}

SOURCES:
{sources}
"""

# ===== Evaluator

@tool
def theory_tool(prompt: str) -> str:
    """This evaluator is able to check the theoretical correctness of the architecture diagram. It follows best practices and provides a detailed analysis."""
    response = llm.with_structured_output(evaluatorSchema).invoke(theory_prompt + prompt)
    return response

@tool
def viability_tool(prompt: str) -> str:
    """This evaluator is able to check the feasibility of the user's ideas. It provides a detailed analysis of the viability of the proposed strategies."""
    response = llm.with_structured_output(evaluatorSchema).invoke(viability_prompt + prompt)
    return response

@tool
def needs_tool(prompt: str) -> str:
    """This evaluator is able to check the user's requirements and verify if they align with the proposed architecture. It focuses on the user's needs."""
    response = llm.with_structured_output(evaluatorSchema).invoke(needs_prompt + prompt)
    return response

@tool
def analyze_tool(image_path: str, image_path2: str) -> str:
    """This evaluator is able to compare two diagrams: a class diagram representing the implementation 
    of a component and a component diagram that places this component within the architectural context. 
    It evaluates whether the component's implementation (class diagram) is properly designed to support 
    quality attributes such as scalability and performance, among others. It provides a detailed assessment 
    highlighting strengths, deficiencies, and improvement suggestions."""
    image = Image.load_from_file(image_path)
    image2 = Image.load_from_file(image_path2)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    response = generative_multimodal_model.generate_content([
        analyze_prompt,
        image,
        image2
    ])
    return response

# ========== Router 

def router(state: GraphState) -> Literal["investigator", "creator", "evaluator", "asr", "unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"
    elif state["nextNode"] == "asr" and not state.get("hasVisitedASR", False):
        return "asr"
    elif state["nextNode"] == "investigator" and not state["hasVisitedInvestigator"]:
        return "investigator"
    elif state["nextNode"] == "creator" and not state["hasVisitedCreator"]:
        return "creator"
    elif state["nextNode"] == "evaluator" and not state["hasVisitedEvaluator"]:
        return "evaluator"
    else:
        return "unifier"

# ========== Nodes definition 

# ===== Supervisor

def supervisor_node(state: GraphState):
    message = [
        {"role": "system", "content": makeSupervisorPrompt(state)},
    ]
    response = llm.with_structured_output(supervisorSchema).invoke(message)

    next_node = response["nextNode"]
    # 👉 Fallback: si aún no pasó por ningún worker, fuerza uno razonable
    if next_node == "unifier" and not (
        state["hasVisitedInvestigator"] or state["hasVisitedCreator"] or
        state["hasVisitedEvaluator"] or state.get("hasVisitedASR", False)
    ):
        uq = (state.get("userQuestion") or "").lower()
        if state.get("imagePath1") or state.get("imagePath2") or "diagram" in uq or "mermaid" in uq:
            next_node = "creator"          # prioriza crear diagrama si lo piden
        else:
            next_node = "investigator"     # si no, investigar primero

    state_updated: GraphState = {
        **state,
        "localQuestion": response["localQuestion"],
        "nextNode": next_node,
    }
    return state_updated

# ===== Investigator
    
researcher_agent = create_react_agent(llm, tools=[LLM, LLMWithImages, local_RAG])

def researcher_node(state: GraphState) -> GraphState:
    system_message = SystemMessage(content=prompt_researcher)
    _push_turn(state, role="system", name="researcher_system", content=prompt_researcher)

    messages_with_system = [system_message] + state["messages"]
    result = researcher_agent.invoke({
        "messages": messages_with_system,
        "userQuestion": state["userQuestion"],
        "localQuestion": state["localQuestion"],
        "imagePath1": state["imagePath1"],
        "imagePath2": state["imagePath2"]
    })

    # Guarda todas las salidas del agente como parte del log del turno
    for msg in result["messages"]:
        # msg.content ya es texto (para tool calls puede venir estructurado)
        _push_turn(state, role="assistant", name="researcher", content=str(getattr(msg, "content", msg)))

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="researcher") for msg in result["messages"]],
        "hasVisitedInvestigator": True
    }

# ===== Creator
def creator_node(state: GraphState) -> GraphState:
    _push_turn(state, role="system", name="creator_system", content=prompt_creator + state["userQuestion"])

    response = llm.invoke(prompt_creator + state["userQuestion"])

    match = re.search(r"```mermaid\n(.*?)```", response.content, re.DOTALL)
    mermaid_code = match.group(1) if match else ""

    _push_turn(state, role="assistant", name="creator", content=response.content)

    return {
        **state,
        "messages": state["messages"] + [response],
        "mermaidCode": mermaid_code,
        "hasVisitedCreator": True
    }

# ===== Evaluator
def evaluator_node(state: GraphState) -> GraphState:
    evaluator_agent = create_react_agent(llm, tools=[theory_tool, viability_tool, needs_tool, analyze_tool])

    eval_prompt = getEvaluatorPrompt(state["imagePath1"], state["imagePath2"])
    _push_turn(state, role="system", name="evaluator_system", content=eval_prompt)

    messages_with_system = [SystemMessage(content=eval_prompt)] + state["messages"]
    result = evaluator_agent.invoke({
        "messages": messages_with_system,
        "userQuestion": state["userQuestion"],
        "localQuestion": state["localQuestion"],
        "imagePath1": state["imagePath1"],
        "imagePath2": state["imagePath2"]
    })

    for msg in result["messages"]:
        _push_turn(state, role="assistant", name="evaluator", content=str(getattr(msg, "content", msg)))

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="evaluator") for msg in result["messages"]],
        "hasVisitedEvaluator": True
    }

# ===== Unifier
def unifier_node(state: GraphState) -> GraphState:
    # Toma la última contribución por rol (si existe)
    def last_by(name: str) -> str:
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.name == name and m.content:
                return m.content
        return ""

    parts = []
    parts.append(f"User question:\n{state['userQuestion']}")
    for role in ["researcher", "evaluator", "asr_evaluator", "asr_recommender", "creator"]:
        c = last_by(role)
        if c:
            parts.append(f"{role} says:\n{c}")

    synthesis_source = "\n\n".join(parts)

    prompt = f"""You are an expert assistant in information synthesis.
Format your response in multiple short paragraphs (2–4). 
Do not repeat earlier turns verbatim. Be concise and only use the content below.

=== SOURCE BEGIN ===
{synthesis_source}
=== SOURCE END ===

Important:
- Keep only NEW synthesis for the current turn.
- Do not restate the full context; avoid redundancy.
- NEVER include mermaid code here.
"""
    response = llm.invoke(prompt)
    return {**state, "endMessage": response.content}
# ===== ASR
def asr_node(state: GraphState) -> GraphState:
    if state["imagePath1"]:
        prompt = f"""You are an expert in software architecture implementation evaluation.
{state["userQuestion"]}
An implementation diagram is available at: {state["imagePath1"]}.
..."""
        _push_turn(state, role="system", name="asr_system", content=prompt)
        result = llm.invoke(prompt)
        message = AIMessage(content=result.content, name="asr_evaluator")
    else:
        prompt = f"""You are an expert in providing recommendations for software architecture.
{state["userQuestion"]}
..."""
        _push_turn(state, role="system", name="asr_system", content=prompt)
        result = llm.invoke(prompt)
        message = AIMessage(content=result.content, name="asr_recommender")

    _push_turn(state, role="assistant", name=str(message.name), content=message.content)

    return {
        **state,
        "messages": state["messages"] + [message],
        "hasVisitedASR": True
    }

# ========== Nodes creation 

builder.add_node("supervisor", supervisor_node)
builder.add_node("investigator", researcher_node)
builder.add_node("creator", creator_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)
builder.add_node("asr", asr_node)

# ========== Edges creation 

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
builder.add_edge("creator", "supervisor")
builder.add_edge("evaluator", "supervisor")
builder.add_edge("unifier", END)
builder.add_edge("asr", "supervisor")

# ========== Graph 

graph = builder.compile(checkpointer=sqlite_saver)

"""
config = {"configurable": {"thread_id": "1"}}

from PIL import Image

graph_image_path = "graph.png"
graph_image = graph.get_graph().draw_mermaid_png()
with open(graph_image_path, "wb") as f:
    f.write(graph_image)

# Updated test invocation with correct keys:
test = graph.invoke({
    "messages": [],
    "userQuestion": "What is ADD 3.0? Provide recommendations for its implementation under budget constraints.",
    "localQuestion": "",
    "hasVisitedInvestigator": False,
    "hasVisitedCreator": False,
    "hasVisitedEvaluator": False,
    "hasVisitedASR": False,
    "nextNode": "supervisor",
    "imagePath1": "",  # No image provided
    "imagePath2": "",
    "endMessage": ""
}, config)

print(test)
"""
