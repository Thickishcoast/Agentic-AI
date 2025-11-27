# src/agent_llm_policy.py
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI


COLLECTION_NAME = "web_agent_experiences"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt: str) -> str:
    """
    Call GPT-5-nano with the given prompt and return the raw text output.
    The prompt already includes instructions to output exactly ONE action line.
    """
    resp = client.responses.create(
        model="gpt-5-nano",  
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=64,
        temperature=0.2,     
    )
    return (resp.output_text or "").strip()


model = SentenceTransformer(MODEL_NAME)
client_qdrant = QdrantClient(host="localhost", port=6333)


def retrieve_memory(obs_text: str, k: int = 3):
    vec = model.encode(obs_text).tolist()
    res = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    return res.points


def build_prompt_with_memory(task: str, obs_text: str, neighbors) -> str:
    memory_strs = []
    for n in neighbors:
        payload = n.payload or {}
        reward = payload.get("reward", 0)
        past_obs = payload.get("obs_text", "")[:300]
        past_action = payload.get("action_str", "")
        memory_strs.append(
            f"- Past step (reward={reward}):\n"
            f"  Observation: {past_obs}\n"
            f"  Action taken: {past_action}"
        )

    memory_block = "\n".join(memory_strs) if memory_strs else "None"

    return f"""
You are a web-browsing agent. You control the browser by issuing ONE command.
You are helping a user with the task:

"{task}"

You can ONLY use these actions:

- CLICK [NODE_ID]
- TYPE [NODE_ID] | text=...
- NONE

The current page is represented as a list of nodes like "[ID] ROLE 'Text'".

Current page:
{obs_text}

Relevant past experiences from memory:
{memory_block}

Choose ONE best next action to move closer to the task.
Respond with exactly ONE line:
- "CLICK [ID]" or
- "TYPE [ID] | text=..." or
- "NONE"
Do not add explanations.
""".strip()


def llm_policy(task: str, obs_text: str) -> str:
    neighbors = retrieve_memory(obs_text, k=3)
    prompt = build_prompt_with_memory(task, obs_text, neighbors)
    raw = call_llm(prompt)
    return raw


def extract_text_from_obs(obs) -> str:
    """
    Turn WebArena observation into plain text for the LLM.
    """
    if isinstance(obs, dict):
        text_part = obs.get("text")

        if isinstance(text_part, str):
            return text_part

        if isinstance(text_part, dict):
            nodes = text_part.get("obs_nodes_info", {})
            lines = []
            for node_id, node in nodes.items():
                lines.append(node.get("text", ""))
            return "\n".join(lines)

    return str(obs)
