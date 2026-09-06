import os
import logfire

from dotenv import load_dotenv

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional

from App.guardrails.rails import (
    initialize_rails,
    guard,
)

from App.agents.graph import rag_agent


load_dotenv()

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN")
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Enterprise Mental Health Companion"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():

    initialize_rails()


# ============================================================
# Request model
# ============================================================

class QueryRequest(BaseModel):

    q: str

    thread_id: Optional[str] = "default_user"


# ============================================================
# Home
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Mental Health Wellbeing Companion API is Live."
    }


# ============================================================
# Graph image
# ============================================================

@app.get("/graph")
def get_graph_image():

    try:

        png_bytes = (
            rag_agent
            .get_graph()
            .draw_mermaid_png()
        )

        return Response(
            content=png_bytes,
            media_type="image/png"
        )

    except Exception as e:

        return {
            "error": f"Could not generate graph image: {e}"
        }


# ============================================================
# Query
# ============================================================

@app.post("/query")
async def query(request: QueryRequest):

    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": q
            }
        ],

        "current_query": q,

        "documents": [],

        "plan": [
            "Start"
        ],

        "status": "Initializing Graph..."
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # ====================================================
        # Gate 1: Guardrails
        # ====================================================

        (
            rail_fired,
            rail_response,
            guardrail_intent
        ) = await guard(q)

        # ----------------------------------------------------
        # Guardrail blocked request
        # ----------------------------------------------------

        if rail_fired:

            logfire.info(
                f"Request blocked by guardrails | "
                f"intent={guardrail_intent} | "
                f"thread={thread_id}"
            )

            return {

                "question": q,

                "answer": rail_response,

                "actual_guardrail_intent": guardrail_intent,

                "actual_planner_intent": None,

                "thought_process": [
                    f"Guardrail Intent: {guardrail_intent}",
                    "Planner: Skipped",
                    "Retrieval: Skipped"
                ],

                "status": "Blocked by guardrails.",

                "sources": []
            }

        # ====================================================
        # Gate 2: LangGraph
        # ====================================================

        final_output = rag_agent.invoke(
            initial_state,
            config=config
        )

        # Planner intent should be set by planner_node
        planner_intent = final_output.get(
            "planner_intent"
        )

        return {

            "question": q,

            "answer": (
                final_output.get("final_answer")
                or ""
            ),

            "actual_guardrail_intent": (
                guardrail_intent
                if guardrail_intent != "unknown"
                else "safe"
            ),

            "actual_planner_intent": planner_intent,

            "thought_process": (
                final_output.get("plan")
                or []
            ),

            "status": final_output.get("status"),

            "sources": (
                final_output.get("documents")
                or []
            )
        }

    except Exception as e:

        logfire.error(
            f"Backend Execution Failed: {e}"
        )

        return {

            "question": q,

            "answer": (
                "I apologize, but I encountered an internal "
                "error while processing your request."
            ),

            "actual_guardrail_intent": "unknown",

            "actual_planner_intent": "unknown",

            "thought_process": [
                "Error encountered during execution."
            ],

            "status": "error",

            "sources": []
        }