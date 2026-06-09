"""Customer Agent server entry point — port 10100."""

from __future__ import annotations

import asyncio
import logging
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from common.registry_client import register
from customer_agent.agent_executor import CustomerAgentExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [customer_agent] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PORT = 10100
AGENT_ENDPOINT = f"http://localhost:{PORT}"


async def _register_with_retry(max_attempts: int = 10, delay: float = 2.0) -> None:
    """Retry registration until the registry is up."""
    info = {
        "agent_name": "customer-agent",
        "version": "1.0",
        "description": "Entry-point legal assistant; routes user questions to the Law Agent",
        "tasks": [],  # Customer Agent is an entry point, not discovered by other agents
        "endpoint": AGENT_ENDPOINT,
        "tags": ["customer", "entry-point", "legal-assistant"],
    }
    for attempt in range(1, max_attempts + 1):
        try:
            await register(info)
            logger.info("Registered with registry (attempt %d)", attempt)
            return
        except Exception as exc:
            logger.warning(
                "Registry not ready (attempt %d/%d): %s — retrying in %.0fs",
                attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)
    logger.error("Failed to register after %d attempts", max_attempts)


async def main() -> None:
    await _register_with_retry()

    agent_card = AgentCard(
        name="Customer Agent",
        description=(
            "Your legal assistant. Ask any legal question — I will route it through "
            "our network of specialist legal, tax, and compliance agents."
        ),
        url=AGENT_ENDPOINT,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="legal_assistant",
                name="Legal Assistant",
                description=(
                    "Answer legal questions by routing them to specialist agents "
                    "covering contract law, tax, and regulatory compliance."
                ),
                tags=["legal", "assistant", "multi-agent"],
            )
        ],
    )

    executor = CustomerAgentExecutor()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    app_builder = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = app_builder.build()

    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi import Body
    from uuid import uuid4
    from langchain_core.messages import HumanMessage, AIMessage
    from customer_agent.graph import build_graph

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        import os
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

    @app.post("/api/chat")
    async def chat_api(payload: dict = Body(...)):
        question = payload.get("question", "")
        trace_id = payload.get("trace_id", str(uuid4()))
        context_id = str(uuid4())
        
        # Build the Customer Agent graph
        graph = build_graph(trace_id=trace_id, context_id=context_id, depth=0)
        
        # Invoke the graph
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": context_id}}
        )
        
        # Extract the last AIMessage content
        answer = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                if not isinstance(msg, HumanMessage):
                    if isinstance(msg, AIMessage):
                        answer = msg.content
                        break
        if not answer:
            for msg in reversed(result.get("messages", [])):
                content = getattr(msg, "content", "")
                if content and not isinstance(msg, HumanMessage):
                    answer = content
                    break
        if not answer:
            answer = "I was unable to process your legal question at this time."
            
        # Get routing decisions for the response
        needs_tax = True
        needs_compliance = True
        try:
            import os
            import json
            trace_file = os.path.join("traces", f"{trace_id}.json")
            if os.path.exists(trace_file):
                with open(trace_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                needs_tax = data.get("needs_tax", True)
                needs_compliance = data.get("needs_compliance", True)
                os.remove(trace_file)
        except Exception:
            pass
            
        return JSONResponse(content={
            "response": answer, 
            "trace_id": trace_id,
            "needs_tax": needs_tax,
            "needs_compliance": needs_compliance
        })

    @app.get("/api/trace/{trace_id}")
    async def get_trace(trace_id: str):
        import os
        import json
        trace_file = os.path.join("traces", f"{trace_id}.json")
        if os.path.exists(trace_file):
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return JSONResponse(content={"status": "routing_decided", **data})
            except Exception as e:
                return JSONResponse(content={"status": "error", "message": str(e)})
        else:
            return JSONResponse(content={"status": "routing_undecided"})

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    logger.info("Customer Agent listening on port %d", PORT)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())