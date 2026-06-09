"""
Day 09 - Lab Assignment: Improved Day 08 Agent using Supervisor-Workers pattern.

Mô hình Supervisor - Workers:
- Supervisor: Phân tích câu hỏi của người dùng và định tuyến động sang các Worker tương ứng.
- Workers (3 workers):
  1. Legal Specialist: Tra cứu các văn bản luật, điều khoản chính thức (Bộ luật Hình sự, Luật Phòng chống ma túy).
  2. News & Case Specialist: Tra cứu các bài viết tin tức thực tế (vụ việc người nổi tiếng như Chi Dân, Andrea Aybar, Miu Lê).
  3. Penalty Calculator: Phân tích và định khung hình phạt chi tiết, so sánh án phạt và mức tiền phạt.
- Synthesizer (Aggregator): Tổng hợp các báo cáo từ các chuyên gia thành câu trả lời hoàn chỉnh có citation.
"""

import sys
import os
import json
import asyncio
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

# Thêm đường dẫn tới Day 08 để tái sử dụng mã nguồn RAG
sys.path.insert(0, r"d:\VinAI\Day08_RAG_pipeline_cohort2")
sys.path.insert(0, r"d:\VinAI\Day08_RAG_pipeline_cohort2\src")

# Thêm thư mục gốc Day 09 vào sys.path để import common.llm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from src.task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from common.llm import get_llm

# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------

def _accumulate_reports(left: list, right: list) -> list:
    """Reducer: gộp danh sách báo cáo từ các worker."""
    return left + right

class SupervisorState(TypedDict):
    query: str
    selected_workers: List[str]
    worker_reports: Annotated[List[dict], _accumulate_reports]
    final_response: str

# ---------------------------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------------------------

def is_legal_doc(metadata: dict) -> bool:
    source = metadata.get("source", "").lower()
    return "luat" in source or "bo-luat" in source

def is_news_doc(metadata: dict) -> bool:
    source = metadata.get("source", "").lower()
    return "article" in source

# ---------------------------------------------------------------------------
# 3. Worker Nodes
# ---------------------------------------------------------------------------

# --- Worker 1: Legal Specialist ---
async def legal_worker(state: SupervisorState) -> dict:
    print("🤖 [Worker: Legal Specialist] Đang truy vấn văn bản luật...")
    query = state["query"]
    
    # Retrieve documents
    chunks = retrieve(query, top_k=6)
    legal_chunks = [c for c in chunks if is_legal_doc(c.get("metadata", {}))]
    
    if not legal_chunks:
        legal_chunks = chunks  # Fallback
        
    context_str = "\n\n".join(
        f"[Nguồn: {c.get('metadata', {}).get('source', 'Tài liệu')}]\n{c['content']}" 
        for c in legal_chunks
    )
    
    prompt = f"""Bạn là Chuyên gia Pháp lý chuyên trách về Luật Hình sự và Phòng chống ma túy Việt Nam.
Hãy phân tích câu hỏi sau dựa trên các văn bản luật trong ngữ cảnh:

Câu hỏi: {query}

Ngữ cảnh tài liệu pháp lý:
{context_str}

Nhiệm vụ:
1. Trích dẫn chính xác các Điều luật, Khoản luật liên quan (ví dụ: Điều 249 Bộ luật Hình sự).
2. Giải thích ý nghĩa pháp lý và trách nhiệm hình sự/dân sự/hành chính.
3. Chỉ đưa ra các thông tin có bằng chứng rõ ràng trong ngữ cảnh. Nếu không có, hãy ghi rõ 'Không có tài liệu pháp lý phù hợp'.

Trả lời bằng tiếng Việt, súc tích và có cấu trúc rõ ràng."""

    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    return {
        "worker_reports": [{
            "worker": "Legal Specialist",
            "report": response.content,
            "sources": [c.get('metadata', {}).get('source', 'Tài liệu') for c in legal_chunks]
        }]
    }

# --- Worker 2: News & Case Specialist ---
async def news_worker(state: SupervisorState) -> dict:
    print("🤖 [Worker: News & Case Specialist] Đang truy vấn tin tức báo chí...")
    query = state["query"]
    
    # Retrieve documents
    chunks = retrieve(query, top_k=6)
    news_chunks = [c for c in chunks if is_news_doc(c.get("metadata", {}))]
    
    if not news_chunks:
        news_chunks = chunks  # Fallback
        
    context_str = "\n\n".join(
        f"[Nguồn: {c.get('metadata', {}).get('source', 'Báo chí')}]\n{c['content']}" 
        for c in news_chunks
    )
    
    prompt = f"""Bạn là Chuyên gia Phân tích Tin tức và Vụ việc.
Hãy tổng hợp và phân tích các vụ việc thực tế liên quan đến câu hỏi dựa trên các bài báo trong ngữ cảnh:

Câu hỏi: {query}

Ngữ cảnh tin tức báo chí:
{context_str}

Nhiệm vụ:
1. Tổng hợp các thông tin thực tế về đối tượng, thời gian, hành vi (ví dụ: ca sĩ Chi Dân, người mẫu Andrea Aybar,...).
2. Nêu rõ nguồn tin tức trích dẫn (ví dụ: [article_01.md]).
3. Không phỏng đoán. Nếu ngữ cảnh không có thông tin về vụ việc thực tế nào, hãy ghi rõ 'Không tìm thấy tin tức thực tế liên quan'.

Trả lời bằng tiếng Việt, khách quan và súc tích."""

    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    return {
        "worker_reports": [{
            "worker": "News & Case Specialist",
            "report": response.content,
            "sources": [c.get('metadata', {}).get('source', 'Báo chí') for c in news_chunks]
        }]
    }

# --- Worker 3: Penalty Calculator ---
async def penalty_worker(state: SupervisorState) -> dict:
    print("🤖 [Worker: Penalty Calculator] Đang định khung hình phạt và tính toán...")
    query = state["query"]
    
    # Retrieve documents
    chunks = retrieve(query, top_k=6)
    
    context_str = "\n\n".join(
        f"[Nguồn: {c.get('metadata', {}).get('source', 'Tài liệu')}]\n{c['content']}" 
        for c in chunks
    )
    
    prompt = f"""Bạn là Chuyên gia Định khung Hình phạt và Tính toán Xử phạt.
Hãy phân tích và so sánh các khung hình phạt, số năm tù, mức phạt tiền dựa trên câu hỏi và ngữ cảnh:

Câu hỏi: {query}

Ngữ cảnh hỗ trợ:
{context_str}

Nhiệm vụ:
1. Xác định mức phạt tiền hoặc khung hình phạt tù tối thiểu và tối đa cho hành vi được mô tả.
2. So sánh mức độ nghiêm trọng giữa các khung hình phạt nếu câu hỏi yêu cầu so sánh.
3. Trích dẫn rõ nguồn căn cứ tính toán (ví dụ: Điều 249 BLHS).

Trả lời bằng tiếng Việt, dưới dạng bảng hoặc danh sách gạch đầu dòng rõ ràng."""

    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    return {
        "worker_reports": [{
            "worker": "Penalty Calculator",
            "report": response.content,
            "sources": [c.get('metadata', {}).get('source', 'Tài liệu') for c in chunks]
        }]
    }

# ---------------------------------------------------------------------------
# 4. Supervisor Routing Node
# ---------------------------------------------------------------------------

async def supervisor_router(state: SupervisorState) -> dict:
    print("👑 [Supervisor] Đang phân tích câu hỏi để định tuyến tới Worker phù hợp...")
    query = state["query"]
    
    prompt = f"""Bạn là Điều phối viên hệ thống (Supervisor). Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và quyết định kích hoạt các chuyên gia (Workers) nào phù hợp nhất.
Các chuyên gia hiện có:
1. "legal_worker": Chuyên gia về văn bản luật, điều khoản, nghị định chính thức.
2. "news_worker": Chuyên gia về tin tức thực tế, các vụ việc bắt giữ, người nổi tiếng.
3. "penalty_worker": Chuyên gia về tính toán, định khung hình phạt, mức phạt tiền và so sánh án phạt.

Hãy phân tích câu hỏi: "{query}"

Trả về kết quả dưới dạng JSON duy nhất, không có markdown hay văn bản thừa:
{{
  "workers": ["tên_worker_1", "tên_worker_2", ...]
}}

Quy tắc:
- Chọn "legal_worker" nếu câu hỏi liên quan đến luật pháp, định nghĩa, điều khoản chung.
- Chọn "news_worker" nếu câu hỏi đề cập đến nghệ sĩ, tin tức cụ thể hoặc thực tế.
- Chọn "penalty_worker" nếu câu hỏi hỏi về số năm tù, mức phạt tiền, hoặc so sánh án phạt.
- Nếu câu hỏi kết hợp nhiều khía cạnh hoặc mang tính bao quát, hãy chọn cả 3.
"""
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
        
    try:
        data = json.loads(raw)
        selected = data.get("workers", [])
    except Exception:
        selected = ["legal_worker", "news_worker", "penalty_worker"]
        
    valid_workers = []
    for w in selected:
        w_clean = w.strip().lower()
        if "legal" in w_clean:
            valid_workers.append("legal_worker")
        elif "news" in w_clean:
            valid_workers.append("news_worker")
        elif "penalty" in w_clean or "calc" in w_clean:
            valid_workers.append("penalty_worker")
            
    if not valid_workers:
        valid_workers = ["legal_worker", "news_worker", "penalty_worker"]
        
    print(f"👑 [Supervisor] Kích hoạt các worker: {valid_workers}")
    return {"selected_workers": valid_workers}

def route_to_workers(state: SupervisorState) -> list[Send]:
    sends = []
    for worker in state["selected_workers"]:
        sends.append(Send(worker, state))
    return sends

# ---------------------------------------------------------------------------
# 5. Synthesizer / Aggregator Node
# ---------------------------------------------------------------------------

async def aggregate_results(state: SupervisorState) -> dict:
    print("👑 [Supervisor] Đang tổng hợp báo cáo từ các worker...")
    query = state["query"]
    reports = state["worker_reports"]
    
    report_contents = []
    all_sources = set()
    
    for r in reports:
        report_contents.append(f"### 📋 Báo Cáo Chuyên Môn - {r['worker']}:\n{r['report']}")
        for src in r["sources"]:
            all_sources.add(src)
            
    combined_reports = "\n\n".join(report_contents)
    sources_str = ", ".join(sorted(list(all_sources)))
    
    prompt = f"""Bạn là Luật sư Trưởng Tổng hợp. Hãy tổng hợp các báo cáo từ các chuyên gia thành một phản hồi RAG hoàn chỉnh, mạch lạc và có cấu trúc tốt cho người dùng.

Câu hỏi của người dùng: {query}

Báo cáo chi tiết từ các chuyên gia:
{combined_reports}

Yêu cầu:
1. Tổng hợp thông tin một cách mạch lạc, loại bỏ trùng lặp, chia thành các phần rõ ràng.
2. Giữ nguyên các trích dẫn tài liệu pháp lý (ví dụ: [Điều 249 BLHS]) và trích dẫn báo chí (ví dụ: [article_01.md]).
3. Trình bày câu trả lời rõ ràng bằng tiếng Việt.
4. Ở cuối cùng, thêm một phần riêng ghi rõ danh sách các nguồn tài liệu đã tham khảo (ví dụ: Nguồn tham khảo: {sources_str}).
5. Nếu các chuyên gia đều báo không có thông tin, hãy trả lời lịch sự rằng cơ sở dữ liệu hiện tại không đủ thông tin để trả lời câu hỏi này.
"""
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    return {"final_response": response.content}

# ---------------------------------------------------------------------------
# 6. Graph Construction
# ---------------------------------------------------------------------------

def build_supervisor_graph():
    graph = StateGraph(SupervisorState)
    
    # Add nodes
    graph.add_node("supervisor_router", supervisor_router)
    graph.add_node("legal_worker", legal_worker)
    graph.add_node("news_worker", news_worker)
    graph.add_node("penalty_worker", penalty_worker)
    graph.add_node("aggregate_results", aggregate_results)
    
    # Define workflow
    graph.add_edge(START, "supervisor_router")
    
    # Conditional routing to workers in parallel
    graph.add_conditional_edges(
        "supervisor_router",
        route_to_workers,
        ["legal_worker", "news_worker", "penalty_worker"]
    )
    
    # Connect workers to aggregate node
    graph.add_edge("legal_worker", "aggregate_results")
    graph.add_edge("news_worker", "aggregate_results")
    graph.add_edge("penalty_worker", "aggregate_results")
    
    # Finish flow
    graph.add_edge("aggregate_results", END)
    
    return graph.compile()

# ---------------------------------------------------------------------------
# 7. Main execution
# ---------------------------------------------------------------------------

async def main():
    load_dotenv()
    
    # Đảm bảo in tiếng Việt trên console Windows không lỗi
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    app = build_supervisor_graph()
    
    # Test query
    q = "Ca sĩ Chi Dân phạm tội gì và khung hình phạt cụ thể là như thế nào?"
    print("\n" + "="*80)
    print(f"CÂU HỎI: {q}")
    print("="*80)
    
    result = await app.ainvoke({
        "query": q,
        "selected_workers": [],
        "worker_reports": [],
        "final_response": ""
    })
    
    print("\n" + "="*80)
    print("KẾT QUẢ TỔNG HỢP CUỐI CÙNG:")
    print("="*80)
    print(result["final_response"])
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
