"""Bài Tập Nâng Cao: Tích hợp Financial Agent, Memory, Custom Tool, và Retry Logic.

Chạy file này bằng lệnh:
py -3.12 -m uv run python exercises/exercise_advanced.py
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from tenacity import retry, stop_after_attempt, wait_exponential

from common.llm import get_llm

# ===========================================================================
# 1. State Definition (Challenge 1: Financial Analysis field added)
# ===========================================================================

def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: giá trị mới ghi đè giá trị cũ."""
    return right if right is not None else (left or "")


class AdvancedState(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    financial_analysis: Annotated[str, _last_wins]  # Challenge 1 field
    final_response: str


# ===========================================================================
# 2. Challenge 3 & 4: Custom API Tool with Retry Logic (tenacity)
# ===========================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=True
)
async def fetch_books_count_api(query: str) -> int:
    """Gọi API công khai để tra cứu tài liệu (ví dụ đại diện cho API luật pháp)."""
    # Sử dụng Open Library API làm demo
    url = f"https://openlibrary.org/search.json?q={query}"
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("numFound", 0)


@tool
async def search_online_resources(keyword: str) -> str:
    """Tra cứu cơ sở dữ liệu thư viện trực tuyến để đếm số lượng tài liệu liên quan đến từ khóa.

    Args:
        keyword: Từ khóa tìm kiếm (ví dụ: 'contract', 'finance', 'privacy').
    """
    try:
        count = await fetch_books_count_api(keyword)
        return f"Tìm thấy {count} tài liệu học thuật trực tuyến liên quan đến từ khóa '{keyword}'."
    except Exception as exc:
        return f"[Lỗi tra cứu tài liệu trực tuyến: {exc}]"


# ===========================================================================
# 3. Agent Nodes (Challenge 4: safe LLM call retry implemented)
# ===========================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=True
)
async def safe_llm_invoke(prompt_text: str) -> str:
    """Gọi LLM an toàn tích hợp cơ chế tự động thử lại nếu lỗi mạng/API."""
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt_text)])
    return response.content


async def law_agent(state: AdvancedState) -> dict:
    """Lead Attorney Agent."""
    prompt = f"""Bạn là luật sư trưởng điều phối. Phân tích khía cạnh hợp đồng thương mại cho câu hỏi sau:
    
Câu hỏi: {state['question']}"""
    analysis = await safe_llm_invoke(prompt)
    return {"law_analysis": analysis}


async def tax_agent(state: AdvancedState) -> dict:
    """Tax Specialist Agent."""
    prompt = f"""Bạn là chuyên gia thuế. Phân tích khía cạnh thuế và nghĩa vụ tài chính doanh nghiệp:
    
Câu hỏi: {state['question']}
Bản án sơ bộ: {state.get('law_analysis', '')}"""
    analysis = await safe_llm_invoke(prompt)
    return {"tax_analysis": analysis}


async def compliance_agent(state: AdvancedState) -> dict:
    """Compliance Specialist Agent."""
    prompt = f"""Bạn là chuyên gia tuân thủ quy định. Phân tích các vi phạm về SEC, SOX, quản trị doanh nghiệp:

Câu hỏi: {state['question']}
Bản án sơ bộ: {state.get('law_analysis', '')}"""
    analysis = await safe_llm_invoke(prompt)
    return {"compliance_analysis": analysis}


async def privacy_agent(state: AdvancedState) -> dict:
    """Privacy Specialist Agent."""
    prompt = f"""Bạn là chuyên gia bảo mật thông tin và GDPR. Phân tích khía cạnh lộ lọt dữ liệu và bồi thường quyền riêng tư:

Câu hỏi: {state['question']}
Bản án sơ bộ: {state.get('law_analysis', '')}"""
    analysis = await safe_llm_invoke(prompt)
    return {"privacy_analysis": analysis}


# Challenge 1: Financial Agent Node
async def financial_agent(state: AdvancedState) -> dict:
    """Financial Damage Analysis Agent."""
    prompt = f"""Bạn là chuyên gia thẩm định tài chính doanh nghiệp. Hãy tính toán/đánh giá các tổn thất tài chính tiềm ẩn,
tiền phạt phạt hợp đồng, chi phí bồi thường và tác động lên dòng tiền của doanh nghiệp:

Câu hỏi: {state['question']}
Bản án sơ bộ: {state.get('law_analysis', '')}"""
    analysis = await safe_llm_invoke(prompt)
    return {"financial_analysis": analysis}


# ===========================================================================
# 4. Routing Logic
# ===========================================================================

def check_routing(state: AdvancedState) -> list[Send]:
    """Quyết định gọi các chuyên gia nào chạy song song."""
    question_lower = state["question"].lower()
    tasks = []

    # Định tuyến sang Tax Agent
    if any(kw in question_lower for kw in ["tax", "irs", "thuế", "avoid"]):
        tasks.append(Send("tax_agent", state))

    # Định tuyến sang Compliance Agent
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation", "tuân thủ"]):
        tasks.append(Send("compliance_agent", state))

    # Định tuyến sang Privacy Agent
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "bảo mật", "rò rỉ"]):
        tasks.append(Send("privacy_agent", state))

    # Challenge 1: Định tuyến sang Financial Agent
    if any(kw in question_lower for kw in ["financial", "damage", "thiệt hại", "tài chính", "tiền", "bồi thường"]):
        tasks.append(Send("financial_agent", state))

    return tasks if tasks else [Send("aggregate_results", state)]


# ===========================================================================
# 5. Aggregator
# ===========================================================================

async def aggregate_results(state: AdvancedState) -> dict:
    """Tổng hợp phân tích từ các chuyên gia thành báo cáo hoàn chỉnh."""
    sections = []
    if state.get("law_analysis"):
        sections.append(f"📋 PHÂN TÍCH PHÁP LÝ:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"💰 PHÂN TÍCH THUẾ:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"✅ PHÂN TÍCH TUÂN THỦ:\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"🔒 PHÂN TÍCH BẢO MẬT DỮ LIỆU & GDPR:\n{state['privacy_analysis']}")
    if state.get("financial_analysis"):
        sections.append(f"📊 PHÂN TÍCH THIỆT HẠI TÀI CHÍNH:\n{state['financial_analysis']}")

    combined = "\n\n".join(sections)
    prompt = f"""Hãy tổng hợp và cấu trúc lại các báo cáo chuyên gia sau thành một bản tóm tắt pháp lý mạch lạc và rõ ràng:

{combined}

Câu hỏi gốc: {state['question']}"""
    
    final_resp = await safe_llm_invoke(prompt)
    return {"final_response": final_resp}


# ===========================================================================
# 6. Graph Compilation (Challenge 2: MemorySaver checkpointer integrated)
# ===========================================================================

def build_advanced_graph():
    graph = StateGraph(AdvancedState)

    # Đăng ký các Nodes
    graph.add_node("law_agent", law_agent)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("financial_agent", financial_agent)  # Challenge 1
    graph.add_node("aggregate_results", aggregate_results)

    # Thiết lập luồng chạy
    graph.add_edge(START, "law_agent")
    graph.add_conditional_edges("law_agent", check_routing)
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("financial_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    # Challenge 2: Tích hợp Memory Checkpointer để ghi nhớ ngữ cảnh hội thoại
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ===========================================================================
# 7. Main Execution Flow
# ===========================================================================

async def main():
    load_dotenv()

    # Khởi tạo đồ thị
    app = build_advanced_graph()

    # Cấu hình phiên chat để lưu trữ bộ nhớ
    config = {"configurable": {"thread_id": "session_advanced_1"}}

    # Lượt chat 1: Câu hỏi chứa từ khóa kích hoạt Financial Agent, Privacy Agent và Law Agent
    question_1 = "Nếu công ty bị rò rỉ dữ liệu người dùng dẫn đến kiện tụng và phạt tiền, hậu quả tài chính và bảo mật là gì?"
    print("=" * 75)
    # Ghi đè mã hóa output UTF-8 để in ra terminal Việt không bị lỗi
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    print(f"LƯỢT CHAT 1: {question_1}")
    print("=" * 75)
    print("Đang xử lý qua các Agent chuyên biệt...\n")

    result_1 = await app.ainvoke(
        {
            "question": question_1,
            "law_analysis": "",
            "tax_analysis": "",
            "compliance_analysis": "",
            "privacy_analysis": "",
            "financial_analysis": "",
            "final_response": "",
        },
        config=config
    )

    print(result_1["final_response"])
    print("\n" + "=" * 75)

    # Lượt chat 2: Kiểm thử tính năng ghi nhớ ngữ cảnh (Challenge 2)
    # Lấy thông tin từ lượt chat trước mà không cần truyền lại toàn bộ dữ liệu
    question_2 = "Dựa trên vụ việc rò rỉ dữ liệu vừa nói, hãy cho biết công ty nên ưu tiên sửa đổi quy trình bảo mật nào đầu tiên?"
    print(f"\nLƯỢT CHAT 2 (Sử dụng Memory): {question_2}")
    print("=" * 75)
    
    # Ở lượt chat 2, Graph tự động nhớ trạng thái từ result_1 nhờ cùng thread_id
    result_2 = await app.ainvoke(
        {
            "question": question_2,
        },
        config=config
    )
    
    print(result_2["final_response"])
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
