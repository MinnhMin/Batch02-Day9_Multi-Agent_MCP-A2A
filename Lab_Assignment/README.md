# Lab Assignment Day 09: Improved Day 08 Agent using Supervisor-Workers

Thư mục này chứa mã nguồn cải tiến của RAG Agent ở Day 08 sang mô hình **Supervisor - Workers** sử dụng LangGraph.

## 🚀 Kiến Trúc Hệ Thống

Hệ thống bao gồm 1 Agent Điều phối (Supervisor) và 3 Agent Thành viên (Workers) chạy song song:

```
                      [User Query]
                           │
                           ▼
                  [Supervisor Router]
                           │
         ┌─────────────────┼─────────────────┐ (Định tuyến động song song)
         ▼                 ▼                 ▼
  [Legal Specialist]  [News Specialist]  [Penalty Calculator]
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                  [Synthesizer Node]
                           │
                           ▼
                    [Final Response]
```

### 1. Supervisor Node (Router)
- Tiếp nhận câu hỏi từ người dùng.
- Sử dụng LLM để phân tích ngữ cảnh câu hỏi và quyết định kích hoạt Worker nào phù hợp nhất (trả về danh sách tên workers dạng JSON).
- Sử dụng `Send()` API của LangGraph để kích hoạt các Worker chạy song song.

### 2. Specialized Workers (3 Workers)
- **Legal Specialist (`legal_worker`)**: Chuyên trách tra cứu các văn bản luật, nghị định chính thức từ Bộ luật Hình sự và Luật Phòng chống ma túy. Trích dẫn chính xác Điều, Khoản luật liên quan.
- **News & Case Specialist (`news_worker`)**: Chuyên trách tra cứu các bài viết tin tức thực tế (vụ việc Chi Dân, Andrea Aybar,...). Tổng hợp thông tin khách quan từ báo chí.
- **Penalty Calculator (`penalty_worker`)**: Chuyên trách so sánh, định khung hình phạt chi tiết (hình phạt tù, mức xử phạt hành chính tiền tệ) đối với các hành vi vi phạm.

### 3. Synthesizer / Aggregator Node
- Tiếp nhận toàn bộ báo cáo phân tích chuyên sâu từ các Worker đã kích hoạt.
- Sử dụng LLM để biên tập, tổng hợp thành một báo cáo cuối cùng thống nhất, mượt mà, loại bỏ trùng lặp và giữ nguyên các citation chính xác.
- Đính kèm phần nguồn tài liệu đã tham khảo ở cuối câu trả lời.

## 📦 Cách Chạy Thử Nghiệm

1. Đảm bảo bạn đang ở thư mục gốc của Day 09 (`Batch02-Day9_Multi-Agent_MCP-A2A`).
2. Kích hoạt môi trường và chạy file:
   ```bash
   uv run python Lab_Assignment/agent_supervisor.py
   ```
3. Hệ thống sẽ tự động thực hiện tìm kiếm, định tuyến động qua các agent và in ra kết quả tổng hợp có trích dẫn.
