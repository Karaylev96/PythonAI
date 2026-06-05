from langchain_core.tools import Tool
from services import vector_manager
 
def get_document_context(query: str) -> str:
    result = vector_manager.search(query, k=3)
    if not result:
        return "No info for documents"
    return "\n\n".join([doc.page_content for doc, score in results])

doc_search_tool = Tool(
    name="DocumentSearch",
    func=get_document_context,
    description="Use this tool to answer document questions"
)

tools = [doc_search_tool]
