class WebSearchTool:
    def __init__(self):
        self.name = "WebSearchTool"

    def search(self, query: str) -> str:
        # 模拟搜索行为
        return f"Search results for '{query}'"

    def __repr__(self):
        return f"<WebSearchTool>"
