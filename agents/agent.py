class Agent:
    def __init__(self, name: str, instructions: str, tools=None):
        self.name = name
        self.instructions = instructions
        self.tools = tools if tools else []

    def __repr__(self):
        return f"<Agent name={self.name}>"
