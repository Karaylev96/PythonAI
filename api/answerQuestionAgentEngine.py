import langchainhub as hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from answerQuestionAgentTool import tools

class AgentManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.promt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, tools, self.promt)
        self.agent_executor = AgentExecutor(agent = self.agent, tools = tools, verbose=True, handle_parsing_errors=True)

    async def run(self, user_input: str):
        response = self.agent_executor.invoke({"input":user_input})
        return response["output"]

ai_agent = AgentManager()