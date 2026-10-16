import langchain_classic.hub as hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from answerQuestionAgentTool import tools
# from AiApiKey import get_api_key
import os
from dotenv import load_dotenv

def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return api_key

class AgentManager:
    def __init__(self):
        self.key = get_api_key()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=self.key)
        self.promt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, tools, self.promt)
        self.agent_executor = AgentExecutor(agent = self.agent, tools = tools, verbose=True, handle_parsing_errors=True)

    async def run(self, user_input: str):
        response = await self.agent_executor.ainvoke({"input": user_input})
        return response["output"]

ai_agent = AgentManager()