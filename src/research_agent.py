# research_agent.py

from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class AIResearchAgent:
    def __init__(self, api_key):
        # Initialize Claude
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5",
            anthropic_api_key=api_key,
            temperature=0.7,
            max_tokens=4096
        )
        
        # Setup search tool (DuckDuckGo - free)
        search = DuckDuckGoSearchRun()
        
        # Define tools
        self.tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Search for current AI research, papers, and concepts. Use this to find the latest information on AI topics, recent developments, and technical details."
            )
        ]
        
        # Initialize agent with Claude
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def research_topic(self, topic):
        """Deep research on an AI topic"""
        
        prompt = f"""You are an expert AI researcher. Provide a comprehensive explanation of: {topic}

Structure your response as follows:

1. **Core Concept Definition**
   - Clear, technical definition
   - Key terminology

2. **Key Components/Architecture**
   - Main building blocks
   - How they work together

3. **Recent Developments (2024-2025)**
   - Latest research and breakthroughs
   - Use the Search tool to find current information

4. **Practical Applications**
   - Real-world use cases
   - Industry adoption

5. **Challenges & Future Directions**
   - Current limitations
   - Open research problems
   - Promising directions

Use the Search tool to find the latest information. Cite sources when possible.
"""
        
        try:
            result = self.agent.run(prompt)
            return result
        except Exception as e:
            # Fallback if agent fails
            return self._direct_research(topic)
    
    def _direct_research(self, topic):
        """Direct research without agent (fallback)"""
        prompt = f"""Provide a comprehensive explanation of this AI topic: {topic}

Include:
- Core concepts and definitions
- Key components and architecture
- Recent developments
- Practical applications
- Challenges and future directions

Be detailed and technical where appropriate."""

        response = self.llm.invoke(prompt)
        return response.content