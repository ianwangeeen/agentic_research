from agent_core import Agent
from dotenv import load_dotenv
import os
from tools.research_search_tool import ResearchSearchTool


# load_dotenv()
# agent = Agent(api_key=os.getenv("API_KEY"),
#             model="claude-sonnet-4-20250514",
#             max_iterations=5,)

# agent.register_tool(
#     name="add",
#     description="Add two integers",
#     func=add,
#     parameters={
#         "a": {"type": "integer", "required": True},
#         "b": {"type": "integer", "required": True},
#     },
# )

# result = agent.run("What is 12 + 30?", verbose=True)
# print("\nFINAL RESULT:", result)

def setup_research_agent(api_key: str) -> Agent:
    """
    Create an Agent with research search capabilities
    
    Args:
        api_key: Anthropic API key
        
    Returns:
        Configured Agent instance
    """
    
    # Initialize the agent
    agent = Agent(api_key=api_key, model="claude-sonnet-4-20250514", max_iterations=3)
    
    # Initialize search tool
    search_tool = ResearchSearchTool()
    
    # Register the search tool with the agent
    agent.register_tool(
        name="search_research_topic",
        description="""
        Search the internet for research papers, academic articles, and scientific content.
        
        Use this tool to:
        - Find recent papers on AI/ML topics
        - Discover state-of-the-art research
        - Locate academic sources and expert opinions
        - Get current information on research directions
        
        The tool returns formatted search results with titles, URLs, sources, and summaries.
        Always cite the sources in your final answer.
        """,
        func=search_tool.search,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research topic to search for (e.g., 'transformer attention mechanisms', 'RLHF techniques')"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                },
                "search_type": {
                    "type": "string",
                    "enum": ["general", "academic", "news"],
                    "description": "Type of search - 'academic' prioritizes papers, 'news' for recent developments, 'general' for broad coverage",
                    "default": "general"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["any", "day", "week", "month", "year"],
                    "description": "Filter by recency - 'year' for recent research, 'any' for all time",
                    "default": "any"
                }
            },
            "required": ["query"]
        }
    )
    
    print("✅ Research agent initialized with search capabilities!")
    print(f"   Model: {agent.model}")
    print(f"   Max iterations: {agent.max_iterations}")
    print(f"   Tools registered: {list(agent.tools.keys())}")
    
    return agent

if __name__ == '__main__':
    load_dotenv()
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    
    # Setup the research agent
    agent = setup_research_agent(
        api_key=anthropic_key,
    )
    
    print("\n" + "="*80)
    print("RESEARCH AGENT READY")
    print("="*80 + "\n")
    
    # Example 1: Simple research query
    print("Example 1: Basic Research Query\n")
    result = agent.run(
        query="Explain the latest developments in LLM agents and multi-agent systems. Include recent papers from 2024-2025.",
        verbose=True
    )
    
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(result)
    
    print("\n\n" + "="*80)
    print("\nExample 2: Comparative Analysis\n")
    result2 = agent.run(
        query="Compare RAG (Retrieval-Augmented Generation) vs fine-tuning approaches for LLMs. What are the trade-offs?",
        verbose=True
    )
    
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(result2)