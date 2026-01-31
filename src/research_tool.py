import anthropic
import json
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import os
import requests


class ResearchSearchTool:
    """Research search tool - same as before"""
        
    def search_research_topic(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "general",
        time_range: str = "any"
    ) -> Dict[str, Any]:
        """Search the internet for research topics"""
        
        try:
            results = self._search_with_duckduckgo(query, num_results)
            
            enhanced_results = self._enhance_results(results, query, search_type)
            
            return {
                "success": True,
                "query": query,
                "search_type": search_type,
                "num_results": len(enhanced_results),
                "results": enhanced_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _search_with_duckduckgo(self, query, num_results):
        """DuckDuckGo search fallback"""
        try:
            from duckduckgo_search import DDGS
            
            ddgs = DDGS()
            results = []
            # enhanced_query = f"{query} research paper OR study OR article"
            enhanced_query = self._build_academic_query(query)
            
            for result in ddgs.text(enhanced_query, max_results=num_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "date": "",
                    "source": self._extract_domain(result.get("href", ""))
                })
            
            return results
        except:
            return []
    
    def _enhance_query_for_research(self, query, search_type):
        """Enhance query based on search type"""
        enhancements = {
            "academic": f'{query} (site:arxiv.org OR site:scholar.google.com OR "research paper")',
            "news": f'{query} (site:nature.com OR site:science.org OR "latest research")',
            "general": f'{query} research study article'
        }
        return enhancements.get(search_type, query)
    
    def _build_academic_query(self, topic: str) -> str:
        domains = [
            "site:arxiv.org",
            "site:openreview.net",
            "site:acm.org",
            "site:ieee.org",
            "site:medium.com",
            "site:scholar.google.com"
        ]

        keywords = [
            "paper",
            "survey",
            "benchmark",
            "evaluation",
        ]

        exclusions = [
            "-blog",
            "-tutorial",
            "-guide",
        ]

        return (
            f"\"{topic}\" "
            f"({' OR '.join(keywords)}) "
            f"({' OR '.join(domains)}) "
            f"(2024 OR 2025) "
            f"{' '.join(exclusions)}"
        )
    
    def _extract_domain(self, url):
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain[4:] if domain.startswith("www.") else domain
        except:
            return "unknown"
    
    def _enhance_results(self, results, query, search_type):
        """Add relevance scoring"""
        trusted_sources = {
            "arxiv.org": 0.95, "scholar.google.com": 0.95,
            "ieee.org": 0.9, "acm.org": 0.9,
            
        }
        
        enhanced = []
        for result in results:
            source = result.get("source", "")
            relevance_score = 0.5
            
            for trusted, score in trusted_sources.items():
                if trusted in source:
                    relevance_score = score
                    break
            
            query_terms = query.lower().split()
            title_lower = result.get("title", "").lower()
            matching_terms = sum(1 for term in query_terms if term in title_lower)
            relevance_score += (matching_terms / len(query_terms)) * 0.2
            relevance_score = min(relevance_score, 1.0)
            
            result["relevance_score"] = round(relevance_score, 2)
            result["is_academic"] = any(trusted in source for trusted in trusted_sources)
            enhanced.append(result)
        
        enhanced.sort(key=lambda x: x["relevance_score"], reverse=True)
        return enhanced


class Agent:
    """Core agent with ReAct-style reasoning loop"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", max_iterations: int = 10):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.tools: Dict[str, Dict[str, Any]] = {}
        
    def register_tool(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Dict[str, Any],
    ):
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,
        }
    
    def setup_research_tools(self, serper_api_key: Optional[str] = None):
        """
        Convenience method to set up all research-related tools
        
        Args:
            serper_api_key: Optional Serper API key for enhanced search
        """
        search_tool = ResearchSearchTool(serper_api_key=serper_api_key)
        
        # Register the search tool
        self.register_tool(
            name="search_research_topic",
            description="""
            Search the internet for research topics, academic papers, and scientific articles.
            
            This tool finds high-quality research content and can filter by:
            - Search type (general, academic, news)
            - Time range (recent vs. foundational papers)
            - Source credibility (prioritizes academic sources)
            
            Use when you need to:
            - Find recent papers on a topic
            - Discover state-of-the-art research
            - Locate academic sources and citations
            - Get research direction overview
            """,
            func=search_tool.search_research_topic,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research topic to search for"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (1-10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["general", "academic", "news"],
                        "description": "Type of search",
                        "default": "general"
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["any", "day", "week", "month", "year"],
                        "description": "Filter by time",
                        "default": "any"
                    }
                },
                "required": ["query"]
            }
        )
        
        print("✅ Research search tool registered!")
        return search_tool

    # ... (keep all your existing methods: planner, executor, critic, run, etc.)
    
    def _planner_prompt(self) -> str:
        planner_prompt = """
            You are a planning agent.

            Your job:
            - Understand the user's objective
            - Decompose it into concrete steps
            - Decide the SINGLE next step to execute

            Rules:
            - Do NOT call tools
            - Do NOT solve the task
            - Be concise and explicit

            Output format:

            <plan>
            objective: ...
            steps:
            - step 1
            - step 2
            next_step: ...
            </plan>
            """
        return planner_prompt
    
    def planner(self, query: str) -> Plan:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._planner_prompt(),
            messages=[{"role": "user", "content": query}],
        )

        text = response.content[0].text

        objective = re.search(r"objective:\s*(.*)", text).group(1)
        steps_block = re.search(r"steps:(.*?)next_step:", text, re.DOTALL).group(1)
        steps = [s.strip("- ").strip() for s in steps_block.splitlines() if s.strip()]
        next_step = re.search(r"next_step:\s*(.*)", text).group(1)

        return Plan(objective=objective, steps=steps, next_step=next_step)

    def _executor_prompt(self) -> str:
        tools_desc = "\n\n".join(
            f"Tool: {name}\nDescription: {info['description']}\nParameters:\n{json.dumps(info['parameters'], indent=2)}"
            for name, info in self.tools.items()
        )
        prompt = f"""
            You are an execution agent.

            You are given ONE task step.
            You may either:
            - Call ONE tool
            - OR return a reasoning-only result

            Rules:
            - Never decide completion
            - Never invent tools
            - Use valid JSON if calling a tool

            Available tools:
            {tools_desc}

            Output format (choose ONE):

            <action>tool_name</action>
            <action_input>{{ JSON }}</action_input>

            OR

            <result>
            text
            </result>
            """
        return prompt
    
    def executor(self, step: str) -> ExecutionResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self._executor_prompt(),
            messages=[{"role": "user", "content": step}],
        )

        text = response.content[0].text

        action_match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
        input_match = re.search(r"<action_input>(.*?)</action_input>", text, re.DOTALL)
        result_match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)

        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = json.loads(input_match.group(1).strip())

            observation = self._execute_tool(tool_name, tool_input)
            print(f"TOOL EXECUTED: {tool_name}({tool_input})")
            return ExecutionResult(
                action=tool_name,
                action_input=tool_input,
                observation=observation,
            )

        if result_match:
            return ExecutionResult(
                action=None,
                action_input=None,
                observation=result_match.group(1).strip(),
            )

        raise RuntimeError("Invalid executor output")

    def _execute_tool(self, name: str, params: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"Error: Tool '{name}' not registered."

        try:
            result = str(self.tools[name]["function"](**params))
            return f"[Tool: {name}] {result}"
        except Exception as e:
            return f"Tool execution error: {e}"

    def _critic_prompt(self) -> str:
        return """
            You are a critic agent.

            Your role:
            - Objectively evaluate progress
            - Decide if the task is complete
            - If not complete, suggest the next step

            Rules:
            - Be strict
            - Prefer correctness over optimism

            Output format:

            <critique>
            complete: true/false
            reason: ...
            next_step: ...
            </critique>
            """
    
    def critic(self, objective: str, observation: str) -> Critique:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._critic_prompt(),
            messages=[
                {"role": "user", "content": f"Objective: {objective}\n\nObservation:\n{observation}"}
            ],
        )

        text = response.content[0].text

        complete = "true" in re.search(r"complete:\s*(.*)", text).group(1).lower()
        reason = re.search(r"reason:\s*(.*)", text).group(1)
        next_step_match = re.search(r"next_step:\s*(.*)", text)

        return Critique(
            complete=complete,
            reason=reason,
            next_step=next_step_match.group(1) if next_step_match else None,
        )

    def run(self, query: str, verbose: bool = True) -> str:
        plan = self.planner(query)

        if verbose:
            print("\n>>> PLAN")
            print(plan)

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}")
                print(f"{'='*60}")
                print(f"▶ Executing: {plan.next_step}")

            execution = self.executor(plan.next_step)

            if verbose:
                print("\n>>> OBSERVATION")
                print(execution.observation[:500])

            critique = self.critic(plan.objective, execution.observation)

            if verbose:
                print("\n>>> CRITIQUE")
                print(critique)

            if critique.complete:
                if verbose:
                    print("\n✅ TASK COMPLETE")
                return execution.observation

            if not critique.next_step:
                raise RuntimeError("Critic did not provide next step")

            plan.next_step = critique.next_step

        raise RuntimeError("Maximum iterations reached without completion")