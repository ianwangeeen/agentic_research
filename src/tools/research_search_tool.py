import anthropic
import json
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import requests
from datetime import datetime
from ddgs import DDGS
from agent_core import Agent  

# ============================================================================
# RESEARCH SEARCH TOOL (Simplified for your Agent)
# ============================================================================

class ResearchSearchTool:
    """
    Simplified search tool optimized for your Agent's executor
    """

    def search(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "general",
        time_range: str = "any"
    ) -> str:
        """
        Search for research topics - returns formatted string for Agent consumption
        
        Returns:
            Formatted string with search results (not dict, for easier Agent parsing)
        """
        
        try:
            results = self._search_with_duckduckgo(query, num_results)
            
            # Format as readable text for the Agent
            return self._format_results(results, query)
            
        except Exception as e:
            return f"Search failed: {str(e)}\nPlease try rephrasing the query or check your internet connection."
    
    def _build_academic_query(self, topic: str) -> str:
        domains = [
            "site:arxiv.org",
            "site:openreview.net",
            "site:acm.org",
            "site:ieee.org",
            "site:medium.com"
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

    def _search_with_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Fallback: DuckDuckGo search"""
        
        try:
            ddgs = DDGS()
            results = []
            
            enhanced_query = self._build_academic_query(query)
            
            for result in ddgs.text(enhanced_query, max_results=num_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": self._extract_domain(result.get("href", ""))
                })
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain[4:] if domain.startswith("www.") else domain
        except:
            return "unknown"
    
    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Format results as readable text for Agent consumption
        """
        
        if not results:
            return f"No results found for query: '{query}'\n\nTry:\n- Rephrasing the query\n- Using more specific keywords\n- Broadening the search terms"
        
        output = f"Search Results for: '{query}'\n"
        output += f"Found {len(results)} relevant sources\n"
        output += "=" * 80 + "\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"[{i}] {result['title']}\n"
            output += f"    Source: {result['source']}\n"
            output += f"    URL: {result['url']}\n"
            output += f"    Summary: {result['snippet'][:200]}...\n"
            output += "\n"
        
        output += "=" * 80 + "\n"
        output += "Use these sources to answer the research question.\n"
        
        return output