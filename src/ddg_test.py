from pprint import pprint

def build_academic_query(topic: str) -> str:
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

def _search_with_duckduckgo(self, query, num_results):
    """DuckDuckGo search fallback"""
    try:
        from ddgs import DDGS
        
        ddgs = DDGS()
        results = []
        enhanced_query = f"{query} research paper OR study OR article"
        # enhanced_query = build_academic_query(query)
        
        for result in ddgs.text(enhanced_query, max_results=num_results):
            # print(result)
            results.append({
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", ""),
                "date": "",
                "source": _extract_domain(self, result.get("href", ""))
            })
        
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []

def _extract_domain(self, url):
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain[4:] if domain.startswith("www.") else domain
        except:
            return "unknown"
    
if __name__ =="__main__":
    pprint(_search_with_duckduckgo("1", "Artificial Intelligence", 5))
    