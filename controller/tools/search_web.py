"""
Web Search using SearXNG
"""
import re
import requests
import json
import utils.system_prompts
from datetime import datetime
from bs4 import BeautifulSoup
from config import config

from .tool_registry import tool, tool_registry

logger = config.get_logger("WebSearch")

def searxng_search(query, num_results=3):
    """
    Runs a search query against the local SearxNG instance and returns top result URLs.
    """
    payload = {
        'q': query,
        'format': 'json',
        'categories': 'general'
    }
    resp = requests.get(config.service.searxng_url, params=payload)
    resp.raise_for_status()
    results = resp.json().get('results', [])
    top_urls = [r['url'] for r in results[:num_results]]
    return top_urls

def extract_main_text(html):
    # Extract visible text from main body
    soup = BeautifulSoup(html, "html.parser")
    for bad in soup(["script", "style", "noscript", "footer", "header", "nav", "aside", "form"]):
        bad.decompose()
    # Combine text from all paragraphs
    p_texts = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
    if not p_texts:
        text = soup.get_text(separator=" ", strip=True)
    else:
        text = "\n".join(p_texts)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text)
    return text

def fetch_website_summary(url, max_length=3000):
    """
    Fetches the main text from a URL and returns a summary.
    """
    text = ""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        html = resp.text

        # Extract main readable content
        text = extract_main_text(html)

        # TODO: LLM summarization option? Bart or Pegasus?
        text = text[:max_length]

        return text
    except Exception as e:
        return text

@tool(
    name="external_information",
    description="Retrieve news and current event information through web search",
    aliases=["web_search", "current_events", "fact_search"]
)
def external_information(query: str = "get me the latest news stories") -> str:
    """
    Get latest information regarding news, facts and current events using SearXNG
    
    Args:
        query: the web search query
        
    Returns:
        LLM Response on retrieved information
    """
    top_urls = searxng_search(query, num_results=3)
    website_snippets = []
    for url in top_urls:
        snippet = fetch_website_summary(url)
        website_snippets.append(f"From {url}: {snippet}...")
    
    today = datetime.now().strftime("%B %d, %Y")

    prompt = f"""
Today is {today}.

User question: 
{query}

Web snippets:
{chr(10).join(website_snippets)}
"""
    
    payload = {
                "model": config.model.intent_model,
                "stream": True,
                "think": config.model.intent_think,
                "system": utils.system_prompts.getWebSummaryPrompt(),
                "prompt": prompt
                }
    
    logger.debug(f"Web Search Result Payload: {payload}")
    
    response = requests.post(
                        config.service.ollama_intent_url,
                        json=payload,
                        stream=True
                    )
    
    full_response = ""
                
    for line in response.iter_lines():     
        if line:
            data = json.loads(line.decode("utf-8"))
            full_response += data.get("response", "")

    # Clean, strip and remove thinking before checking intent or saving to chat history
    final_response = re.sub(
        r'^.*?</think>',
        '',
        full_response,
        flags=re.DOTALL
    )

    return final_response.strip()

if __name__ == "__main__":
    print("Web Search")
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calling:")
    queries = ["who is the current us president", "who is top of the formula 1 driver championship", "summarise the latest research on autism"]

    for query in queries:
        result = tool_registry.execute_tool("external_information", kwargs={"query": query})
        print(f"Query: {query}, Result: {result}")


