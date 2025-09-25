# sentiment_pipeline.py
import os
import mlflow
import argparse
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchRun, YahooFinanceNewsTool
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
import yfinance as yf
import json
import re

# Pydantic model for structured output
class SentimentProfile(BaseModel):
    """Structured sentiment profile for the company."""
    company_name: str = Field(description="Name of the company")
    stock_code: str = Field(description="Stock ticker symbol")
    newsdesc: str = Field(description="Concise summary of the fetched news")
    sentiment: str = Field(description="Overall sentiment: Positive, Negative, or Neutral")
    people_names: List[str] = Field(description="List of people names mentioned")
    places_names: List[str] = Field(description="List of places mentioned")
    other_companies_referred: List[str] = Field(description="List of other companies referred to")
    related_industries: List[str] = Field(description="List of related industries")
    market_implications: str = Field(description="Implications for the market or stock")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")

# Initialize LLM (Azure OpenAI)
def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",  # Use appropriate version
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),  # or "gpt-4o"
        temperature=0.1,
    )

# Function to extract ticker using search and LLM
def get_stock_ticker(company_name: str, llm, search_tool):
    ticker_prompt = ChatPromptTemplate.from_template(
        "Based on the following search results, extract the primary stock ticker symbol (e.g., AAPL) for the company '{company_name}'. "
        "The ticker should be for the main listing on NASDAQ or NYSE. Return only the ticker symbol in uppercase, nothing else.\n\n"
        "Search results: {search_results}"
    )
    
    # Search for ticker
    search_query = f"stock ticker symbol for {company_name}"
    search_results = search_tool.run(search_query)
    
    # Use LLM to parse ticker
    chain = (
        {"company_name": RunnablePassthrough(), "search_results": search_results}
        | ticker_prompt
        | llm
    )
    
    ticker = chain.invoke(company_name).content.strip()
    
    # Validate with yfinance
    if not ticker or not re.match(r'^[A-Z]{1,5}$', ticker):
        raise ValueError(f"Invalid ticker extracted: {ticker}")
    
    stock = yf.Ticker(ticker)
    if stock.info.get('symbol') != ticker:
        raise ValueError(f"Ticker {ticker} not valid for {company_name}")
    
    return ticker

# Function to fetch and summarize news
def fetch_news(ticker: str, news_tool):
    news_items = news_tool.run(ticker)
    # news_items is a string, parse to list of dicts if needed; assuming it's formatted
    # For simplicity, split by lines or use as is
    if isinstance(news_items, str):
        # Basic split for headlines
        headlines = [line.strip() for line in news_items.split('\n') if line.strip()]
    else:
        headlines = [item.get('title', '') for item in news_items if isinstance(item, dict)]
    
    news_summary = " ".join(headlines[:10])  # Top 10 headlines
    return news_summary

# Main pipeline function with MLflow tracing
def run_sentiment_pipeline(company_name: str):
    with mlflow.start_run(run_name=f"Sentiment Pipeline for {company_name}"):
        mlflow.log_param("input_company", company_name)
        
        llm = get_llm()
        search_tool = DuckDuckGoSearchRun()
        news_tool = YahooFinanceNewsTool()
        
        # Step 1: Extract ticker
        with mlflow.start_run(nested=True, run_name="ticker_extraction"):
            mlflow.log_param("step", "ticker_extraction")
            ticker = get_stock_ticker(company_name, llm, search_tool)
            mlflow.log_param("extracted_ticker", ticker)
            mlflow.log_metric("ticker_confidence", 1.0)  # Assume high for now
        mlflow.end_run()
        
        # Step 2: Fetch news
        with mlflow.start_run(nested=True, run_name="news_fetching"):
            mlflow.log_param("step", "news_fetching")
            news_summary = fetch_news(ticker, news_tool)
            mlflow.log_param("news_summary_length", len(news_summary))
        mlflow.end_run()
        
        # Step 3: Analyze sentiment
        with mlflow.start_run(nested=True, run_name="sentiment_analysis"):
            mlflow.log_param("step", "sentiment_analysis")
            
            # Prompt for sentiment analysis
            analysis_prompt = ChatPromptTemplate.from_template(
                "Analyze the following news about {company_name} ({stock_code}) and generate a structured sentiment profile.\n\n"
                "News: {news_summary}\n\n"
                "Classify the overall sentiment as Positive, Negative, or Neutral.\n"
                "Extract: people names, places, other companies, related industries.\n"
                "Provide market implications and a confidence score (0.0 to 1.0).\n"
                "Output ONLY valid JSON matching this schema: {schema}"
            )
            
            # Get schema for JSON output
            parser = JsonOutputParser(pydantic_object=SentimentProfile)
            format_instructions = parser.get_format_instructions()
            
            chain = (
                analysis_prompt.partial(schema=format_instructions)
                | llm
                | parser
            )
            
            result = chain.invoke({
                "company_name": company_name,
                "stock_code": ticker,
                "news_summary": news_summary
            })
            
            # Log prompt (for debugging, log the input to LLM)
            mlflow.log_param("analysis_prompt_input", json.dumps({
                "company_name": company_name,
                "stock_code": ticker,
                "news_summary": news_summary[:500]  # Truncate for logging
            }))
            
            mlflow.log_param("sentiment", result.get("sentiment", ""))
            mlflow.log_metric("confidence_score", result.get("confidence_score", 0.0))
        
        mlflow.end_run()
        
        # Log the full output
        mlflow.log_dict(result.dict(), "sentiment_profile")
        
        return result.dict()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run sentiment pipeline for a company")
#     parser.add_argument("--company", type=str, required=True, help="Company name (e.g., 'Microsoft')")
#     args = parser.parse_args()
    
#     result = run_sentiment_pipeline(args.company)
#     print(json.dumps(result, indent=2))