# rag_website_explainer/web_scraper.py
from docling.document_converter import DocumentConverter
import polars as pl
from typing import List, Dict, Any

def scrape_website_with_docling(url: str) -> List[Dict[str, Any]]:
    """
    Scrapes a website using Docling and returns its content as a list of dictionaries.
    Each dictionary represents a section/chunk of the document.
    """
    converter = DocumentConverter()
    try:
        # Convert the URL to a document object
        result = converter.convert(url)
        
        # Export the document content to Markdown for easier text extraction
        markdown_content = result.document.export_to_markdown()
        
        # Simple chunking: Split by double newlines (paragraphs)
        # For more advanced chunking, LangChain's text splitters are recommended
        # after getting the full text.
        chunks = [chunk.strip() for chunk in markdown_content.split("\n\n") if chunk.strip()]
        
        # Create a list of dictionaries with text and metadata
        data = [{"text": chunk, "source_url": url, "chunk_id": i} for i, chunk in enumerate(chunks)]
        return data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def get_website_content_as_polars_df(url: str) -> pl.DataFrame:
    """
    Scrapes the website and returns content in a Polars DataFrame.
    """
    scraped_data = scrape_website_with_docling(url)
    if scraped_data:
        return pl.DataFrame(scraped_data)
    # Return an empty DataFrame with expected schema if no data is scraped
    return pl.DataFrame({
        "text": pl.Series(dtype=pl.String),
        "source_url": pl.Series(dtype=pl.String),
        "chunk_id": pl.Series(dtype=pl.UInt32)
    })

if __name__ == "__main__":
    test_url = "https://docs.pola.rs/py-polars/html/reference/dataframe/index.html" # Example Polars docs page
    print(f"Scraping: {test_url}")
    df = get_website_content_as_polars_df(test_url)
    print(df.head())
    print(f"Total chunks: {len(df)}")
