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
        result = converter.convert(url)
        
        markdown_content = result.document.export_to_markdown()
        
        # Simple chunking: Split by double newlines (paragraphs)
        chunks = [chunk.strip() for chunk in markdown_content.split("\n\n") if chunk.strip()]
        
        data = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "source_url": url,
                "chunk_id": i,
                "char_count": len(chunk), # NEW: Character count
                "word_count": len(chunk.split()), # NEW: Simple word count
                # You could add 'has_table' or 'has_code' if Docling provides such metadata
                # or if you want to implement simple regex checks here.
            }
            data.append(chunk_data)
        return data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def get_website_content_as_polars_df(url: str) -> pl.DataFrame:
    """
    Scrapes the website and returns content in a Polars DataFrame.
    Includes character and word counts for each chunk.
    """
    scraped_data = scrape_website_with_docling(url)
    if scraped_data:
        # Explicitly cast new columns for robustness
        return pl.DataFrame(scraped_data).with_columns([
            pl.col("char_count").cast(pl.UInt32),
            pl.col("word_count").cast(pl.UInt32)
        ])
    # Return an empty DataFrame with expected schema including new columns
    return pl.DataFrame({
        "text": pl.Series(dtype=pl.String),
        "source_url": pl.Series(dtype=pl.String),
        "chunk_id": pl.Series(dtype=pl.UInt32),
        "char_count": pl.Series(dtype=pl.UInt32),
        "word_count": pl.Series(dtype=pl.UInt32)
    })

if __name__ == "__main__":
    test_url = "https://docs.pola.rs/py-polars/html/reference/dataframe/index.html"
    print(f"Scraping: {test_url}")
    df = get_website_content_as_polars_df(test_url)
    print(df.head())
    print(f"Total chunks: {len(df)}")
    if not df.is_empty():
        print("\nChunk Statistics:")
        print(df.select(
            pl.col("char_count").mean().alias("Avg Char Count"),
            pl.col("char_count").median().alias("Median Char Count"),
            pl.col("char_count").min().alias("Min Char Count"),
            pl.col("char_count").max().alias("Max Char Count"),
            pl.col("word_count").mean().alias("Avg Word Count"),
            pl.col("word_count").median().alias("Median Word Count"),
        ))
