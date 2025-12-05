"""CLI interface for legal provision retrieval system."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from .indexing import build_index
from .retrieval import LegalRetriever
from .models import SearchResult


def generate_html_results(
    results: List[SearchResult],
    query: str,
    output_path: str,
    search_params: dict
) -> None:
    """Generate a nice HTML file with search results."""
    from html import escape
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LegalFlow Search Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .query-box {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #3498db;
        }}
        
        .query-text {{
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .meta-info {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 15px 0;
            font-size: 0.9em;
            color: #666;
        }}
        
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .meta-label {{
            font-weight: 600;
            color: #555;
        }}
        
        .results-count {{
            background: #27ae60;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 600;
            margin: 15px 0;
            display: inline-block;
        }}
        
        .result {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            transition: box-shadow 0.3s;
        }}
        
        .result:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .result-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .result-number {{
            background: #3498db;
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .result-scores {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .score-badge {{
            background: #ecf0f1;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 600;
        }}
        
        .score-combined {{
            background: #3498db;
            color: white;
        }}
        
        .score-embedding {{
            background: #9b59b6;
            color: white;
        }}
        
        .score-keyword {{
            background: #e67e22;
            color: white;
        }}
        
        .result-meta {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 15px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .meta-badge {{
            background: #ecf0f1;
            padding: 4px 10px;
            border-radius: 12px;
        }}
        
        .method-badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85em;
        }}
        
        .method-embedding {{
            background: #9b59b6;
            color: white;
        }}
        
        .method-keyword {{
            background: #e67e22;
            color: white;
        }}
        
        .method-both {{
            background: #27ae60;
            color: white;
        }}
        
        .result-text {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.8;
            font-size: 1.05em;
        }}
        
        .source-file {{
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .article-hint {{
            color: #27ae60;
            font-weight: 600;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            .result-header {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 LegalFlow Search Results</h1>
            <div class="query-box">
                <div class="query-text">Query: {escape(query)}</div>
            </div>
            <div class="meta-info">
                <div class="meta-item">
                    <span class="meta-label">Date:</span>
                    <span>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Results:</span>
                    <span class="results-count">{len(results)}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Top-K:</span>
                    <span>{search_params.get('top_k', 'N/A')}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Min Score:</span>
                    <span>{search_params.get('min_score', 'N/A')}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Search Multiplier:</span>
                    <span>{search_params.get('search_multiplier', 'N/A')}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Weights:</span>
                    <span>Embedding: {search_params.get('weight_embedding', 'N/A')}, Keyword: {search_params.get('weight_keyword', 'N/A')}</span>
                </div>
            </div>
        </div>
"""
    
    # Add each result
    for i, result in enumerate(results, 1):
        # Determine method badge class
        method_class = f"method-{result.method}"
        
        # Build scores HTML
        scores_html = f'<span class="score-badge score-combined">Combined: {result.score:.4f}</span>'
        if result.embedding_score is not None:
            scores_html += f'<span class="score-badge score-embedding">Embedding: {result.embedding_score:.4f}</span>'
        if result.keyword_score is not None:
            scores_html += f'<span class="score-badge score-keyword">Keyword: {result.keyword_score:.4f}</span>'
        
        # Build metadata HTML
        meta_html = f'<span class="meta-badge">ID: {result.id}</span>'
        meta_html += f'<span class="meta-badge source-file">Source: {escape(result.source_file)}</span>'
        if result.article_hint:
            meta_html += f'<span class="meta-badge article-hint">{escape(result.article_hint)}</span>'
        meta_html += f'<span class="method-badge {method_class}">Method: {result.method}</span>'
        
        # Get full text (not truncated)
        full_text = result.text
        
        html_content += f"""
        <div class="result">
            <div class="result-header">
                <div class="result-number">{i}</div>
                <div class="result-scores">
                    {scores_html}
                </div>
            </div>
            <div class="result-meta">
                {meta_html}
            </div>
            <div class="result-text">{escape(full_text)}</div>
        </div>
"""
    
    # Close HTML
    html_content += """
        <div class="footer">
            <p>Generated by LegalFlow - Legal Provision Retrieval System</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write to file
    output_file.write_text(html_content, encoding='utf-8')
    print(f"\n✓ HTML results saved to: {output_file.absolute()}")


def cmd_build_index(args: argparse.Namespace) -> None:
    """Command to build index."""
    try:
        build_index(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_chunk_size=args.max_chunk_size
        )
    except Exception as e:
        print(f"Error building index: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    """Command to search."""
    try:
        retriever = LegalRetriever(
            index_dir=args.index_dir,
            model_name=args.model_name
        )

        # If top_k=0, set to very large value (all results)
        top_k = args.top_k if args.top_k > 0 else 999999

        results = retriever.search(
            query=args.query,
            top_k=top_k,
            min_score=args.min_score,
            search_multiplier=args.search_multiplier,
            weight_embedding=args.weight_embedding,
            weight_keyword=args.weight_keyword,
            display_limit=args.display_limit,
        )

        if not results:
            print("No results found matching criteria.")
            if args.output_html:
                # Still generate HTML with no results message
                generate_html_results(
                    results=[],
                    query=args.query,
                    output_path=args.output_html,
                    search_params={
                        'top_k': top_k,
                        'min_score': args.min_score,
                        'search_multiplier': args.search_multiplier,
                        'weight_embedding': args.weight_embedding,
                        'weight_keyword': args.weight_keyword,
                    }
                )
            return

        print(f"\nFound {len(results)} results:\n")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result}")
            print("-" * 80)
        
        # Generate HTML output if requested
        if args.output_html:
            generate_html_results(
                results=results,
                query=args.query,
                output_path=args.output_html,
                search_params={
                    'top_k': top_k,
                    'min_score': args.min_score,
                    'search_multiplier': args.search_multiplier,
                    'weight_embedding': args.weight_embedding,
                    'weight_keyword': args.weight_keyword,
                }
            )

    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="LegalFlow - Legal provision retrieval system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build-index command
    parser_build = subparsers.add_parser(
        "build-index",
        help="Build FAISS index from .txt files"
    )
    parser_build.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with .txt files containing legal provisions"
    )
    parser_build.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for index and metadata"
    )
    parser_build.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Name of embedding model (default: paraphrase-multilingual-MiniLM-L12-v2)"
    )
    parser_build.add_argument(
        "--max-chunk-size",
        type=int,
        default=1200,
        help="Maximum chunk length in characters (default: 1200)"
    )
    parser_build.set_defaults(func=cmd_build_index)

    # query command
    parser_query = subparsers.add_parser(
        "query",
        help="Search for legal provision fragments"
    )
    parser_query.add_argument(
        "--index-dir",
        type=str,
        default="./data/index",
        help="Directory containing index.faiss and metadata.json (default: ./data/index)"
    )
    parser_query.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text"
    )
    parser_query.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Maximum number of results (default: 100). Use 0 to display all results."
    )
    parser_query.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score (cosine similarity) to include (default: 0.0)"
    )
    parser_query.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Name of embedding model (must be same as used for building)"
    )
    parser_query.add_argument(
        "--search-multiplier",
        type=float,
        default=100.0,
        help="Multiplier determining how many more candidates to search than top_k (default: 100.0)"
    )
    parser_query.add_argument(
        "--weight-embedding",
        type=float,
        default=1.0,
        help="Weight for embedding-based score (default: 1.0)"
    )
    parser_query.add_argument(
        "--weight-keyword",
        type=float,
        default=1.0,
        help="Weight for keyword/BM25-based score (default: 1.0)"
    )
    parser_query.add_argument(
        "--display-limit",
        type=int,
        default=300,
        help="Maximum characters to display in text preview (default: 300). Use 0 to display full text."
    )
    parser_query.add_argument(
        "--output-html",
        type=str,
        default=None,
        help="Path to save HTML file with search results (e.g., ./output/results.html). If not specified, no HTML file is generated."
    )
    parser_query.set_defaults(func=cmd_query)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
