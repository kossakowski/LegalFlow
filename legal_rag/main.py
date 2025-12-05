"""CLI interface for legal provision retrieval system."""

import argparse
import sys
from pathlib import Path

from .indexing import build_index
from .retrieval import LegalRetriever


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
            return

        print(f"\nFound {len(results)} results:\n")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result}")
            print("-" * 80)

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
    parser_query.set_defaults(func=cmd_query)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
