"""Interfejs CLI do systemu retrieval przepisów prawnych."""

import argparse
import sys
from pathlib import Path

from .indexing import build_index
from .retrieval import LegalRetriever


def cmd_build_index(args: argparse.Namespace) -> None:
    """Komenda do budowy indeksu."""
    try:
        build_index(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_chunk_size=args.max_chunk_size
        )
    except Exception as e:
        print(f"Błąd podczas budowy indeksu: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    """Komenda do wyszukiwania."""
    try:
        retriever = LegalRetriever(
            index_dir=args.index_dir,
            model_name=args.model_name
        )
        
        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            min_score=args.min_score
        )
        
        if not results:
            print("Nie znaleziono żadnych wyników spełniających kryteria.")
            return
        
        print(f"\nZnaleziono {len(results)} wyników:\n")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result}")
            print("-" * 80)
        
    except Exception as e:
        print(f"Błąd podczas wyszukiwania: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="LegalFlow - System retrieval przepisów prawnych",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Dostępne komendy")
    
    # Komenda build-index
    parser_build = subparsers.add_parser(
        "build-index",
        help="Buduje indeks FAISS z plików .txt"
    )
    parser_build.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Katalog z plikami .txt zawierającymi przepisy"
    )
    parser_build.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Katalog wyjściowy dla indeksu i metadanych"
    )
    parser_build.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Nazwa modelu embeddingowego (domyślnie: paraphrase-multilingual-MiniLM-L12-v2)"
    )
    parser_build.add_argument(
        "--max-chunk-size",
        type=int,
        default=1200,
        help="Maksymalna długość chunka w znakach (domyślnie: 1200)"
    )
    parser_build.set_defaults(func=cmd_build_index)
    
    # Komenda query
    parser_query = subparsers.add_parser(
        "query",
        help="Wyszukuje fragmenty przepisów"
    )
    parser_query.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Katalog zawierający index.faiss i metadata.json"
    )
    parser_query.add_argument(
        "--query",
        type=str,
        required=True,
        help="Tekst zapytania"
    )
    parser_query.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maksymalna liczba wyników (domyślnie: 20)"
    )
    parser_query.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimalny score (cosine similarity) do uwzględnienia (domyślnie: 0.0)"
    )
    parser_query.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Nazwa modelu embeddingowego (musi być taki sam jak przy budowie)"
    )
    parser_query.set_defaults(func=cmd_query)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()


