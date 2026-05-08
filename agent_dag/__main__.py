"""
CLI Runner — ``python -m agent_dag``.

Provides a command-line interface for running the pipeline against
JSON datasets of products and stores.

Usage::

    # Run with mock LLM (zero config)
    python -m agent_dag run \\
        --products examples/data/products.json \\
        --stores examples/data/stores.json

    # Run with OpenAI
    python -m agent_dag run \\
        --products data/products.json \\
        --stores data/stores.json \\
        --llm openai --model gpt-4o-mini

    # Run with custom config
    python -m agent_dag run \\
        --products data/products.json \\
        --stores data/stores.json \\
        --config examples/config.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from .__version__ import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent_dag",
        description="Agent DAG Pipeline — 7-node autonomous content generation",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # ── run command ──
    run_p = sub.add_parser("run", help="Run the pipeline against product/store datasets")
    run_p.add_argument(
        "--products", required=True,
        help="Path to products JSON file (array of product objects)",
    )
    run_p.add_argument(
        "--stores", required=True,
        help="Path to stores JSON file (array of store context objects)",
    )
    run_p.add_argument(
        "--config", default=None,
        help="Path to pipeline configuration JSON file",
    )
    run_p.add_argument(
        "--output", default="./output",
        help="Output directory (default: ./output)",
    )
    run_p.add_argument(
        "--llm", default=None,
        help="LLM provider override: mock, openai, google (default: from config)",
    )
    run_p.add_argument(
        "--model", default=None,
        help="LLM model override (e.g., gpt-4o-mini, gemini-2.5-flash-lite)",
    )
    run_p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    return parser


async def run_batch(args: argparse.Namespace) -> int:
    """Execute the pipeline for all products × stores."""
    from .config import load_config
    from .orchestrator import run_pipeline
    from .shared.data_contracts import StoreContext
    from .shared.llm_protocol import create_llm_client
    from .shared.output_schema import PipelineRunOutput, ProductResult

    # ── Load config ──
    config = load_config(args.config)

    # ── Override LLM from CLI flags ──
    llm_provider = args.llm or config.llm.provider
    llm_model = args.model or config.llm.model
    output_dir = args.output or config.output_dir

    # ── Load datasets ──
    products_path = Path(args.products)
    stores_path = Path(args.stores)

    if not products_path.exists():
        print(f"Error: Products file not found: {products_path}", file=sys.stderr)
        return 1
    if not stores_path.exists():
        print(f"Error: Stores file not found: {stores_path}", file=sys.stderr)
        return 1

    with open(products_path) as f:
        products = json.load(f)
    with open(stores_path) as f:
        stores_data = json.load(f)

    print(f"Agent DAG Pipeline v{__version__}")
    print(f"  Products: {len(products)} items from {products_path}")
    print(f"  Stores:   {len(stores_data)} locations from {stores_path}")
    print(f"  LLM:      {llm_provider}" + (f" ({llm_model})" if llm_model else ""))
    print(f"  Output:   {output_dir}")
    print()

    # ── Create LLM client ──
    llm_client = create_llm_client(
        llm_provider,
        model=llm_model if llm_model else None,
        api_key=config.llm.resolve_api_key(),
    )

    # ── Initialize output ──
    output = PipelineRunOutput(
        pipeline_version=__version__,
        config_used={
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "products_file": str(products_path),
            "stores_file": str(stores_path),
        },
    )

    # ── Execute pipeline: products × stores ──
    batch_start = time.monotonic()
    total = len(products) * len(stores_data)
    completed = 0

    for store_data in stores_data:
        store_context = StoreContext(**store_data)
        print(f"  Store: {store_context.store_name} ({store_context.country_code})")

        for product in products:
            product_id = product.get("id", product.get("sku", "unknown"))

            try:
                result = await run_pipeline(
                    product,
                    store_context,
                    trace_id=f"cli-{store_context.store_id}-{product_id}",
                )

                product_result = ProductResult(
                    product_id=product_id,
                    success=result.success,
                    duration_ms=result.duration_ms,
                    failure_node=result.failure_node,
                    failure_reason=result.failure_reason,
                    content=result.content if isinstance(result.content, dict) else None,
                    metadata=result.metadata,
                )

                output.add_result(store_context.store_id, store_data, product_result)

                status = "✅" if result.success else "❌"
                completed += 1
                print(f"    {status} {product_id} ({result.duration_ms:.1f}ms)"
                      + (f" — {result.failure_reason}" if not result.success else ""))

            except Exception as exc:
                completed += 1
                product_result = ProductResult(
                    product_id=product_id,
                    success=False,
                    failure_node="CLI_RUNNER",
                    failure_reason=str(exc)[:200],
                )
                output.add_result(store_context.store_id, store_data, product_result)
                print(f"    ❌ {product_id} — Exception: {exc}")

    # ── Write output ──
    output_path = output.write_json(output_dir)
    batch_ms = (time.monotonic() - batch_start) * 1000

    print()
    print(f"  Done: {completed}/{total} products processed in {batch_ms:.0f}ms")
    print(f"  Pass rate: {output.summary.pass_rate:.1%}")
    print(f"  Output: {output_path}")

    return 0


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s %(levelname)s: %(message)s")

    if args.command == "run":
        exit_code = asyncio.run(run_batch(args))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
