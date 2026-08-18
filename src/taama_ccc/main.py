from __future__ import annotations

import argparse

from taama_ccc import check, rebuild_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taama-ccc",
        description="Singapore HSA/SFA claim compliance checker",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    rebuild_index_parser = subparsers.add_parser(
        "rebuild-index",
        help="Parse a regulatory corpus and (re)build the Qdrant index",
    )
    rebuild_index.add_arguments(rebuild_index_parser)

    check_parser = subparsers.add_parser(
        "check",
        help="Check a product's claims against the indexed corpus",
    )
    check.add_arguments(check_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "rebuild-index":
        rebuild_index.run(args)
    elif args.command == "check":
        check.run(args)


if __name__ == "__main__":
    main()
