from utils import data_path
import argparse


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weights_name",
        help="The stem of the filename containing the weights",
        type=str,
    )
    parser.add_argument(
        "type",
        help="The type of removal procedure",
        choices=["channel", "channel_mask", "blob", "random"],
        type=str,
    )
    return parser


def main() -> None:
    mcaf_path = data_path / "MCAF"
    args = parse_args()
    weights_name = args.weights_name  # pyright: ignore[reportAttributeAccessIssue]
    mcaf_type = args.weights_name  # pyright: ignore[reportAttributeAccessIssue]
    count_path = mcaf_path / f"{mcaf_type}_{weights_name}_count.csv"
    time_path = mcaf_path / f"{mcaf_type}_{weights_name}_time.csv"


if __name__ == "__main__":
    main()
