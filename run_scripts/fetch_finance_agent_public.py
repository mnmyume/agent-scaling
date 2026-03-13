import argparse
import os
import urllib.request


DEFAULT_URL = "https://raw.githubusercontent.com/vals-ai/finance-agent/main/data/public.csv"
DEFAULT_OUT = os.path.join("datasets", "finance-agent-public.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Finance-Agent public.csv into the local datasets/ folder."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="CSV URL to download")
    parser.add_argument(
        "--out", default=DEFAULT_OUT, help="Output path (default: datasets/finance-agent-public.csv)"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Downloading: {args.url}")
    with urllib.request.urlopen(args.url, timeout=60) as resp:
        data = resp.read()

    with open(args.out, "wb") as f:
        f.write(data)

    print(f"Wrote: {args.out} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
