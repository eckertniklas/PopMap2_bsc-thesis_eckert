

import argparse


def process():


    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sh_path", type=str, help="Shapefile with boundaries and census")
    args = parser.parse_args()

    process(args.sh_path)


if __name__ == "__main__":
    main()
    print("Done!")






