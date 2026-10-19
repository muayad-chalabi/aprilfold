"""CLI entry point for aprilfold."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: aprilfold <command> [args...]\n")
        print("Commands:")
        print("  generate    Marker cube or voxel target (--mono for the foldable build)")
        print("  twin        Printed twin of a robot (URDF): one marker cube per link")
        print("  web         Open the standalone voxel target designer")
        print("\nRun 'aprilfold <command> --help' for command-specific help.")
        sys.exit(0)

    if sys.argv[1] == "--version":
        from aprilfold import __version__
        print(f"aprilfold {__version__}")
        sys.exit(0)

    command = sys.argv[1]
    sys.argv = [f"aprilfold {command}"] + sys.argv[2:]

    if command == "generate":
        from aprilfold.generate import main as generate_main
        generate_main()
    elif command == "twin":
        from aprilfold.twin import main as twin_main
        twin_main(sys.argv[1:])
    elif command == "web":
        from aprilfold.web.app import main as web_main
        web_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Available commands: generate, twin, web", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
