import sys

RESET = "\033[0m"
COLOR_INFO = "\033[94m"   # bright blue
COLOR_WARNING = "\033[93m"  # bright yellow
COLOR_ERROR = "\033[91m"   # bright red
COLOR_DEBUG = "\033[96m"   # bright cyan
COLOR_SUCCESS = "\033[92m" # bright green


def _format_message(label: str, emoji: str, color: str, message: str) -> str:
    return f"{color}{emoji} [{label}] {message}{RESET}"


def log_info(message: str) -> None:
    """Log an informational message."""
    print(_format_message("INFO", "ℹ️", COLOR_INFO, message), file=sys.stdout)


def log_warning(message: str) -> None:
    """Log a warning message."""
    print(_format_message("WARNING", "⚠️", COLOR_WARNING, message), file=sys.stdout)


def log_error(message: str) -> None:
    """Log an error message."""
    print(_format_message("ERROR", "❌", COLOR_ERROR, message), file=sys.stderr)


def log_debug(message: str) -> None:
    """Log a debug message."""
    print(_format_message("DEBUG", "🐛", COLOR_DEBUG, message), file=sys.stdout)


def log_success(message: str) -> None:
    """Log a success message."""
    print(_format_message("SUCCESS", "✅", COLOR_SUCCESS, message), file=sys.stdout)
