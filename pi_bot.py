#!/usr/bin/env python3
"""Thin shim — preserves ``python pi_bot.py [--chat]``."""

from pi_bot.main import cli

if __name__ == "__main__":
    cli()
