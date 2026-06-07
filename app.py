"""Hugging Face Spaces entrypoint.

HF Spaces expects a file named ``app.py`` at the repository root that exposes
a ``demo`` object (or calls ``demo.launch()``). This thin wrapper delegates to
the actual UI module so the Makefile target (``make app``) also works.
"""
from app.gradio_app import demo  # noqa: F401

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
