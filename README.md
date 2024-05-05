# Local LLM Chatbot

For data privacy reasons and to save costs, I've created ChadGPT, the poor man's ChatGPT!

Pay attention to tokens per second to understand your hardware's performance for LLM inference.

## Dependencies

Outside of dependencies in `pyproject.toml` that can be installed with `poetry install` based on lock file. Make sure
you install [Ollama](https://ollama.com/) and whatever model you choose. Here I'm using "llama3" but this can be easily
parametrized.

## Usage

```
streamlit run chadgpt.py
```