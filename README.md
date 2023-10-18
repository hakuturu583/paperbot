# Paper bot

Read and manage papers easily with language model.

[![](https://img.youtube.com/vi/4N1_B57tRiI/0.jpg)](https://www.youtube.com/watch?v=4N1_B57tRiI)

# How to use.
1. Clone this repository.

```bash
git clone https://github.com/hakuturu583/paperbot.git
```

2. Install poetry
3. Setup environment with poetry.

```
cd paperbot
poetry install .
```
4. Setup qdrant cloud

Please access [this URL](https://qdrant.tech/) and register setup your qdrant instance.

5. Export environment variable for qdrant.

```bash
export QDRANT_API_KEY={API_KEY_OF_YOUR_QDRANT_INSTANCE}
export QDRANT_URI={URL_OF_YOUR_QDRANT_INSTANCE}
```

6. Contract Open AI API and get API key.
See also [this](https://book.st-hakky.com/docs/open-ai-create-api-key/).

7. Export environment variable for Open AI.

```bash
export OPENAI_API_KEY={API_KEY_OF_OPENAI}
```

# Roadmap
[x] Summary paper.
[x] Answer question about paper.
[x] Japanese support.
[x] Gradio integration.
[ ] Slack integration. 
[ ] Find paper with arxiv API.
