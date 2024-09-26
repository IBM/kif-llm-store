# KIF LLM Store #

A [Knowledge Integration Framework (KIF)](https://github.com/IBM/kif) plugin to provide a Wikidata-view over Large Language Models (LLM).

## What is it? ##

KIF is a framework designed for integrating diverse knowledge sources, including RDF-based interfaces, relational databases and, CSV files. It leverages the Wikidata's data model and vocabulary to expose a unified view of the integrated sources. The result is a virtual knowledge base which behaves like an "extended Wikidata" that can be queried through a lightweight query interface. More details about KIF can be found in [this paper](https://arxiv.org/abs/2403.10304).

For a data source to be accessed via KIF filters--KIF's query interface--it is necessary to create a `Store` that, based on user-defined mappings, will enable access to the underlying data source in its native language.

LLM Store is a KIF Store whose underlying data sources are LLMs. Therefore, when issuing KIF filters to LLM Store, it will be transformed into prompts that will probe the underlying LLM.


LLM Store is powered by [LangChain](https://www.langchain.com/langchain)!


## Getting started ##

### Installation ###
---
#### Using PyPI (soon) ####

```
pip install kif-llm-store
```
#### Using this repository ####

1. Clone this repository:

  ```bash
  git clone https://github.com/IBM/kif-llm-store
  cd kif-llm-store
  ```

2. Create a virtual environment, activate it, and install the requirements:

  ```bash
  python -m venv venv
  ```

  ```bash
  source venv/bin/activate
  ```

  ```bash
  pip install -r requirements.txt
  ```

3. Set the environment variables
  ```
  LLM_API_KEY=your_api_key
  LLM_API_ENDPOINT=platform_endpoint
  ```
---

To instantiate an LLM Store it is necessary to indicate a LLM provider to access models. The LLM provider can be `open_ai` to access models from OpenAI, `ibm` to access models from IBM WatsonX, and `ollama` to access models from [Ollama](https://ollama.com/). Depending on the plataform selected you need to provide the credentials to access it.

### Imports: ###

```python
# Import KIF namespacee
from kif_lib import *
# Import LLM Store main abstraction
from kif_llm_store import LLM_Store
# Import LLM Providers identifiers to set the LLM Provider in which the LLM Store will run over.
from kif_llm_store.store.llm.constants import LLM_Providers
```

### Instantiate it: ###

```python
# Using IBM WatsonX models
kb = Store(LLM_Store.store_name,
    llm_provider=LLM_Providers.IBM,
    model_id='meta-llama/llama-3-70b-instruct',
    api_key=os.environ['LLM_API_KEY'],
    base_url=os.environ['LLM_API_ENDPOINT'],
    model_params={
        'decoding_method': 'greedy',
    },
    project_id =os.environ['WATSONX_PROJECT_ID'],
)
```
```python
# Using OpenAI models
kb = Store(LLM_Store.store_name,
    llm_provider=LLM_Providers.OPEN_AI,
    model_id='gpt-4o',
    api_key=os.environ['LLM_API_KEY'],
    model_params={
        'temperature': 0,
    },
)
```

As KIF LLM Store uses LangChain, you can instantiate LLM Store direct with a [LangChain Chat Model](https://python.langchain.com/v0.2/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html), for intance:

```python
# Import LangChain OpenAI Integration
from langchain_openai import ChatOpenAI

# Instantiate a LangChain model for OpenAI
model = ChatOpenAI(model='gpt-3.5-turbo', api_key=os.environ['LLM_API_KEY'])

# Instantiate a LLM Store passing the model as a parameter
kb = Store(store_name=LLM_Store.store_name, model=model)
```

This approach enables you to run LLM Store with any LangChain Integration not only the models listed in `LLM_Providers`.

## Hello World ##

Matches statements where the subject is the Wikidata Item representing the entity `Brazil` and the property is the Wikidata Property for `shares border with`. This filter should retrieves statements linking to other entities that share a border with it:

```python
stmts = kb.filter(subject=wd.Brazil, property=wd.shares_border_with, limit=10)
for stmt in stmts:
    display(stmt)
```

## Documentation ##

See [documentation](https://ibm.github.io/kif-llm-store/) and [examples](./examples).


## Citation ##

Marcelo Machado, João M. B. Rodrigues, Guilherme Lima, Sandro
R. Fiorini, Viviane T. da Silva. ["LLM Store: Leveraging Large Language Models as Sources of Wikidata-Structured Knowledge"].
2024.

## ISWC LM-KBC 2024 Challenge
Our LLM Store solution to the [ISWC LM-KBC 2024 Challenge](https://lm-kbc.github.io/challenge2024/) can be [accessed here](https://github.com/IBM/kif-llm-store/tree/lm-kbc-challenge).


## License ##

Released under the [Apache-2.0 license](./LICENSE).










