# KIF LLM Store
A Large Language Model Store for the [Knowledge Integration Framework](https://github.com/IBM/kif).

This repository contains the the implementation of our system for the [ISWC LM-KBC 2024 Challenge](https://lm-kbc.github.io/challenge2024/). 

KIF is a framework designed for integrating diverse knowledge sources, including RDF-based interfaces, relational databases and, CSV files. It leverages the Wikidata's data model and vocabulary to expose a unified view of the integrated sources. The result is a virtual knowledge base which behaves like an ``extended Wikidata'' that can be queried through a lightweight query interface.

For a data source to be accessed via queries in KIF, it is necessary to create a `Store` that, based on user-defined mappings, will enable access to the underlying data source in its native language.

The system we describe here is a Store KIF (a.k.a. KIF LLM Store) that accesses a LM. Therefore, when issuing KIF queries the KIF LLM Store transforms the pattern passed by the user into prompts for the LM.

More details about KIF can be found in [this paper](https://arxiv.org/abs/2403.10304).

In the following sections we demonstrate how to run our system based on the KIF LLM Store and show the results of our approach in Knowledge Base Completion task described in the [ISWC 2042 LM-KBC Challenge](https://lm-kbc.github.io/challenge2024/).

To instantiate an LLM Store it is necessary to indicate which platform you will use to access the Language Models. The platform can be `gpt` to access models from OpenAI, `bam` to access models from IBM Research's Big AI Model (BAM), and `hf` to access models from Hugging Face Hub. We also have a `generic` type to access models through HTTP requests, an example would be [Ollama](https://ollama.com/). Depending on the plataform selected you need to provide the credentials to access it.

For convenience, in this Challenge, we used the `Llama3-8B-Instruct` model from IBM's BAM. However, the results should remain the same considering the same model on other platforms. **Therefore, when executing the run.py command (as we will see later) we are hardcoding the platform and the model. If you want to change the platform, check the execution parameters.**


## Files

```text
.
├── data
│   ├── train.jsonl
│   └── val.jsonl
├── kif_llm_store
│   └── store
│       └── llm
│           ├── bam.py
│           ├── llm.py
├── evaluate.py
├── examples
├── README.md 
├── requirements.txt
└── run.py 

```

## Getting started


### Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/IBM/kif-llm-store
    cd kif-llm-store
    ```

2. Create a virtual environment, activate it, and install the requirements:

    ```bash
    python3.12 -m venv kif-llm-store
    ```

    ```bash
    source kif-llm-store/bin/activate
    ```

    ```bash
    pip install -r requirements.txt
    ```

3. Set the environment variables
    ```
    LLM_API_KEY=your_api_key
    LLM_API_ENDPOINT=platform_endpoint
    ```
    
4. Run the system to generate the csv file with the responses:
    ```bash
    # default settings
    python run.py -i data/lm-kbc-2024/test.jsonl
    ```
    ```bash
    # running to generate response for a limited number of relations (e.g., awardWonBy and personHasCityOfDeath)
    python run.py -i data/lm-kbc-2024/test.jsonl --filter_relation awardWonBy --filter_relation personHasCityOfDeath
    ```

    ```bash
    # Using masked template (masked based)
    python run.py -i data/lm-kbc-2024/lm-kbc-2024/test.jsonl -pt data/lm-kbc-2024/prompt_templates/masked_prompts.csv
    ```

     Our predictions to the ISWC LM-KBC 2024 Challenge was generated using the questions template in /data/lm-kbc-2024/prompt_templates/question_prompts.csv:
    ```bash
    # Using questions template (question based)
    
    python run.py -i data/lm-kbc-2024/lm-kbc-2024/test.jsonl --prompt_template data/lm-kbc-2024/prompt_templates/question_prompts.csv
    ```

    Other command line parameters (run `python run.py --help` to see all the available parameters):
   ```
   --llm or -l, Select the LLM platform (default: 'bam'),
       Usage: python run.py -i data/lm-kbc-2024/lm-kbc-2024/test.jsonl -l gpt
   
   --model_id or -m, Select the LLM Model id (default: 'meta-llama/llama-3-8b-instruct'),
       Usage: python run.py -i data/lm-kbc-2024/lm-kbc-2024/test.jsonl -l gpt -m gpt-3.5-turbo
   
    --no-context or -nc #Use this flag to execute the system without using the context extraction,
       Usage: python run.py -i data/lm-kbc-2024/lm-kbc-2024/test.jsonl -nc
   ```






  

  
