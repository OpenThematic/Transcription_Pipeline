import os
import openai

"""
Your role: Expert python programmer, LLM Expert, LLAMAINDEX expert.

Let's create a simple LLAMA Index based python application with the following requirements:

For the Data Loader we will use the JSON dataloader: from llama_index.readers.json import JSONReader

The JSON Structure is a list of dicts with the following:
[
    {
        "start": 1.0999999999999996,
        "end": 4.32,
        "speaker": "SPEAKER_00",
        "text": " ...fungerar, den tar upp allt ljud, schysst."
    },
    {
]

Metadata, start, end, speaker
document, text


For the LLM we will use together.ai: from llama_index.llms.together import TogetherLLM

For embeddings we will use Nomic: from llama_index.embeddings.nomic import NomicEmbedding

For vector storage we will use Chroma: from llama_index.vector_stores.chroma import ChromaVectorStore

When Retrieving documents from the vector store, we would like to ALSO collect nearby documents based on the following logic:
All ajacent documents with the same speaker should be included.  The first document with a different speaker should be included on each end, such that the documents or chunks we pull look as follows:

{
Doc1: Speaker_01 - text
Doc2: Speaker_00 - text
Doc3: Speaker_00 - text
Doc4: Speaker_00 - text (target doc)
Doc5: Speaker_00 - text
Doc6: Speaker_02 -text
}

Once the documents are retrieved we will use a COMPACT chat response mode to query the LLM with a chat like interface.

https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html

##Additional Requireements:
1) The initial document intake/chunking/embedding/vector storage should only be performed once. The resulting vector store should persist to local storage and be retrieved on subsequent runs if using the same target file.
https://github.com/run-llama/llama_index/blob/main/docs/examples/ingestion/document_management_pipeline.ipynb (example cache)
https://github.com/run-llama/llama_index/blob/main/docs/examples/ingestion/ingestion_gdrive.ipynb Ingestion from Gdrive (redis / cache)


2) Once loaded, the main loop should allow the user to continue  to chat with the document.  Including memory of prior responses.

##Documents and resources:
Concepts high level overview of llama index https://docs.llamaindex.ai/en/stable/understanding/understanding.html
API Reference: https://docs.llamaindex.ai/en/stable/api_reference/index.html

The Service context is important when customizing configurations (such as LLM / vector storage etc) https://docs.llamaindex.ai/en/stable/api_reference/service_context.html

3) Document summary index - create a summary on top of each index which can be used for querying first: https://github.com/run-llama/llama_index/blob/main/docs/examples/index_structs/doc_summary/DocSummary.ipynb

3) Fine tuning / training - https://github.com/run-llama/llama_index/blob/main/docs/examples/llama_dataset/downloading_llama_datasets.ipynb Run with GPT4 / fine tune lower model.

4) Semantic Chunking - https://github.com/run-llama/llama_index/blob/main/docs/examples/node_parsers/semantic_chunking.ipynb

5) Llama index python package registry - https://pretty-sodium-5e0.notion.site/ce81b247649a44e4b6b35dfb24af28a6?v=53b3c2ced7bb4c9996b81b83c9f01139 

6) JSON Loader - Langchain - https://python.langchain.com/docs/modules/data_connection/document_loaders/json
"""

