import os
import json
import nltk

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.mgmt.search import SearchManagementClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from langchain_community.vectorstores import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from datastore import DocumentBase

embedding_model_name = "text-embedding-3-large"
azure_openai_embedding_dimensions = 1024

nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")

class DocumentIndex(object):

    def __init__(self, config):
        self.index_name = None

        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002", 
            api_version="2023-05-15", 
            azure_endpoint="https://caio-openai.openai.azure.com/", 
            api_key = config["AZURE_OPENAI_EMBEDDING_API_KEY"],
            # tiktoken_enable=True, 
        )

        self.search_endpoint = "https://caioam-ai-search.search.windows.net"
        self.search_api_key = config["AZURE_SEARCH_API_KEY"]
        
        self.documentBase = config["document_base"]
        
        self.index_client = SearchIndexClient(
                endpoint=self.search_endpoint, 
                credential=AzureKeyCredential(key=self.search_api_key)
            )
        

    def create_index(self, index_name, documents = None):
        knowlege_store = AzureSearch(
            azure_search_endpoint=self.search_endpoint,
            azure_search_key=self.search_api_key,
            index_name=index_name,
            embedding_function=self.embeddings.embed_query,
            semantic_configuration_name="default"
        )

        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            knowlege_store.add_documents(documents=docs)

        return knowlege_store
    
    def search_index(self, index_name, search_key):
        retriever = self.get_retriever(index_name)
        output = retriever.invoke("Find the content for "+search_key)
        return output
    
    def add_documents_to_index(self, docid, document_path):
        documents = self.documentBase.load_documents(document_path)
        filelist = []
        for document in documents:
            filename = os.path.basename(document.metadata['source'])
            result = self.add_document(docid, document)
            
            filelist.append(filename)
            
        return filelist
    
    def add_document(self, index_name, document):
        try:
            knowlege_store = AzureSearch(
                azure_search_endpoint=self.search_endpoint,
                azure_search_key=self.search_api_key,
                index_name=index_name,
                embedding_function=self.embeddings.embed_query,
                semantic_configuration_name="default"
            )

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            docs = text_splitter.split_documents([document])
            # print(docs)
            result = knowlege_store.add_documents(documents=docs)
            return len(result)
        except Exception as e:
            print("Error")
            raise(e)
            # return 0

    def remove_document(self, index_name, document):
        knowlege_store = AzureSearch(
            azure_search_endpoint=self.search_endpoint,
            azure_search_key=self.search_api_key,
            index_name=index_name,
            embedding_function=self.embeddings.embed_query,
            semantic_configuration_name="default"
        )
        
        knowlege_store.delete(['ids'])

    def exists_index(self, docid):
        # print(self.index_client)
        try:
            self.index_client.get_index(docid)
            return True
        except Exception as e:
            print(e)

        return False
    
    def delete_index(self, docid):
        if self.exists_index(docid):
            self.index_client.delete_index(docid)

    def get_retriever(self, docid):
        retriever = AzureAISearchRetriever(content_key="content", top_k=1, index_name=docid, 
                                       service_name=self.search_endpoint, 
                                       api_key=self.search_api_key,
                                       )
        
        return retriever


    # Create a search index
    def get_or_create_knowledge_index(self, index_name): 
        index_client = SearchIndexClient(
            endpoint=self.azure_openai_endpoint, 
            credential=AzureKeyCredential(key=self.search_api_key))
        
        search_index = index_client.get_index(name = index_name)

        if search_index == None:
            self.create_knowledge_index(index_name)
            documents = self.documentBase.load_documents(index_name)
            self.update_index_with_documents(index_name, documents)

        return search_index

    def create_knowledge_index(self, index_name):
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="category", type=SearchFieldDataType.String,
                            filterable=True),
            SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=azure_openai_embedding_dimensions, vector_search_profile_name="myHnswProfile"),
            SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=azure_openai_embedding_dimensions, vector_search_profile_name="myHnswProfile"),
        ]

        # Configure the vector search configuration  
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw"
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                    vectorizer_name="myVectorizer"
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="myVectorizer",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=self.azure_openai_endpoint,
                        # deployment_name=azure_openai_embedding_deployment,
                        # model_name=embedding_model_name,
                        api_key=self.azure_openai_key
                    )
                )
            ]
        )



        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                keywords_fields=[SemanticField(field_name="category")],
                content_fields=[SemanticField(field_name="content")]
            )
        )

        # Create the semantic settings with the configuration
        semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create the search index with the semantic settings
        index = SearchIndex(name=index_name, fields=fields,
                            vector_search=vector_search, semantic_search=semantic_search)
        

        result = self.index_client.create_or_update_index(index)
        print(f'{result.name} created')

    def create_embeddings(self, documents):
        # for document in documents:
        titles = [item['title'] for item in documents]
        content = [item['content'] for item in documents]
        
        title_response = self.embeddings.create(input=titles, model=embedding_model_name, dimensions=azure_openai_embedding_dimensions)
        title_embeddings = [item.embedding for item in title_response.data]
        
        content_response = self.embeddings.create(input=content, model=embedding_model_name, dimensions=azure_openai_embedding_dimensions)
        content_embeddings = [item.embedding for item in content_response.data]

        for i, item in enumerate(documents):
            # Generate embeddings for title and content fields
            title = item['title']
            content = item['content']
            item['titleVector'] = title_embeddings[i]
            item['contentVector'] = content_embeddings[i]

        # Output embeddings to docVectors.json file
        output_path = os.path.join('..', 'output', 'docVectors.json')
        output_directory = os.path.dirname(output_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with open(output_path, "w") as f:
            json.dump(documents, f)

    def update_index_with_documents(self, index_name, documents):
        self.create_embeddings(documents)
        output_path = os.path.join('..', 'output', 'docVectors.json')
        output_directory = os.path.dirname(output_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with open(output_path, 'r') as file:  
            documents = json.load(file)  
        search_client = SearchClient(endpoint=self.search_endpoint, index_name=index_name, credential=AzureKeyCredential(self.search_api_key))
        result = search_client.upload_documents(documents)
        print(f"Uploaded {len(documents)} documents") 