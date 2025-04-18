import os
import json

from langchain import hub
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.tools import create_retriever_tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI, OpenAI, ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain.agents import initialize_agent, AgentExecutor, AgentType, create_react_agent, create_tool_calling_agent
from langchain_google_community import GoogleSearchAPIWrapper

from langchain_core.messages import trim_messages

from datastore import DocumentBase
from indexes import DocumentIndex

OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_ORG_ID = "OPENAI_ORG_ID"
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
os.environ["GOOGLE_CSE_ID"] = "GOOGLE_CSE_ID"

search = GoogleSearchAPIWrapper(k=2)
 
tool = Tool(
    name="web_search",
    description="Search google for recent results.",
    func=search.run
)

def handle_parsing_error(error) -> str:
    return str(error)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature="0",
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID,
)  

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID,
)

trimmer = trim_messages(
     token_counter=llm,
     strategy = "last",
     max_tokens=50,
     start_on="human",
     end_on=("human", "tool"),
     include_system=True
)

def add_documents_to_index(docid, document_path):
    index = DocumentIndex()
    documents = DocumentBase().load_documents(document_path)
    filelist = []
    for document in documents:
        filename = os.path.basename(document.metadata['source'])
        # if filename.split("_")[0] == "NorthStar":
        # print(filename)
        result = index.add_document(docid, document)
        # print(result)
        filelist.append(filename)
        
    return filelist
    

def create_knowledge_search_tool(docid):
    documents = DocumentBase().load_documents(docid)
   
    index = DocumentIndex()
    if not index.exists_index(docid):
        print("The index does not exist for ", docid)
        print("Creating new index")
        index.create_index(docid, documents)

    retriever = index.get_retriever(docid)
    
    knowledge_search_tool = create_retriever_tool(retriever, "knowledge_search", "Search through our internal knowledge base")

    return knowledge_search_tool

def query_rfp_knowledgebase(docid, text):
    prompt_messages = [
        (
            "system",
            "You are a helpful assistant with access to both a comprehensive internal knowledge base and the internet. When responding to a user's query, prioritize information from your internal database using vector search. If no relevant information is found, then perform a web search to provide an answer. Always clearly state whether the information comes from your internal knowledge base or a web search." 
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]

    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    retreiver_tool = create_knowledge_search_tool(docid)
    
    agent = create_tool_calling_agent(llm=llm, tools=[retreiver_tool, tool], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[retreiver_tool, tool], verbose=True)
    
    response = agent_executor.invoke({"input": text})
    
    return response['output']

def query_llm_with_search(text):
    try:
        # knowledge_search_tool = create_knowledge_search_tool(docid)
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, [tool], prompt)
        # agent = initialize_agent(tools=[tool], llm=llm, agent=AgentType.SELF_ASK_WITH_SEARCH)

        agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True, handle_parsing_errors=handle_parsing_error)

        # query_input = f"{text}\n\nAction: web_search"
        response = agent_executor.invoke({"input": text})

        return response['output']
    except Exception as e:
        # print(e)
        return "Error"
    
def get_rfp_response_structure(docid, rfpText, templateText=None):
    try:
        with open('prompts/generate_rfp_structure.txt', 'r') as f:
            prompt_text = f.read()
            prompt = PromptTemplate.from_template(template=prompt_text)
            retreiver_tool = create_knowledge_search_tool(docid)
    
            agent = create_react_agent(llm, [retreiver_tool, tool], prompt)
            agent_executor = AgentExecutor(agent=agent, tools=[retreiver_tool, tool], verbose=True, handle_parsing_errors=handle_parsing_error)

            response = agent_executor.invoke({"rfpText": rfpText, "refText": templateText})
            return response['output']
    except Exception as e:
        raise(e)
    
def get_rfp_response_content(sectionHeading, rfpText, refText):
    try:
        with open('prompts/generate_rfp_content.txt', 'r') as f:
            prompt_text = f.read()
            prompt = PromptTemplate.from_template(template=prompt_text)

            agent = create_react_agent(llm, [tool], prompt)
            agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True, handle_parsing_errors=handle_parsing_error)

            response = agent_executor.invoke({"sectionHeading": sectionHeading, "rfpText": rfpText, "refText": refText})
            return response['output']
    except Exception as e:
        raise(e)
    
def summarize_rfp_text(rfpText):
    try:
        with open('prompts/summarize_rfp.txt', 'r') as f:
            prompt_text = f.read()
            prompt = PromptTemplate.from_template(template=prompt_text)
            agent = create_react_agent(llm, [tool], prompt)
            agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True, handle_parsing_errors=handle_parsing_error)

            response = agent_executor.invoke({"rfpText": rfpText})
            return response['output']
    except Exception as e:
        raise(e)
    
# def review_rfp_content(text):
#     try:
#         wordlist = []
#         with open('config/review_wordlist.txt', 'r') as w:
#             word_text = w.readlines()
#             for text in word_text:
#                 wordlist.append(text)

#         # print("\n".join(wordlist))

#         with open('prompts/review_rfp_content.txt', 'r') as f:
#             prompt_text = f.read()
#             prompt = PromptTemplate.from_template(template=prompt_text)
#             agent = create_react_agent(llm, [tool], prompt)
#             agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True, handle_parsing_errors=handle_parsing_error)

#             # response = agent_executor.invoke({"text": text, "wordlist": "\n".join(wordlist)})
#             response = agent_executor.invoke({"text": text})
#             return response['output']
#     except Exception as e:
#         raise(e)
    
def review_rfp_content(content):

    wordlist = []
 
    try:
        with open('config/review_wordlist.txt', 'r') as w:
            word_text = w.readlines()
            for text in word_text:
                target_word_part = text.split("[")[0].strip().strip('"')
                target_words = []
                if '/' in target_word_part:
                    target_words = [''.join(e for e in item.strip() if e.isalnum() or e.isspace()) for item in target_word_part.split('/')]
                else:
                    target_words = [target_word_part.strip()]
                    
                alternate_wordpart = text.split("versus")[1].strip()
                # alternatives = []
                alternate_word_items = alternate_wordpart.split('"')
                alternate_words = [item.strip() for item in alternate_word_items if item.strip() not in [',', 'or', '']]
                
                # if ',' in alternate_words:
                #     alternate_words = alternate_words.split(",")
                #     for item in alternate_words:
                #         if ' or ' in item:
                #             alternate_items = item.split(" or ")
                #             for alternate_item in alternate_items:
                #                 alternate_item = alternate_item.strip('"')
                #                 alternatives.append(alternate_item)
                # elif ' or ' in alternate_words:
                #     alternate_items = alternate_words.split(" or ")
                #     for alternate_item in alternate_items:
                #         alternate_item = alternate_item.strip('"')
                #         alternatives.append(alternate_item)

                # print(alternatives)
                other_words = ", ".join(alternate_words)
                for target_word in target_words:
                    wordlist.append(target_word+": ["+ other_words +"]")

        words = "\n".join(wordlist)
        print(words)

        prompt_template = f"""
            You are an AI model tasked with changing a given text content by replacing avoidable words and phrases with appropriate substitute. 
            You will be provided with a list of avoidable words or phrases with corresponding list of alternative words or phrases in the Context. 

            Replace each "TARGET_WORD" in the text content with an appropriate word from its corresponding list of alternatives.

            The list of avoidable words and substitute has following format:
            TARGET_WORD: [alternative1, alternative2, alternative3, ...]

            List of Avoidable Words and Substitutes:
            {words}

            Text content:
            {content}

        """

        prompt = PromptTemplate.from_template(template=prompt_template).format(text=text)
        # print(prompt)

        response = llm.invoke(prompt).content

        return response

    except Exception as e:
        raise(e)
    
def analyze_rfp_document(rfpText):
    try:
        with open('prompts/analyze_rfp.txt', 'r') as f:
            prompt_text = f.read()
            prompt = PromptTemplate.from_template(template=prompt_text).format(rfpText=rfpText)
            
            response = llm.invoke(prompt)
            return response.content
    except Exception as e:
        raise(e)