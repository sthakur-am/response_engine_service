import os
import string
import logging
from typing import Callable, List, Sequence, Tuple

from langchain_core.agents import AgentAction
from langchain_openai import AzureOpenAI, AzureChatOpenAI #OpenAI, ChatOpenAI,
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import Tool, create_retriever_tool
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import (
    Agent,
    initialize_agent, 
    AgentExecutor, 
    AgentType, 
    create_react_agent, 
    create_tool_calling_agent,
)
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from operator import itemgetter
from langchain_core.messages import trim_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.prompt_values import ChatPromptValue

from azure.core.credentials import AzureKeyCredential
from langchain_core.messages import BaseMessage
from langchain.agents.format_scratchpad.tools import (
    format_to_tool_messages,
)
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
# from langchain.agents import ReActOutputParser

from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper

from indexes import DocumentIndex
from datastore import DocumentBase
from utils import load_rfp_review_wordlist, convert_markdown_to_html

class LLMClient(object):

    def __init__(self, config):
        self.tools = []

        api_wrapper = BingSearchAPIWrapper(
            bing_subscription_key=config['BING_SUBSCRIPTION_KEY'],
            bing_search_url="https://api.bing.microsoft.com/"
        ) 

        self.search_tool = BingSearchResults(
            name="web_search", 
            api_wrapper=api_wrapper
        )

        self.llm = AzureChatOpenAI(
            azure_endpoint="https://caio-openai.openai.azure.com/",
            api_key = config['AZURE_OPENAI_API_KEY'],
            api_version="2024-12-01-preview",
            max_tokens=4096,
            azure_deployment="gpt-4o-mini",
            temperature=0.5,
            top_p=1.0,
            tiktoken_model_name="gpt-4o-mini"
        )

        self.documentBase = config["document_base"]
        self.documentIndex = config["document_index"]

        self.trimmer = trim_messages(
            token_counter=self.llm,
            strategy = "last",
            max_tokens=50,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
        )

        # if 'web_search' in config and config["web_search"] == "true":
        #     self.tools.append(search_tool)

        if 'index' in config:
            self.tools.append(self.create_retriever_tool(config['index']))
        # else:
        #     self.tools.append(self.create_retriever_tool(INDEX_NAME))

    
    def create_retriever_tool(self, index_name):
        # index = DocumentIndex()
        print("Loading index ", index_name)
        retriever = self.documentIndex.get_retriever(index_name)
        
        knowledge_search_tool = create_retriever_tool(retriever, "knowledge_search", "Search through our internal knowledge base")

        return knowledge_search_tool
    
    def create_agent(self, task_name, agent_type="default"):
        prompt_file = f"prompts/{task_name}.txt"
        prompt = self.load_prompt(prompt_file)
        print(agent_type)
        if agent_type == "react":
            # print("Creating react agent")
            agent = create_react_agent(self.llm, self.tools, prompt)
        else:
            # print("Creating default agent")
            agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
                                    #    , max_iterations=10)

        return agent_executor
    
    def load_prompt(self, prompt_file):
        print("Loading prompt file ", prompt_file)
        with open(prompt_file, 'r') as f:
            prompt_text = f.read()
            prompt = PromptTemplate.from_template(template=prompt_text)

        if prompt:
            # print(prompt)
            return prompt
        else:
            raise FileNotFoundError("File "+prompt_file+" not found!")
        
    def run_task(self, task_name, params, mode="default"):
        if mode == "react":
            agent = self.create_agent(task_name, "react")
        else:
            agent = self.create_agent(task_name)
        logging.info(f"Running {task_name} in {mode} mode")
        response = agent.invoke(params)

        return response["output"]
    
    def query_index(self, query):
        prompt_messages = [
            (
                "system",
                "You are a helpful assistant with access to both a comprehensive internal knowledge base and the internet. When responding to a user's query, prioritize information from your internal database using vector search. If no relevant information is found, then perform a web search to provide an answer. Always clearly state whether the information comes from your internal knowledge base or a web search." 
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]

        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        # retreiver_tool = self.create_knowledge_search_tool(docid)
        
        agent = create_tool_calling_agent(llm=self.trimmer|self.llm, tools=self.tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=2)
        
        response = agent_executor.invoke({"input": query})
        
        return response['output']
    
    def summarize_doc(self, docpath):
        docs = self.documentBase.load_documents(docpath)
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=5000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(docs)
        # summarize each chunk
        summary_texts = []
        for text in texts:
            res = self.summarize_text([text], "stuff")
            summary_texts.append(res["output_text"])


        return "\n".join(summary_texts)
    
    def summarize_text(self, text, mode="stuff"):
        prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
              Return your response in paragraphs which covers the key points of the text.
              ```{text}```
              PARAGRAPH SUMMARY:
            """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

        if mode == "map_reduce":
            chain = load_summarize_chain(self.llm, chain_type=mode)
        else:
            chain = load_summarize_chain(self.llm, chain_type=mode, prompt=prompt)
        
        return chain.invoke(text)
    
    def extract_key_sections(self, docs):
        # Map
        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main content sections 
        Helpful Answer:"""

        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # Reduce
        reduce_template = """The following is set of sections of a given document:
        {docs}
        Take these and distill it into a final, consolidated list of the main sections. 
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )

        split_docs = text_splitter.split_documents(docs)
        response = map_reduce_chain.invoke(split_docs)
        return response["output_text"]
    

    def extract_key_themes(self, docs):
        # Map
        map_template = """The following is a set of RFP documents
        {docs}
        Based on this list of docs, please identify the summary of asks and requirements that need to be addressed in the corresponding response document
        Helpful Answer:"""

        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # Reduce
        reduce_template = """The following is set of summary of a given document:
        {docs}
        Take these and distill it into a final, consolidated summary of asks and requirements in the document. 
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )

        split_docs = text_splitter.split_documents(docs)
        response = map_reduce_chain.invoke(split_docs)["output_text"]
        return response
    
    def generate_sections(self, rfpText, sections):
        prompt_template = """Produce a list of section for a response document for responding to an RFP that has following text:
        {text}
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (
            "Your job is to produce a final list of sections\n"
            "We have provided an existing list of sections as a starting point: {sections}\n"
            "We have the opportunity to refine the list of sections"
            "with new context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original list of sections"
            "Rank the list of sections by their relevance to the provided context and return top 15 sections only"
        )
        refine_prompt = PromptTemplate.from_template(refine_template)
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=False,
            input_key="input_documents",
            output_key="output_text",
        )

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(docs)

        # chain = chain({"input_documents": split_docs, "sections": sections}, return_only_outputs=True)
        result = chain.invoke({"input_documents": split_docs, "sections": sections}, return_only_outputs=True)
        print(result)

    def extract_section_content(self, docpath, sectionName):
        documents = self.documentBase.load_documents(docpath)
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_documents(documents)

        prompt_template = """The following is a set of RFP documents
        {docs}
        Based on this list of docs, please extract relevant content for the section 
        {sectionName}
        Content:"""
        prompt = PromptTemplate.from_template(prompt_template).format(docs=docs, sectionName=sectionName)
        response = self.llm.invoke(prompt)

        return response["output"]
    
    def extract_rfp_metadata(self, rfpText):
        logging.info("Extracting metadata from RFP")
        try:
            with open('prompts/analyze_rfp.txt', 'r') as f:
                prompt_text = f.read()
                prompt = PromptTemplate.from_template(template=prompt_text).format(rfpText=rfpText)
                
                response = self.llm.invoke(prompt)
                if response.content:
                    return self.transform_text_to_json(response.content)
                else:
                    return '{}'
        except Exception as e:
            raise(e)
    
    def transform_text_to_json(self, content):
        if '{' in content:
            content = content.split('{')[1]
        else:
            return '{}'
        if '}' in content:
            content = content.split('}')[0]
        else:
            return '{}'
        
        return '{'+content+'}'
    
    def summarize_rfp(self, rfpText):
        logging.info("Getting list of asks from the given RFP")

        prompt_template = """
        system

        You are an AI model tasked with summarizing the asks in a given RFP text. 
        You will be provided with the RFP text to analyze its content for extracting specific asks. 
        Your goal is to create a list of clear, concise, and specific asks that can provide guidance for drafting appropriate response for the RFP. 


        human

        Generate a concise, clear, and specific list of asks for the given RFP text available in the Context.
        The list of asks generated should comprehensively cover the requirements and expectations of the RFP as specified in the RFP text.
        The goal is to create an actionable summary of content creation tasks that can address the requirements and expectations of the RFP.
        Return the list of asks.

        Begin!

        RFP Text: {rfpText}
        """

        prompt = PromptTemplate.from_template(prompt_template).format(rfpText=rfpText)
        response = self.llm.invoke(prompt)

        return response.content
    
    def generate_response_sections(self, rfpText, rfpRequirements):
        logging.info("Running section list generation prompt")
        prompt_template = """

        You are an AI agent designed to analyze and respond to Request for Proposal (RFP) documents. Your task is to generate a list of section headings for a response document based on the given RFP document and a reference list of section headings. Follow these steps:

        1. **Analyze the RFP Document**: Carefully read and understand the content of the provided RFP summary and RFP requirements. Use this summary and requirements to structure your response for this RFP document.

        2. **Review Reference List of Section Headings**: Review the provided reference list of section headings to understand the typical structure and organization of a response document and adapt it for responding to the provided RFP summary and requirements.

        3. **Generate Section Headings**: Based on your analysis of the RFP document and the reference list, generate a list of section headings for the response document. Ensure that the headings are relevant to the RFP requirements and logically organized.

        4. **Provide Descriptions**: For each section heading, write a brief description explaining the purpose and content of that section.

        5. **List RFP Requirements**: For each section heading, list RFP requirements that this section addresses. Use requirement text and not the number.

        6. ""Generate prompt**: Foe each section heading, generate appropriate content generation prompt considering the description and corresponding RFP requirements.

        7. **Return the List of Section Headings**: Return the list of section headings with description, RFP requirements, and prompt**: Combine the list of section headings with corresponding description, RFP requirement texts, and prompt and then return this list.

        Reference List of Section Headings:
        - Cover Letter
        - Executive Summary
        - Company Profile
        - Project Understanding and Approach
        - Scope of Work
        - Key Personnel and Team Structure
        - Relevant Experience and Case Studies
        - Technical Proposal
        - Project Management and Reporting
        - Pricing Proposal
        - Compliance and Certifications
        - Attachments and Appendices

        RFP Summary: {rfpText}
        RFP Requirements: {rfpRequirements}

        Example Output
        Description: description
        RFP Requirements: rfp requirement text
        Prompt: prompt text


        """

        prompt = PromptTemplate.from_template(prompt_template).format(rfpText=rfpText, rfpRequirements=rfpRequirements)
        response = self.llm.invoke(prompt)
        # print(response.content)
        # return response.content
        
        if response.content:
            return self.transform_section_list(response.content)
        else:
            raise Exception("Error generating list of sections!")
    
    def transform_section_list(self, text):
        sections = []
        sections_text = text.split("\n\n")
        for section_text in sections_text:
            section = {}
            for char in string.punctuation:
                section_text = section_text.strip(char)

            if len(section_text.strip()) > 0:
                section_lines = section_text.split("\n")
                title = section_lines[0].strip()
                if '.' in title:
                    title_text = title.split('.')[1].strip()
                    title_index = title.split('.')[0].strip()
                else:
                    title_text = title

                for char in string.punctuation:
                    title_text = title_text.strip(char)
                
                if '.' in title:
                    section["title"] = '.'.join([title_index, title_text.strip()])
                else:
                    section['title'] = title_text

                for section_line in section_lines[1:]:
                    # print(section_line)
                    if "Description"  in section_line:
                        description = section_line.split(":")[1].strip()
                        for char in string.punctuation:
                            description = description.strip(char)
                        section["description"] = description.strip()
                    if "RFP Requirements" in section_line:
                        # print(section_line)
                        requirements = section_line.split(":")[1].strip()
                        for char in string.punctuation:
                            requirements = requirements.strip(char)
                        section["requirements"] = requirements.strip()
                    if "Prompt" in section_line:
                        # print(section_line)
                        prompt = section_line.split(":")[1].strip()
                        for char in string.punctuation:
                            prompt = prompt.strip(char)
                        section["prompt"] = prompt.strip()
                
                sections.append(section)
        logging.info(sections[1:-1])
        return sections[1:-1]
                
                    
    
    def generate_rfp_section_content(self, sectionHeading, sectionRequirements, rfpText):
        if '.' in sectionHeading:
            parts = sectionHeading.split('.')
            if parts[0].isdigit():
                sectionHeading = parts[1]
        if 'Company' in sectionHeading.strip():
            sectionHeading = "Company Overview for Alvarez and Marsal"

        
        prompt_template = """
            system

            You are an AI agent tasked with responding to a given RFP and you are writing content for the section {sectionHeading} of this response document.
            For generating content for this section, use knowledge_search and web_search tools to find content for addressing following requirement:
            {sectionRequirements}
            You must search the internal knowledgebase using knowledge_search tool for preparing content for this section. You may use web_search tool to further enrich this content.
            If you do not find any relevant content, use web_search tool to search the web to find relevant content.
            Your task is to generate a professional quality content by customizing the retrieved content appropriatrely for responding to this RFP. 
            If you are not able to find the relevant content say you couldn't find any relevant content and return.
            if web_search tool is not working, use the output of knowledge_search as the final answer.

            human
            Carefully read and understand the RFP Summary to understand content requirements for the section to prepare content for the section {sectionHeading} of the response document.
            Retrieve the content from the internal indexed knowledgebase using vector search for section {sectionHeading}.
            If the retrieved content is not relevant, use web search tool to find appropriate content.
            Paraphrase this retrieved content ensuring that it is appropriate and applicable for this RFP response document.
            Generate final content for this section of RFP response document using the content prepared.
            If no relevant content is found, say so.

            Section Heading: {sectionHeading}
            Section Requirements: {sectionRequirements}
            RFP Summary: {rfpText}

        """

        prompt = PromptTemplate.from_template(prompt_template).format(sectionHeading=sectionHeading, rfpText=rfpText, sectionRequirements=sectionRequirements)

        tools = []
        tools.extend(self.tools)
        tools.append(self.search_tool)

        self.llm.bind_tools(tools)
        response = self.llm.invoke(prompt)

        return response.content
    
    def condense_prompt(self, prompt: ChatPromptValue) -> ChatPromptValue:
        messages = prompt.to_messages()
        
        out_messages = trim_messages(messages,
            token_counter=self.llm,
            strategy = "last",
            max_tokens=50,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,

        )
        if len(out_messages) > 0:
            messages = out_messages

        return ChatPromptValue(messages=messages)
    
    def review_content(self, content, words):
        prompt_template = """
            system

            You are an AI model tasked with replacing specific words and phrases in the given text content with corresponding substitutes as provided in the context. 
            You will be provided with a list of specific words or phrases to replace with corresponding list of alternative words or phrases in the Context. 

            human

            Find and replace each TARGET_WORD in the text content with an appropriate substitute from the corresponding list of alternatives.
            While identifying the TARGET_WORD in the given text content compare by converting each item to lower case.
            The list of TARGET_WORD with their corresponding list of alternatives has following format:
            TARGET_WORD: [alternative1, alternative2, alternative3, ...]
            Return only the text content

            List of Specific Words or Phrases:
            {words}

            Text content:
            {content}

        """
        prompt = PromptTemplate.from_template(prompt_template).format(words=words, content=content)

        response = self.llm.invoke(prompt)

        return response.content
    
from langchain.callbacks.base import AsyncCallbackHandler
class CustomAsyncCallbackhandler(AsyncCallbackHandler):
    def on_tool_end(self, output, *, run_id, parent_run_id = None, tags = None, **kwargs):
        print(output)
        return super().on_tool_end(output, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)
    def on_chain_end(self, outputs, run_id, parent_run_id, tags):
        print(outputs)
