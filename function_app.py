import azure.functions as func
import logging
import json
import os
from datetime import datetime

# from gpt import (
#     add_documents_to_index,
#     get_rfp_response_structure,
#     get_rfp_response_content, 
#     summarize_rfp_text, 
#     review_rfp_content, 
#     query_rfp_knowledgebase,
#     analyze_rfp_document
# )

from databases import Tasklist
from exceptions import GPTException
from datastore import DocumentBase
from indexes import DocumentIndex
from llm import LLMClient
from utils import load_rfp_review_wordlist
from security import get_user_token

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

documentBase = DocumentBase({
    "STORAGE_ACCESS_KEY": str(os.environ["STORAGE_ACCESS_KEY"]),
    "STORAGE_CONNECTION_STR": str(os.environ["STORAGE_CONNECTION_STR"])
})

index = DocumentIndex({
    "AZURE_OPENAI_EMBEDDING_API_KEY": str(os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"]),
    "AZURE_SEARCH_API_KEY": str(os.environ["AZURE_SEARCH_API_KEY"]),
    "document_base": documentBase
    })

@app.route(route="call_gpt")
def chat_with_rfpengine(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('call_gpt service processing new query.')
    docId = req.params.get('doc_id')
    
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    query = req.params.get('query')
    if not query:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            query = req_body.get('query')

    if docId and query:
        client = LLMClient({
            "index": docId,
            "web_search": "true", 
            "BING_SUBSCRIPTION_KEY": str(os.environ["BING_SUBSCRIPTION_KEY"]), 
            "AZURE_OPENAI_API_KEY": str(os.environ["AZURE_OPENAI_API_KEY"]),
            "document_base": documentBase,
            "document_index": index
        })
        response = {}
        try:
            res = client.query_index(query)
            response['status'] = "success"
            response['data'] = res

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)
            
        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a query in the request body for a response from the service.",
             status_code=200
        )

@app.route(route="summarize_document", auth_level=func.AuthLevel.FUNCTION)
def summarize_document(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a summarize document request.')

    docId = req.params.get('doc_id')
    docPath = req.params.get('doc_path')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    if not docPath:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docPath = req_body.get('doc_path')

    if docId:
        response = {}
        
        try:
            client = LLMClient({
                "index": docId,
                "web_search": "true", 
                "BING_SUBSCRIPTION_KEY": str(os.environ["BING_SUBSCRIPTION_KEY"]), 
                "AZURE_OPENAI_API_KEY": str(os.environ["AZURE_OPENAI_API_KEY"]),
                "document_base": documentBase,
                "document_index": index
            })
            
            tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})

            logging.info(f"Summarizing document {docPath}")
            summary = client.summarize_doc(docId+"/"+docPath)

            logging.info(f"Updating document summary for {os.path.basename(docPath)}")
            tasklist.update_doc_summary(docId, os.path.basename(docPath), summary)
            
            response['status'] = "success"
            response['data'] = os.path.basename(docPath)
            
        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)
            logging.info(str(ex))
        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass document path in the request body for analyzing RFP.",
             status_code=200
        )

@app.route(route="analyze_rfp", auth_level=func.AuthLevel.FUNCTION)
def analyze_rfp(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a analyze RFP request.')

    docId = req.params.get('doc_id')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    if docId:
        response = {}
        client = LLMClient({
            "index": docId,
            "web_search": "true", 
            "BING_SUBSCRIPTION_KEY": str(os.environ["BING_SUBSCRIPTION_KEY"]), 
            "AZURE_OPENAI_API_KEY": str(os.environ["AZURE_OPENAI_API_KEY"]),
            "document_base": documentBase,
            "document_index": index
        })
        rfpText = None
        try:
            tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
            docs = tasklist.get_docs(docId)
            rfpFilename = [doc["doc_path"] for doc in docs if doc["doc_type"] == "rfp"][0]
            logging.info(f"Loading RFP text from {rfpFilename}")
            # documentBase = DocumentBase({
            #     "STORAGE_ACCESS_KEY": str(os.environ["STORAGE_ACCESS_KEY"]), 
            #     "STORAGE_CONNECTION_STR": str(os.environ["STORAGE_CONNECTION_STR"])
            # })
            rfpText = documentBase.extract_text_from_blob(docId+"/rfp/"+rfpFilename)
            
            if rfpText:
                logging.info("Extracting metadata from the doc")
                res = client.extract_rfp_metadata(rfpText)
                response['status'] = "success"
                response['data'] = res
            else:
                raise Exception("No RFP file found!")

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)
            logging.info(str(ex))
        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass RFP text in the request body for analyzing RFP.",
             status_code=200
        )

@app.route(route="summarize_rfp", auth_level=func.AuthLevel.FUNCTION)
def summarize_rfp(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a summarize RFP request.')

    docId = req.params.get('doc_id')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    if docId:
        response = {}
        tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
        client = LLMClient({
            "index": docId, 
            "BING_SUBSCRIPTION_KEY": str(os.environ["BING_SUBSCRIPTION_KEY"]), 
            "AZURE_OPENAI_API_KEY": str(os.environ["AZURE_OPENAI_API_KEY"]),
            "document_base": documentBase,
            "document_index": index
        })

        try:
            docs = tasklist.get_docs(docId)
            rfpFilename = [doc["doc_path"] for doc in docs if doc["doc_type"] == "rfp"][0]
            rfpText = tasklist.get_doc_summary(docId, rfpFilename)

            res = client.summarize_rfp(rfpText)
            response['status'] = "success"
            response['data'] = res

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)
            logging.info(ex)
        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass RFP text in the request body for a response TOC.",
             status_code=200
        )


@app.route(route="generate_rfp_structure", auth_level=func.AuthLevel.FUNCTION)
def generate_rfp_structure(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    docId = req.params.get('doc_id')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    if docId:
        response = {}
        
        client = LLMClient({
            "index": docId,
            "web_search": "true", 
            "BING_SUBSCRIPTION_KEY": str(os.environ["BING_SUBSCRIPTION_KEY"]), 
            "AZURE_OPENAI_API_KEY": str(os.environ["AZURE_OPENAI_API_KEY"]),
            "document_base": documentBase,
            "document_index": index
        })
        
        tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
        try:
            docs = tasklist.get_docs(docId)
            rfpFilename = [doc["doc_path"] for doc in docs if doc["doc_type"] == "rfp"][0]
            
            rfpText = tasklist.get_doc_summary(docId, rfpFilename)
            rfpRequirements = client.run_task("summarize_rfp", {"rfpText": rfpText}, "react")

            sections = client.generate_response_sections(rfpText, rfpRequirements)
            
            logging.info(sections)
            response['status'] = "success"
            response['data'] = sections

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass RFP text in the request body for a response TOC.",
             status_code=200
        )
    
@app.route(route="generate_response_sections", auth_level=func.AuthLevel.FUNCTION)
def generate_response_sections(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    docId = req.params.get('doc_id')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    if docId:
        response = {}

        client = LLMClient({
            "index": docId,
            "web_search": "true", 
            "BING_SUBSCRIPTION_KEY": str(os.environ["BING_SUBSCRIPTION_KEY"]), 
            "AZURE_OPENAI_API_KEY": str(os.environ["AZURE_OPENAI_API_KEY"]),
            "document_base": documentBase,
            "document_index": index
        })
        # tasklist = Tasklist({})
        tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
        try:
            docs = tasklist.get_docs(docId)
            rfpFilename = [doc["doc_path"] for doc in docs if doc["doc_type"] == "rfp"][0]
            
            rfpText = tasklist.get_doc_summary(docId, rfpFilename)
            
            rfpRequirements = client.summarize_rfp(rfpText)
            sections = client.generate_response_sections(rfpText, rfpRequirements)
            
            logging.info(sections)
            res = sections
            response['status'] = "success"
            response['data'] = res

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass RFP text in the request body for a response TOC.",
             status_code=200
        )


@app.route(route="generate_content", auth_level=func.AuthLevel.FUNCTION)
def generate_content(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    sectionTitle = req.params.get('section_title')
    if not sectionTitle:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            sectionTitle = req_body.get('section_title')

    sectionRequirements = req.params.get('section_requirements')
    if not sectionRequirements:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            sectionRequirements = req_body.get('section_requirements')

    docId = req.params.get('doc_id')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    if docId:
        tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
        docs = tasklist.get_docs(docId)

        rfpFilename = [doc["doc_path"] for doc in docs if doc["doc_type"] == "rfp"][0]
        rfpText = tasklist.get_doc_summary(docId, rfpFilename) 

    if sectionTitle and rfpText:
        response = {}
        
        client = LLMClient({
            "index": docId,
            "web_search": "true", 
            "BING_SUBSCRIPTION_KEY": str(os.environ["BING_SUBSCRIPTION_KEY"]), 
            "AZURE_OPENAI_API_KEY": str(os.environ["AZURE_OPENAI_API_KEY"]),
            "document_base": documentBase,
            "document_index": index
        })
        
        try:
            res = client.generate_rfp_section_content(sectionTitle, sectionRequirements, rfpText)
            response['status'] = "success"
            response['data'] = res

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a section heading in the request body for a personalized response.",
             status_code=200
        )

@app.route(route="review_content", auth_level=func.AuthLevel.FUNCTION)
def review_content(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    docId = req.params.get('doc_id')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    content = req.params.get('content')
    if not content:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            content = req_body.get('content')

    if docId and content:
        response = {}
        
        client = LLMClient({
            "index": docId,
            "web_search": "true", 
            "BING_SUBSCRIPTION_KEY": str(os.environ["BING_SUBSCRIPTION_KEY"]), 
            "AZURE_OPENAI_API_KEY": str(os.environ["AZURE_OPENAI_API_KEY"]),
            "document_base": documentBase,
            "document_index": index
        })
        try:
            words = load_rfp_review_wordlist()
            
            res = client.review_content(content, words)
            response['status'] = "success"
            response['data'] = res

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass content in the request body for a personalized response.",
             status_code=200
        )
    

@app.route(route="generate_tasklist", auth_level=func.AuthLevel.FUNCTION)
def generate_tasklist(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a create_tasklist request.')
   
    docId = req.form.get('doc_id')
    
    if docId:
        response = {}
        try:
            tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
            status = tasklist.get_document_draft_status(docId)
            if not status:
                tasklist.add_document_drafts(docId);
            
            res = tasklist.get_tasks(docId)

            response['status'] = "success"
            response['data'] = res
            
        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a document id in the request body for a document tasks status.",
             status_code=200
        )

@app.route(route="update_taskstatus", auth_level=func.AuthLevel.FUNCTION)
def update_taskstatus(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a update_taskstatus request.')
    
    req_body = req.get_json()
    docId = req_body.get('doc_id')
    taskStatus = req_body.get("task_status")
    logging.info(docId)
    
    if docId:
        
        response = {}
        try:
            tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
            res = tasklist.update_tasks(docId, taskStatus)

            response['status'] = "success"
            response['data'] = res
            
        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a document id in the request body for updating document tasks status.",
             status_code=200
        )

@app.route(route="upload_ref_files", auth_level=func.AuthLevel.FUNCTION)
def upload_ref_files(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a upload_ref_files request.')
    
    docId = req.form.get('doc_id')
    tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})

    logging.info(f"Uploading files for RFP draft # {docId}")
    if docId:
        response = {}
        docs = []
        try:
            ref_files_count = 0
            for key in req.files:
                file_type = key.split(":")[0]
                if file_type == "ref":
                    ref_files_count += 1
                file_item = req.files[key]
                
                logging.info(f"Uploading {file_type} file {file_item.filename}")
                # documentBase = DocumentBase({
                #     "STORAGE_ACCESS_KEY": str(os.environ["STORAGE_ACCESS_KEY"]), 
                #     "STORAGE_CONNECTION_STR": str(os.environ["STORAGE_CONNECTION_STR"])
                # })
                documentBase.upload_file(docId+"/"+file_type+"/"+file_item.filename, file_item.read())
                
                tasklist.add_docs(docId, [{"filepath": file_item.filename, "type": file_type}])
                docs.append({'doc_path': file_item.filename, 'doc_type': file_type})

            if ref_files_count > 0:
                index.add_documents_to_index(docId, docId+"/ref")

            logging.info(f"Total {len(docs)} documents processed.")
            response['status'] = "success"
            response['data'] = docs
           
        except Exception as ex:
            logging.info(ex)
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a document id in the request body for a document tasks status.",
             status_code=200
        )

@app.route(route="list_documents", auth_level=func.AuthLevel.FUNCTION)    
def list_documents(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a list_documents request.')
    
    req_body = req.get_json()
    docId = req_body.get('doc_id')
    logging.info(docId)

    if docId:
        
        response = {}
        try:
            logging.info(str(os.environ['SqlConnectionString']))
            tasklist = Tasklist({'connection_str': str(os.environ['SqlConnectionString'])})
            res = tasklist.get_docs(docId)

            response['status'] = "success"
            response['data'] = res
           
        except Exception as ex:
            logging.exception(ex)
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a document id in the request body for getting list of documents.",
             status_code=200
        )
    
@app.route(route="load_document_template", auth_level=func.AuthLevel.FUNCTION)    
def load_document_template(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a load_document_template request.')
    
    req_body = req.get_json()
    docId = req_body.get('doc_id')
    logging.info(docId)

    if docId:
        
        response = {}
        try:
            tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
            docs = tasklist.get_docs(docId)
            templateFilename = [doc["doc_path"] for doc in docs if doc["doc_type"] == "sample"][0]
            logging.info("Loading template file "+templateFilename)
            # documentBase = DocumentBase({
            #     "STORAGE_ACCESS_KEY": str(os.environ["STORAGE_ACCESS_KEY"]), 
            #     "STORAGE_CONNECTION_STR": str(os.environ["STORAGE_CONNECTION_STR"])
            # })
            text = documentBase.read_blob_as_base64(docId+"/sample/"+templateFilename)

            response['status'] = "success"
            response['data'] = text.decode('utf-8')
           
        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a document id in the request body for getting list of documents.",
             status_code=200
        )
    
@app.route(route="request_user_access_token", auth_level=func.AuthLevel.FUNCTION)    
def request_user_access_token(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a list_documents request.')
      
    response = {}
    try:
        access_token = get_user_token()
        response['status'] = "success"
        response['data'] = access_token
    except Exception as ex:
        response['status'] = "error"
        response['data'] = str(ex)

    return func.HttpResponse(json.dumps(response))

@app.route(route="complete_drafting", auth_level=func.AuthLevel.FUNCTION)    
def complete_drafting(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a complete_drafting request.')
      
    response = {}
    docId = req.params.get('doc_id')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    if docId:
        try:
            # delete entries in the database
            tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
            tasklist.delete_document_tasks(docId)
            tasklist.delete_document_resources(docId)

            # delete blob resources
            # documentBase = DocumentBase({
            #     "STORAGE_ACCESS_KEY": str(os.environ["STORAGE_ACCESS_KEY"]), 
            #     "STORAGE_CONNECTION_STR": str(os.environ["STORAGE_CONNECTION_STR"])
            # })
            documentBase.delete_blob_path(docId)

            # delete indexes
            # index = DocumentIndex()
            index.delete_index(docId)

            # update drafting status
            tasklist.complete_document_draft(docId)

            response['status'] = "success"
            response['data'] = docId

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a document id in the request body for completing the document drafting process.",
             status_code=200
        )
    
@app.route(route="get_drafting_status", auth_level=func.AuthLevel.FUNCTION)    
def get_drafting_status(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a get_drafting_status request.')
      
    response = {}
    docId = req.params.get('doc_id')
    if not docId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            docId = req_body.get('doc_id')

    if docId:
        try:
            # get document draft status
            tasklist = Tasklist({"connection_str": str(os.environ["SqlConnectionString"])})
            status = tasklist.get_document_draft_status(docId)
            
            if status['start_date']:
                status['start_date'] = status['start_date'].strftime('%B %d, %Y')
            else:
                status['start_date'] = ''

            if status['end_date']:
                status['end_date'] = status['end_date'].strftime('%B %d, %Y')
            else:
                status['end_date'] = ''

            logging.info(status)
            json.dumps(status)
            response['status'] = "success"
            response['data'] = status

        except Exception as ex:
            response['status'] = "error"
            response['data'] = str(ex)

        return func.HttpResponse(json.dumps(response))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a document id in the request body for getting the status of the document drafting process.",
             status_code=200
        )