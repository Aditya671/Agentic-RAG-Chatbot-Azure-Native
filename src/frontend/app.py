import sys
import os
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))
import asyncio
from dotenv import load_dotenv
from pathlib import Path
current_dir = Path(__file__).resolve().parent
dotenv_path = current_dir.parent.parent / ".env"
load_dotenv(dotenv_path)

from datetime import datetime, timezone
from typing import Optional, Dict, Any
import chainlit as cl
import ast
import tempfile
from chainlit.input_widget import Select, Switch, Slider
from chainlit.types import Feedback
from chainlit.user import User
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

from src.backend.UploadFileWrapper import UploadedFileWrapper
from src.backend.utility import generate_blob_sas_url
from src.backend.azure_blob_file_retriever import AzureBlobFileRetriever
from src.backend.cosmos_db_date_layer import CosmosDBDataLayer
from src.backend.ai_models import AIModelTypes
# from src.backend.user_uploaded_file_indexer import UserUploadedFileIndexer
from src.backend.config import config, Environment
from src.backend.credential_manager import CredentialManager
from src.backend.agentic_ai_system import AsyncAgenticAiSystem
from app_logger import setup_logger


# Initialize
logger, log_filename = setup_logger('chainlit_app_logger')
model_enum = AIModelTypes
models_list  = [model.value  for model in model_enum ]

def load_blob_bytes():
    salesforce_index_config = config.indexes.get('salesforce')
    salesforce_credential_manager = CredentialManager(key_vault_url=salesforce_index_config.key_vault.get("url"))
    blob_service = BlobServiceClient.from_connection_string(\
        salesforce_credential_manager.client.get_secret(salesforce_index_config.storage_account.get('connection_string')).value \
    )
    container_client = blob_service.get_container_client(salesforce_index_config.storage_account.get('container_name'))
    azure_blob_agent = AzureBlobFileRetriever(container_client_service=container_client)
    blob_stream = azure_blob_agent.get_latest_file_stream(prefix='your_file')
    blob_bytes = bytes('', encoding='latin1')
    if blob_stream is not None:
        logger.info(f"""[AgenticAiSystem] Downloaded BlobStream: {blob_stream.name}, size={blob_stream.size}")""")
        blob_bytes = blob_stream.to_bytes()
    metadata = azure_blob_agent.get_blob("metadata.json")
    return {'bytes':blob_bytes, 'metadata': metadata.to_str()}
blob_bytes = load_blob_bytes()

def app_default_setting(
    select_index = '',
    select_ai_model = AIModelTypes.GPT51,
    select_response_mode = 'low',
    set_model_top_k = 20,
    set_creativity_level = 0.1,
    enable_coding_assistant=False
):
    user = cl.user_session.get('user')
    indexes = []
    all_tools = [
        Select(
            id="select_index", label="Select KnowledgeBase",
            initial_value=select_index,
            values=indexes,
            description='Select the Knowledge Base'
        ),
        Select(
            id="select_ai_model", label="Choose AI Model",
            initial_value=select_ai_model,
            items={model.name: model.value for model in model_enum},
            description='Choose the AI model for reasoning'
        ),
        Select(
            id="select_response_mode", label="Response Conciseness",
            initial=select_response_mode,
            initial_value=select_response_mode,
            items={'Brief (short, to-the-point answers)': 'low', \
                'Expanded (in-depth explanations)': 'high'},
            description="""Adjust how concise gpt-5's responses should be. """
            ),
        Slider(
            id="set_model_top_k", label="Adjust Top Search Results",
            initial=set_model_top_k, min=0, max=30, step=1,
            description="""Choose how many documents to include in your search. 
            More documents give you broader information but may make the response a bit slower.
            """
        ),
        Slider(
            id="set_creativity_level", label="Tune Creativity Level",
            initial=set_creativity_level, min=0.0, max=1.0, step=0.1,
            description="""Adjusts how creative the AI's answers will be. 
            Lower settings (closer to 0) provide more consistent, factual responses. 
            Higher settings (closer to 1) allow more varied and creative answers, but may be less accurate.
            """
        ),
    ]
    if "Data Science" in user_groups:
        coding_tool = [
            Switch(
                type="switch",
                id="enable_coding_assistant",
                label="Enable Coding Assistant",
                initial=enable_coding_assistant,
                tooltip="Enable or disable the coding assistant",
                description="Toggles the coding assistant's availability for help"
            )
        ]
        all_tools.extend(coding_tool)
    return all_tools

@cl.oauth_callback
async def on_oauth_callback(\
    provider_id: str, token: str, raw_user: Dict[str, str],\
    default_user: User, id_token: Optional[str] = None) -> Optional[User]:
    default_user.metadata["id_token"] = token
    GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0/me"
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{GRAPH_API_ENDPOINT}/memberOf?$select=displayName,mail,id"
    response = requests.get(url, headers=headers)
    group_names = []
    if response.status_code == 200:
        groups = response.json().get("value", [])
        group_names = [{'displayName': group["displayName"], 'id': group['id']} for group in groups if "@odata.type" in group and "group" in group["@odata.type"]]
    default_user.metadata["claims"] = raw_user
    default_user.metadata["groups"] = group_names
    default_user.metadata["tenant"] = raw_user.get("tid")
    default_user.display_name = raw_user.get("displayName")
    return default_user

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Ask me questions about anything....",
            message="Tell me about what you are capabilities are?",
            icon="/public/favicon.png",
        ),
        cl.Starter(
            label='What can you do?',
            message='What can you help me with?',
            icon="/public/favicon.png",
        )
    ]

@cl.on_settings_update
async def on_settings_change(settings):
    logger.info(f"[AgenticAiSystem] App Settings Changed: {str(settings)}")
    await cl.ChatSettings(\
        app_default_setting(\
            select_index=settings['select_index'] or '',\
            select_ai_model=settings['select_ai_model'],\
            select_response_mode=settings['select_response_mode'],\
            set_creativity_level=settings['set_creativity_level'],\
            set_model_top_k=settings['set_model_top_k'], \
            enable_coding_assistant=settings['enable_coding_assistant']
        )\
    ).send()
    return cl.user_session.set('settings', settings)

@cl.data_layer
def get_data_layer():
    """
    Establish CosmosDb-based data layer for persisting chat history.
    """
    select_index = 'aiim'
    environment = os.getenv('ENVIRONMENT', 'local')
    
    credential_manager = CredentialManager(key_vault_url=config.indexes[select_index].key_vault.get("url"))
    if environment == 'local' or environment == Environment.DEVELOPMENT.value:
        url = credential_manager.client.get_secret(\
                config.indexes[select_index].dev_cosmos_db['uri']\
            ).value
        database_id = config.indexes[select_index].dev_cosmos_db['database_id']
        container_id = config.indexes[select_index].dev_cosmos_db['container_id']
    elif environment ==  Environment.UAT.value:
        url = credential_manager.client.get_secret(\
                config.indexes[select_index].uat_cosmos_db['uri']\
            ).value
        database_id = config.indexes[select_index].uat_cosmos_db['database_id']
        container_id = config.indexes[select_index].uat_cosmos_db['container_id']
    elif environment ==  Environment.PRODUCTION.value:
        url = credential_manager.client.get_secret(\
                config.indexes[select_index].prod_cosmos_db['uri']\
            ).value
        database_id = config.indexes[select_index].prod_cosmos_db['database_id']
        container_id = config.indexes[select_index].prod_cosmos_db['container_id']
    else:
        url = credential_manager.client.get_secret(\
                config.indexes[select_index].dev_cosmos_db['uri']\
            ).value
        database_id = config.indexes[select_index].dev_cosmos_db['database_id']
        container_id = config.indexes[select_index].dev_cosmos_db['container_id']
    logger.info(f"[AgenticAiSystem] Data Layer: CosmosDb(URL={url}, Container={container_id}, Database={database_id}, Environment={environment})")
    return CosmosDBDataLayer(
        credential=DefaultAzureCredential(),
        url=url,
        database_id=database_id,
        container_id=container_id
    )

@cl.on_chat_start
async def start():
    try:
        cl.user_session.set('chat_history', [])
        user = cl.user_session.get('user')
        if 'groups' not in user.metadata and user is not None:
            GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0/me"
            headers = {"Authorization": f"Bearer {user.metadata['id_token']}"}
            url = f"{GRAPH_API_ENDPOINT}/memberOf?$select=displayName,mail,id"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                groups = response.json().get("value", [])
                group_names = [{'displayName': group["displayName"], 'id': group['id']} for group in groups if "@odata.type" in group and "group" in group["@odata.type"]]
                user.metadata['groups'] = group_names
                cl.user_session.set('user', user)
        settings = cl.user_session.get('settings')
        if settings is None:
            settings = await cl.ChatSettings(app_default_setting()).send()
            cl.user_session.set('settings', settings)
        temp_upload_dir = tempfile.mkdtemp(prefix="llama_index_")
        agent = AsyncAgenticAiSystem(\
                selected_model = model_enum(settings['select_ai_model']) or AIModelTypes.GPT51,\
                similarity_top_k=settings['set_model_top_k'],\
                reasoning_effect=settings['select_response_mode'],\
                llm_creativity_level=settings['set_creativity_level'],\
                index_name=settings['select_index'],\
                session_id=cl.user_session.get('id'),\
                upload_root_dir=temp_upload_dir,\
                # upload_root_dir=f'user_uploads\\{user.identifier}',\
                conversation_thread = [], \
                blob_bytes = blob_bytes,
                enable_coding_assistant=settings['enable_coding_assistant']
            )
        logger.info(f"[AgenticAiSystem] LoggedIn user {user.identifier or 'local_user'}, \n"
            f"Initiating new Thread: {cl.user_session.get('id')} \n"
            f"ON DateTime {datetime.now().isoformat()} \n"
            f"With Thread Settings {str(settings)} \n"
            f"Temp Root Directory on ({temp_upload_dir}) \n"
        )
        cl.user_session.set('agentic_engine', agent)
        return True
    except Exception as e:
        logger.exception(e)


@cl.on_chat_resume
async def on_chat_resume(thread):
    app_chat_settings = cl.user_session.get('settings')
    user = cl.user_session.get('user')
    if 'settings' in thread:
        user_settings = thread['settings']
        settings = await cl.ChatSettings(
            app_default_setting(\
                select_index=user_settings['select_index'] or '',\
                select_ai_model=user_settings['select_ai_model'],\
                select_response_mode=user_settings['select_response_mode'],\
                set_creativity_level=user_settings['set_creativity_level'],\
                set_model_top_k=user_settings['set_model_top_k'], \
            enable_coding_assistant=settings['enable_coding_assistant']
            )
            ).send()        
    elif app_chat_settings is None:
        settings = await cl.ChatSettings(app_default_setting()).send()
    else:
        settings = await cl.ChatSettings(\
            app_default_setting(\
                select_index=app_chat_settings['select_index']  or '',\
                select_ai_model=app_chat_settings['select_ai_model'],\
                select_response_mode=app_chat_settings['select_response_mode'],\
                set_creativity_level=app_chat_settings['set_creativity_level'],\
                set_model_top_k=app_chat_settings['set_model_top_k'], \
                enable_coding_assistant=app_chat_settings['enable_coding_assistant']
            )
            ).send()

    cl.user_session.set('settings', settings)
    credential_manager = CredentialManager(key_vault_url=config.indexes[settings['select_index']].key_vault.get("url"))
    
    if 'elements' in thread:
        for element in thread['elements']:
            if ('url' in element and element['url'] is not None) and element['type'] == 'pdf':
                element['url'] = generate_blob_sas_url( \
                        account_name=config.indexes[settings['select_index']].storage_account["storage_account_name"], \
                        account_key=credential_manager.client.get_secret(\
                            config.indexes[settings['select_index']].storage_account['account_key']\
                        ).value, \
                        container_name=config.indexes[settings['select_index']].storage_account["container_name"], \
                        blob_name=element["name"], \
                    )
            if ('url' in element and element['url'] is None) and element['type'] == 'pdf':
                    element['path'] = str(element['name'])
    
    if cl.user_session.get('agentic_engine') is None:
        agent = AsyncAgenticAiSystem(\
                selected_model = model_enum(settings['select_ai_model']) or AIModelTypes.GPT51,\
                similarity_top_k=settings['set_model_top_k'],\
                llm_creativity_level=settings['set_creativity_level'],\
                reasoning_effect=settings['select_response_mode'],\
                index_name=settings['select_index'],\
                session_id=cl.user_session.get('id'),\
                upload_root_dir=tempfile.mkdtemp(prefix="llama_index_") ,
                conversation_thread=cl.user_session.get('chat_history'), \
                blob_bytes = blob_bytes,
                enable_coding_assistant=settings['enable_coding_assistant']
            )
        cl.user_session.set('agentic_engine', agent)
    else:
        agentic_engine = cl.user_session.get('agentic_engine')
        agentic_engine.set_conversation_thread(thread=cl.user_session.get('chat_history'))
        agentic_engine.set_selected_model(selected_model=app_chat_settings['select_ai_model'])
        agentic_engine.set_llm_creativity_level(llm_creativity_level=app_chat_settings['set_creativity_level'])
        agentic_engine.set_reasoning_effect(reasoning_effect=app_chat_settings['select_response_mode'])
        agentic_engine.set_similarity_top_k(similarity_top_k=app_chat_settings['set_model_top_k'])
        agentic_engine.set_index_name(index_name=app_chat_settings['select_index'])
        agentic_engine.set_coding_assistant(enable_coding_assistant=app_chat_settings['enable_coding_assistant'])

                

@cl.on_feedback
async def on_feedback(feedback: Feedback):
    chat_history = cl.user_session.get('chat_history')
    if chat_history is None:
        cl.user_session.set('chat_history', [])
        chat_history = []
    for step in chat_history:
        if step.get("stepId") == feedback.forId:
            step["feedbackScore"] = feedback.value
            step["feedbackComment"] = feedback.comment
    return cl.user_session.set('chat_history', chat_history)

@cl.on_message
async def on_message(message: cl.Message):
    try:
        chat_history = cl.user_session.get('chat_history')
        if chat_history is None:
            cl.user_session.set('chat_history', [])
        is_existing_message = message.id
        if is_existing_message in [c['stepId'] for c in chat_history]:
            reference_msg = next(
                (m for m in chat_history if m["stepId"] == is_existing_message),
                None
            )
            reference_time = datetime.fromisoformat(reference_msg["createdAt"])
            # Filter messages created after reference
            chat_history[:] = [
                m for m in chat_history
                if datetime.fromisoformat(m["createdAt"]) < reference_time
            ]
            chat_history[:] = [ m for m in chat_history if m["stepId"] != is_existing_message ]
            cl.user_session.set('chat_history', chat_history)

        settings = cl.user_session.get('settings')
        if settings is None:
            settings = await cl.ChatSettings(app_default_setting()).send()
        else:
            settings = await cl.ChatSettings(\
                    app_default_setting(\
                        select_index=settings['select_index'] or '',\
                        select_ai_model=settings['select_ai_model'],\
                        select_response_mode=settings['select_response_mode'],\
                        set_creativity_level=settings['set_creativity_level'],\
                        set_model_top_k=settings['set_model_top_k'], \
                        enable_coding_assistant=settings['enable_coding_assistant']
                    )\
                ).send()
        cl.user_session.set('settings', settings)
        credential_manager = CredentialManager(key_vault_url=config.indexes[settings['select_index']].key_vault.get("url"))
        user = cl.user_session.get('user')        
        agentic_engine = cl.user_session.get('agentic_engine')
        if cl.user_session.get('agentic_engine') is None:
            agentic_engine = AsyncAgenticAiSystem(\
                selected_model = model_enum(settings['select_ai_model']) or AIModelTypes.GPT51,\
                similarity_top_k=settings['set_model_top_k'],\
                reasoning_effect=settings['select_response_mode'],\
                llm_creativity_level=settings['set_creativity_level'],\
                index_name=settings['select_index'],\
                session_id=cl.user_session.get('id'),\
                upload_root_dir=tempfile.mkdtemp(prefix="llama_index_"),\
                # upload_root_dir=f'user_uploads\\{user.identifier}',\
                conversation_thread=chat_history, \
                blob_bytes = blob_bytes,
                enable_coding_assistant=settings['enable_coding_assistant']
            )
            cl.user_session.set('agentic_engine', agentic_engine)
        else:
            agentic_engine.set_conversation_thread(thread=chat_history)
            agentic_engine.set_reasoning_effect(reasoning_effect=settings['select_response_mode'])
            agentic_engine.set_selected_model(selected_model=settings['select_ai_model'])
            agentic_engine.set_llm_creativity_level(llm_creativity_level=settings['set_creativity_level'])
            agentic_engine.set_similarity_top_k(similarity_top_k=settings['set_model_top_k'])
            agentic_engine.set_index_name(index_name=settings['select_index'])
            agentic_engine.set_coding_assistant(enable_coding_assistant=settings['enable_coding_assistant'])
        final_answer = cl.Message(content='')
        # await final_answer.stream_token('Processing the Query...')
        async def animate_status():
            dots = "."
            while True:
                final_answer.content = f"Processing the Query{dots}"
                await final_answer.update()
                await asyncio.sleep(0.35)  # animation speed
                dots = "." if len(dots) == 8 else dots + "."

        animation_task = asyncio.create_task(animate_status())
        
        ui_response = await final_answer.send()
        try:
            user_prompt = message.content.strip()
            chat_history.append({'stepId': message.id,
                                 'parentId': message.parent_id,
                                 'role':'user', 'content': user_prompt,
                                 'createdAt': final_answer.created_at or datetime.now().replace(tzinfo=timezone.utc),
                                 'feedbackScore':None,
                                 'feedbackComment':''
                                 })
            if not user_prompt:
                animation_task.cancel()
                # try:
                #     await animation_task
                # except asyncio.CancelledError:
                #     pass
                await cl.Message(content="⚠️ Prompt is empty. Please type something.").send()
                return
            uploaded_files_summary = {}
            uploaded_files = message.elements
            if isinstance(uploaded_files, list) and len(uploaded_files) > 0:
                try:
                    file_wrappers = [UploadedFileWrapper(f.path, f.name) for f in uploaded_files]
                    file_content = f'The user has uploaded the following files {", ".join([f.name for f in uploaded_files])}, assist them in their queries if related to the uploaded files'
                    uploaded_files_summary = await agentic_engine.upload_and_index_files(file_wrappers)
                    for user_file in uploaded_files:
                        user_file_element = await cl.Message(content="Files uploaded and indexed successfully!").send()
                        file_summary = uploaded_files_summary.get(user_file.name, "No summary")
                        await cl.Text(\
                            name=user_file.name,\
                            content=file_summary,\
                        ).send(for_id=user_file_element.id)
                    chat_history.append({'role':'system', 'content': file_content})
                except Exception as e:
                    await cl.Message(content=f"Error uploading files: {str(e)}").send()
                    chat_history.append({
                        'stepId': ui_response.id,
                        'parentId': message.id,
                        'role':'assistant', 'content': f"Error uploading files: {str(e)}",
                        'createdAt': ui_response.created_at or datetime.now().replace(tzinfo=timezone.utc),
                        'feedbackScore':None,
                        'feedbackComment':''
                        })
                    return cl.user_session.set('chat_history', chat_history)
            
            ans = await agentic_engine.run_agent_async(user_prompt)
            # Stop animation before streaming the real answer
            animation_task.cancel()
            # try:
            #     await animation_task
            # except asyncio.CancelledError:
            #     pass
            await stream_answer_and_citations(ui_response, ans.response.content, credential_manager, settings)
            chat_history.append({
                'stepId': ui_response.id,
                'parentId': message.id,
                'role':'assistant', 'content': ans.response.content,
                'createdAt': ui_response.created_at or datetime.now().replace(tzinfo=timezone.utc),
                'feedbackScore':None,
                'feedbackComment':''
                })
            if len(chat_history) == 2:
                await get_data_layer().update_thread(thread_id=message.thread_id, name=agentic_engine.generate_thread_title())
            logger.info(f"[AgenticAiSystem] LoggedIn user {user.identifier or 'local_user'}, \n"
                f"Resuming Thread: {message.thread_id} \n"
                f"ON DateTime {datetime.now().isoformat()} \n"
                f"With Thread Settings {str(settings)} \n"
            )
            return cl.user_session.set('chat_history', chat_history)

        except Exception as e:
            await cl.Message(content=f"❌ Error: {str(e)}").send()
            chat_history.append({
                'stepId': ui_response.parent_id,
                'parentId': message.id,
                'role':'assistant', 'content': f"❌ Error: {str(e)}",
                'createdAt': ui_response.created_at or datetime.now().replace(tzinfo=timezone.utc),
                'feedbackScore':None,
                'feedbackComment':''
                })
            return cl.user_session.set('chat_history', chat_history)
    except Exception as e:
        logger.exception(e)
        await cl.Message(content=f"InternalServerError: {str(e)}").send()
        chat_history.append({
                'stepId': ui_response.parent_id,
                'parentId': message.id,
                'role':'assistant', 'content': f"InternalServerError: {str(e)}",
                'createdAt': ui_response.created_at or datetime.now().replace(tzinfo=timezone.utc),
                'feedbackScore':None,
                'feedbackComment':''
                })
        return cl.user_session.set('chat_history', chat_history)

async def stream_answer_and_citations(\
    target_msg_element: cl.Message,\
    response_content: str,\
    credential_manager: CredentialManager,\
    thread_settings: Dict[str, Any]\
):
    # Set Content to "" from "Processing the Query..." content, displayed on UI - Code Start
    target_msg_element.content = ""
    await target_msg_element.update()
    # Set Content to "" from "Processing the Query..." content, displayed on UI - Code End
    # Check if Citations are avaialable in Text Content - Code Start
    CITATION_KEY = "Citations:"
    citation_index = response_content.find(CITATION_KEY)
    if citation_index != -1:
        ui_main_response_content = response_content[:citation_index].strip()
        for stream_ui_content_as_token in ui_main_response_content.split(" "):
            await target_msg_element.stream_token(stream_ui_content_as_token + " ")
        await target_msg_element.update()
        # IF Citations are available Prepare Source List for Elements Display - Code Start
        source_list = []
        try:
            start_index = response_content.index(CITATION_KEY) + len(CITATION_KEY)
            remaining_text = response_content[start_index:].strip()
            citation_list_str = remaining_text[:remaining_text.index(']') + 1]
            source_list = ast.literal_eval(citation_list_str)
        except (ValueError, SyntaxError):
            source_list = []

        # IF source_list is not empty, Prepare Elements for Rendering - Code Start
        elements_markdown_html_for_UI = []
        if source_list:
            for list_index, source in enumerate(source_list):
                if 'mimetype' in source and source['mimetype'] == 'pdf':
                    url = generate_blob_sas_url(  
                        account_name=config.indexes[thread_settings['select_index']].storage_account["storage_account_name"],  
                        account_key=credential_manager.client.get_secret(\
                            config.indexes[thread_settings['select_index']].storage_account['account_key']\
                        ).value,  
                        container_name=config.indexes[thread_settings['select_index']].storage_account["container_name"],  
                        blob_name=source["source_node"],  
                    )
                    await cl.Pdf(
                            name=source["source_node"],
                            size="small",
                            url=url if 'user_uploads' not in source['source_node'] else None,
                            path=source['source_node'] if 'user_uploads' in source['source_node'] else None,
                            display="side",
                            page=int(source["page_number"])
                        ).send(for_id=target_msg_element.id)
                    file_name = str(source["title"])
                    file_name = file_name.split("/")[-1].lower().replace('.pdf', '').title()
                    elements_markdown_html_for_UI.append(\
                        f"""&emsp;**Reference [{list_index + 1}]**:&NewLine;"""\
                        f"&emsp;&emsp;**Source**: {file_name} (**PageNumber**: {source['page_number']})&NewLine;" \
                        f"&emsp;&emsp;**SourcePath**: {source['source_node']} &NewLine;" \
                    )
                elif 'mimetype' in source and source['mimetype'] == 'url':
                    elements_markdown_html_for_UI.append(f"""&emsp;**Reference [{list_index + 1}]**:&NewLine;"""\
                                f"""&emsp;&emsp;**Source**: [{source['title']}]({source['source_node']})&NewLine;""")

            if len(elements_markdown_html_for_UI) > 0:
                elements_as_ui_content = " &NewLine;&NewLine; **Citations (References)**:&NewLine;" + "".join(elements_markdown_html_for_UI)
                for stream_ui_content_as_token in elements_as_ui_content.split():
                    await target_msg_element.stream_token(stream_ui_content_as_token + " ")
            await target_msg_element.update()
    else:
        for stream_ui_content_as_token in response_content.split(" "):
            await target_msg_element.stream_token(stream_ui_content_as_token + " ")
        await target_msg_element.update()
    # Set Content to "" from "Processing the Query..." content, displayed on UI - Code End