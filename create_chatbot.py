import logging as l
import langchain_chroma as lc
import langchain_ollama as lo
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

l.basicConfig(filename='rough.log', filemode='w', format='%(levelname)s - %(message)s', level=l.INFO)

EMBEDDING_MODEL_NAME = 'nomic-embed-text'
DB_DIRECTORY = './asu_rc'
COLLECTION_NAME = 'ASU-Research-Computing'
LLM_NAME = 'gemma3'
TEMPERATURE = 0.1
TOP_K = 3
SCORE_THRESHOLD = 0.7
SYSTEM_PROMPT = "You are a helpful assistant for ASU Research Computing."

embedding_model = lo.OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
vector_store = lc.Chroma(embedding_function=embedding_model, collection_name=COLLECTION_NAME, persist_directory=DB_DIRECTORY)
llm = lo.ChatOllama(model=LLM_NAME, temperature=TEMPERATURE)
conversation_history = ChatMessageHistory()

def classify_query(query: str) -> str:
    query_nature_classifier_prompt = """Classify the user query into one of the following categories:
1. Greetings - General greetings. For eg: "hi", "hello", "how are you doing?", "what's up?", etc.
2. Relevant Question - Something relevant to ASU's supercomputers, HPC clusters or research computing including administrative and business policies. For eg: "how do I list available modules on Sol?", "what supercomputers does ASU have?", "can you list the days I can schedule an appointment with ASU research computing?", etc.
3. Previous Message Clarification - Requiring more information on some previous answer/response by the AI assistant. For eg: "can you elaborate on this?", "can I get more information regarding this?", "what do you mean?", "I did not understand that.", etc.
4. Action Requests - Asking about the live status of the ASU supercomputers, performing some action or running some command. For eg: "is the Agave cluster down right now?", "cancel my job <Job ID>.", "can you delete the following files/folders?", etc
5. Irrelevant Question - Irrelevant questions that don't concern computers, supercomputing, HPC clusters. For eg: "who let the dogs out?", "is there a God?", "what is the capital of USA?", "who is ASU's president?", etc.
6. Other - Anything else that doesn't fit any of the above categories.

Your output should only contain the index (from 1 to 6)."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(query_nature_classifier_prompt),
        HumanMessage(query)
    ]
    response = str(llm.invoke(messages).content)
    return response

def get_previous_messages(turns: int) -> list:
    max_previous_turns = -2 * turns
    if len(conversation_history.messages) >= max_previous_turns:
        return conversation_history.messages[max_previous_turns:-1]
    return conversation_history.messages[:-1]

def are_queries_related(previous_query: HumanMessage, previous_response: AIMessage, current_query: str) -> bool:
    find_relation_prompt = f"""Are the following current query and previous query and its response related? Answer only in 'YES' or 'NO'.
Current Query: {current_query}
Previous Query: {previous_query.content}
Previous Response: {previous_response.content}"""
    messages = [
        SystemMessage(find_relation_prompt),
    ]
    response = str(llm.invoke(messages).content)
    return 'YES' in response

def filter_messages_based_on_similarity(previous_messages: list, query: str) -> list:
    previous_related_queries_and_responses = []
    for i, message in enumerate(previous_messages):
        if isinstance(message, HumanMessage):
            if are_queries_related(message, previous_messages[i + 1], query):
                previous_related_queries_and_responses.append(message)
                previous_related_queries_and_responses.append(previous_messages[i + 1])
    return previous_related_queries_and_responses

def get_information_from_database(query: str, k: int) -> list:
    retriever = vector_store.as_retriever(search_kwargs={'k': k}, search_type='similarity')
    documents = retriever.invoke(input=query)
    return documents

def create_prompt(title: str, documents: list) -> str:
    prompt = title + ':\n'
    for document in documents:
        if isinstance(document, HumanMessage) or isinstance(document, AIMessage):
            prompt = prompt + str(document.content)
        elif isinstance(document, Document):
            prompt = prompt + str(document.page_content)
    return prompt

def answer_greetings(query: str) -> dict:
    greetings_query_prompt = """The following user query has been classified as "greeting". Provide an appropriate response to it."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(greetings_query_prompt),
        HumanMessage(query)
    ]
    response = {
        'message': str(llm.invoke(messages).content),
        'classification': 'greetings',
        'context': [],
        'previous_messages_referred': []
    }
    return response

def answer_previous_message_clarification(query: str) -> dict:
    previous_message_clarification_prompt = """The following user query is classified as "previous response clarification". The user is asking to elaborate your previous answer."""
    previous_messages = get_previous_messages(turns=1)
    previous_messages = create_prompt('Previous User Query and Assistant Response', previous_messages)
    context = get_information_from_database(query, k=TOP_K)
    context = create_prompt('Extracted Information from Documentation', context)
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(previous_message_clarification_prompt),
        SystemMessage(previous_messages),
        SystemMessage(context),
        HumanMessage(query)
    ]
    response = {
        'message': str(llm.invoke(messages).content),
        'classification': 'previous message clarification',
        'context': context,
        'previous_messages_referred': previous_messages
    }
    return response

def answer_relevant_question(query: str) -> dict:
    context = get_information_from_database(query, k=TOP_K)
    context = create_prompt('Extracted Information from Documentation', context)
    previous_messages = get_previous_messages(turns=5)
    previous_messages = filter_messages_based_on_similarity(previous_messages, query)
    previous_messages = create_prompt('Previous related user queries and responses', previous_messages)
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(context),
        SystemMessage(previous_messages),
        HumanMessage(query)
    ]
    response = {
        'message': str(llm.invoke(messages).content),
        'classification': 'relevant question',
        'context': [],
        'previous_messages_referred': []
    }
    return response

def answer_action_request(query: str) -> dict:
    action_request_prompt = """The following user query has been classified as "action request". But you do not have access to any of the ASU supercomputers or terminal to execute the user query and are just an AI assistant who can quickly refer stored documentation and answer the user's query. Let the user know this in your response."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(action_request_prompt),
        HumanMessage(query)
    ]
    response = {
        'message': str(llm.invoke(messages).content),
        'classification': 'action request',
        'context': [],
        'previous_message_referred': []
    }
    return response

def answer_irrelevant_question(query: str) -> dict:
    irrelevant_question_prompt = """The following user query has been classified as "irrelevant". So let the user know this and ask them to provide more context if they feel this query is relevant."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(irrelevant_question_prompt),
        HumanMessage(query)
    ]
    response = {
        'message': str(llm.invoke(messages).content),
        'classification': 'irrelevant question',
        'context': [],
        'previous_message_referred': []
    }
    return response

def answer_other(query: str) -> dict:
    other_prompt = """The following user query doesn't fit any of the categories and is classifies into "other" category. Let the user know that you do not have an answer for it and they should contact ASU Research Computing team with this query for best response."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(other_prompt),
        HumanMessage(query)
    ]
    response = {
        'message': str(llm.invoke(messages).content),
        'classification': 'other',
        'context': [],
        'previous_message_referred': []
    }
    return response

def main():
    while True:
        user_query = input('You: ')
        if user_query == '\\quit':
            break
        elif user_query == '\\clear':
            conversation_history.clear()
            continue
        conversation_history.add_user_message(user_query)
        query_class = int(classify_query(user_query))
        if query_class == 1:
            response = answer_greetings(user_query)
        elif query_class == 2:
            response = answer_relevant_question(user_query)
        elif query_class == 3:
            response = answer_previous_message_clarification(user_query)
        elif query_class == 4:
            response = answer_action_request(user_query)
        elif query_class == 5:
            response = answer_irrelevant_question(user_query)
        else:
            response = answer_other(user_query)
        conversation_history.add_ai_message(response['message'])
        print(f'AI Assistant: {response['message']}')

if __name__ == '__main__':
    main()
