# Extra-W3-AI_FOR_SOFTWARE_ENGINEERING

## What is a Conversational RAG (Retrieval-Augmented Generation)?

Conversational RAG (Retrieval-Augmented Generation) combines LLMs (Large Language Models) with external knowledge sources to enhance Q&A applications.

## How to do a Conversational RAG (Retrieval-Augmented Generation)?

The documentation demonstrates building such applications using LangChain with two approaches: Chains and Agents.

## How Chains work?

The Chains approach involves using predefined logic to handle the retrieval and response generation processes. This ensures predictability and consistency in how the system responds to queries by following a structured flow.

### Steps:

- Construct Retriever
  Load and index the blog content to create a retriever.

```
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

- Contextualize Question
  Formulate questions using chat history.

```
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

- Answer Question
  Generate answers using the retrieved context.

```
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

- Manage Chat History
  Statefully manage chat history to maintain conversation context.

```
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```

- Run agent
  Try the agent to show that it's working.

```
query = "What is Self-Reflection prompt?"
session_id = "cooking-session"
response = conversational_rag_chain.invoke({"input": query}, config={"configurable": {"session_id": session_id}})
print(response["answer"])

second_query = "What are some tips for making a Self-Reflection prompt perfectly?"
response = conversational_rag_chain.invoke({"input": second_query}, config={"configurable": {"session_id": session_id}})
print(response["answer"])
```

### How Agents work?

The Agents approach leverages the reasoning capabilities of LLMs to make decisions during execution, providing flexibility. Agents can decide when and how to perform retrieval steps, and manage multiple retrieval steps if necessary.

### Steps:

- Construct Retriever
  Load and index the blog content to create a retriever.

```
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

- Build Retriever Tool
  Convert the retriever into a tool that the agent can use.

```
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]
```

- Create Agent
  Create an agent with tool-calling capabilities, using LangGraph's prebuilt agent functionality and SQLite for checkpointing.

```
memory = SqliteSaver.from_conn_string(":memory:")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_react_agent(llm, tools, checkpointer=memory)
```

- Execute Queries
  Try some queries to prove that it's working.

```
query_1 = "What is Self-Reflection prompt?"
config = {"configurable": {"thread_id": "abc123"}}

for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query_1)]}, config=config
):
    print(s)
    print("----")

query_2 = "What are some tips for making a Self-Reflection prompt perfectly?"
for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query_2)]}, config=config
):
    print(s)
    print("----")
```
