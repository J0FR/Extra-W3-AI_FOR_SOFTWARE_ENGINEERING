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

- Output Terminal

```
The Self-Reflection prompt involves showing two-shot examples to an agent, where each example consists of a pair of a failed trajectory and an ideal reflection for guiding future changes in the plan. These reflections are then added to the agent's working memory, up to three, to provide context for querying the Language Model (LLM) for generating high-level questions based on observations/statements. The goal is to enable autonomous agents to improve iteratively by refining past action decisions and correcting mistakes through self-reflection.
To create a successful Self-Reflection prompt, it is essential to provide clear and concise examples of failed trajectories and ideal reflections for guiding future changes in the plan. Ensure that the reflections are actionable and specific, focusing on areas for improvement or adjustments in decision-making. Additionally, incorporating up to three reflections into the agent's working memory can enhance the context for querying the Language Model (LLM) effectively.
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

- Output Terminal

```
{'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_ydQdsfWxboybYJAAIeTJ45KH', 'function': {'arguments': '{"query":"Self-Reflection prompt"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 69, 'total_tokens': 89}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c626f18b-01b3-45e4-9157-558d15e58863-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'Self-Reflection prompt'}, 'id': 'call_ydQdsfWxboybYJAAIeTJ45KH', 'type': 'tool_call'}], usage_metadata={'input_tokens': 69, 'output_tokens': 20, 'total_tokens': 89})]}}
----
{'tools': {'messages': [ToolMessage(content="Another quite distinct approach, LLM+P (Liu et al. 2023), involves relying on an external classical planner to do long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface to describe the planning problem. In this process, LLM (1) translates the problem into “Problem PDDL”, then (2) requests a classical planner to generate a PDDL plan based on an existing “Domain PDDL”, and finally (3) translates the PDDL plan back into natural language. Essentially, the planning step is outsourced to an external tool, assuming the availability of domain-specific PDDL and a suitable planner which is common in certain robotic setups but not in many other domains.\nSelf-Reflection#\nSelf-reflection is a vital aspect that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes. It plays a crucial role in real-world tasks where trial and error are inevitable.\n\nPrompt LM with 100 most recent observations and to generate 3 most salient high-level questions given a set of observations/statements. Then ask LM to answer those questions.\n\n\nPlanning & Reacting: translate the reflections and the environment information into actions\n\nPlanning is essentially in order to optimize believability at the moment vs in time.\nPrompt template: {Intro of an agent X}. Here is X's plan today in broad strokes: 1)\nRelationships between agents and observations of one agent by another are all taken into consideration for planning and reacting.\nEnvironment information is present in a tree structure.\n\nFig. 3. Illustration of the Reflexion framework. (Image source: Shinn & Labash, 2023)\nThe heuristic function determines when the trajectory is inefficient or contains hallucination and should be stopped. Inefficient planning refers to trajectories that take too long without success. Hallucination is defined as encountering a sequence of consecutive identical actions that lead to the same observation in the environment.\nSelf-reflection is created by showing two-shot examples to LLM and each example is a pair of (failed trajectory, ideal reflection for guiding future changes in the plan). Then reflections are added into the agent’s working memory, up to three, to be used as context for querying LLM.\n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.", name='blog_post_retriever', tool_call_id='call_ydQdsfWxboybYJAAIeTJ45KH')]}}
----
{'agent': {'messages': [AIMessage(content='Self-reflection is a vital aspect that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes. It plays a crucial role in real-world tasks where trial and error are inevitable. Self-reflection involves prompting a language model with the 100 most recent observations and asking it to generate three high-level questions based on those observations. This process helps in gaining insights and improving decision-making for autonomous agents.', response_metadata={'token_usage': {'completion_tokens': 85, 'prompt_tokens': 672, 'total_tokens': 757}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-74113b1f-284a-4ad0-b295-2787a794f9b4-0', usage_metadata={'input_tokens': 672, 'output_tokens': 85, 'total_tokens': 757})]}}
----
{'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_JMHCOswkg8nEcRMVnHp30Ucy', 'function': {'arguments': '{"query":"Tips for making a Self-Reflection prompt"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 24, 'prompt_tokens': 777, 'total_tokens': 801}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-6eb7cae9-0676-4fdf-97bd-eb6c76d81ba7-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'Tips for making a Self-Reflection prompt'}, 'id': 'call_JMHCOswkg8nEcRMVnHp30Ucy', 'type': 'tool_call'}], usage_metadata={'input_tokens': 777, 'output_tokens': 24, 'total_tokens': 801})]}}
----
{'tools': {'messages': [ToolMessage(content="Prompt LM with 100 most recent observations and to generate 3 most salient high-level questions given a set of observations/statements. Then ask LM to answer those questions.\n\n\nPlanning & Reacting: translate the reflections and the environment information into actions\n\nPlanning is essentially in order to optimize believability at the moment vs in time.\nPrompt template: {Intro of an agent X}. Here is X's plan today in broad strokes: 1)\nRelationships between agents and observations of one agent by another are all taken into consideration for planning and reacting.\nEnvironment information is present in a tree structure.\n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.\n\nFig. 3. Illustration of the Reflexion framework. (Image source: Shinn & Labash, 2023)\nThe heuristic function determines when the trajectory is inefficient or contains hallucination and should be stopped. Inefficient planning refers to trajectories that take too long without success. Hallucination is defined as encountering a sequence of consecutive identical actions that lead to the same observation in the environment.\nSelf-reflection is created by showing two-shot examples to LLM and each example is a pair of (failed trajectory, ideal reflection for guiding future changes in the plan). Then reflections are added into the agent’s working memory, up to three, to be used as context for querying LLM.\n\nAnother quite distinct approach, LLM+P (Liu et al. 2023), involves relying on an external classical planner to do long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface to describe the planning problem. In this process, LLM (1) translates the problem into “Problem PDDL”, then (2) requests a classical planner to generate a PDDL plan based on an existing “Domain PDDL”, and finally (3) translates the PDDL plan back into natural language. Essentially, the planning step is outsourced to an external tool, assuming the availability of domain-specific PDDL and a suitable planner which is common in certain robotic setups but not in many other domains.\nSelf-Reflection#\nSelf-reflection is a vital aspect that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes. It plays a crucial role in real-world tasks where trial and error are inevitable.", name='blog_post_retriever', tool_call_id='call_JMHCOswkg8nEcRMVnHp30Ucy')]}}
----
{'agent': {'messages': [AIMessage(content='Some tips for making a Self-Reflection prompt perfectly include:\n\n1. Prompt the language model with the 100 most recent observations and ask it to generate three high-level questions based on those observations.\n2. Translate the reflections and environment information into actionable plans for optimization.\n3. Consider relationships between agents and observations for effective planning and reacting.\n4. Utilize resources like internet access, long-term memory management, GPT-3.5 powered agents, and file output for efficient self-reflection.\n5. Continuously review and analyze actions to ensure optimal performance.\n6. Engage in constructive self-criticism to improve behavior.\n7. Reflect on past decisions and strategies to refine approaches.\n8. Be smart and efficient in completing tasks with the least number of steps.\n\nThese tips can help in creating a comprehensive and effective Self-Reflection prompt for autonomous agents.', response_metadata={'token_usage': {'completion_tokens': 173, 'prompt_tokens': 1384, 'total_tokens': 1557}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-57ce2c1e-5070-4567-9698-dcaa5420bf02-0', usage_metadata={'input_tokens': 1384, 'output_tokens': 173, 'total_tokens': 1557})]}}
----
```
