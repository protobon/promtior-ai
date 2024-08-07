# promtior-ai: Conversational RAG Application
### Description
Conversational Retrieval-Augmented Generation (RAG) application that uses **Langchain**, **Chroma** and **OpenAI** to answer questions about 
the company Promtior.ai.

This project was created for a challenge launched by Promtior.ai. For more details, checkout the
`/data/promtior.pdf` file.

[Try it out](http://promti-publi-m0rhp4qeuozy-32323717.us-east-2.elb.amazonaws.com/promtior/playground/) 👈

Suggested questions:
>What services does Promtior offer?
> 
>When was the company founded?
> 
>What are their use cases?
> 
>What is their contact information?


### Project Structure

The project consists of two main parts:

`database.py`: Script for gathering relevant information about the company and storing it in a vector store, 
saved to the persistent directory `db/`.

`server.py`: Contains the application logic.

For more details about the application logic, refer to `doc/project documentation.pdf`.