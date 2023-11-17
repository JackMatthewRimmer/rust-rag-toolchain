# Prerequisites
 * Docker running on machine
 * .env file containing (Will need to name these)
    * Source database url
    * Source database password
    * Source database username
    * open ai api key

# Anticipated modules
* **Chunking module**
   * most likely want it to facilitate large amounts of raw text.
   then once we load a row -> new thread where its split into chunks, made into a vector and stored in vector database

* **Vector DB interface** 
   * module for all things to do with the local vector database 
   * probs want this to include the abstraction for adding and searching the vector database

* **GPT embedding interface**
   * self explanatory, just a clean abstraction for generating an embedding for some text

* **rag-toolchain**
   * this is where we want to provide the public interface for the library
   * also 

* **rag-orchestrator**
   * probably want a module to handle the control flow of data and bridge all of these libraries together