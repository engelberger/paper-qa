from typing import Any, Dict, List, Optional, Tuple, Union
from functools import reduce
import re
from .utils import maybe_is_text, maybe_is_truncated
from .qaprompts import distill_chain, qa_chain, edit_chain
from dataclasses import dataclass
from .readers import read_doc
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

@dataclass
class Answer:
    """A class to hold the answer to a question."""

    question: str
    answer: str
    context: str
    references: str
    formatted_answer: str


class Docs:
    """A collection of documents to be used for answering questions."""

    def __init__(self, chunk_size_limit: int = 3000) -> None:
        if not isinstance(chunk_size_limit, int):
            raise TypeError("chunk_size_limit should be an integer")
        if chunk_size_limit < 0:
            raise ValueError("chunk_size_limit should be a positive integer")
        self.docs = dict()
        self.chunk_size_limit = chunk_size_limit
        self.keys = set()
        self._faiss_index = None


    def add(
        self,
        path: str,
        citation: str,
        key: Optional[str] = None,
        disable_check: bool = False,
    ) -> None:
        """Add a document to the collection.
        Args:
            path (str): Path to the document (e.g. a PDF).
            citation (str): Citation of the document, e.g. "Smith 2020".
            key (str, optional): Key to refer to the document. If not specified, we
                try to extract it from the citation. Defaults to None.
            disable_check (bool, optional): Disable the check that the document
                has been loaded correctly. Defaults to False.
        """
        if not os.path.exists(path):
            raise ValueError(f"Path {path} not found.")
        if path in self.docs:
            raise ValueError(f"Document {path} already in collection.")
        if key is None:
            # get first name and year from citation
            try:
                author = re.search(r"([A-Z][a-z]+)", citation).group(1)
            except AttributeError:
                # panicking - no word??
                raise ValueError(
                    f"Could not parse key from citation {citation}. Consider just passing key explicitly - e.g. docs.py (path, citation, key='mykey')"
                )
            try:
                year = re.search(r"(\d{4})", citation).group(1)
            except AttributeError:
                year = ""
            key = f"{author}{year}"
        suffix = ""
        while key + suffix in self.keys:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        key += suffix
        self.keys.add(key)

        texts, metadata = read_doc(path, citation, key)
        # loose check to see if document was loaded
        if not disable_check and not maybe_is_text("".join(texts)):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )

        self.docs[path] = dict(texts=texts, metadata=metadata, key=key)
        if self._faiss_index is not None:
            self._faiss_index.add_texts(texts, metadatas=metadata)

    # to pickle, we have to save the index as a file
    def __getstate__(self):
        if self._faiss_index is None:
            self._build_faiss_index()
        state = self.__dict__.copy()
        try:
            state["_faiss_index"].save_local("faiss_index")
            del state["_faiss_index"]
        except:
            print("Error in saving faiss index")
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Update the state of the class with the state that was passed in.
        # This is a standard way to restore the state of a class.
        self.__dict__.update(state)
        # Load the FAISS index from the state.
        self._faiss_index = FAISS.load_local("faiss_index", OpenAIEmbeddings())


    def _build_faiss_index(self) -> None:
        """
        A method to build a FAISS index of the documents in the collection.
        In order to do this, we need to create a list of all the texts in our documents 
        and a list of all the metadata in our documents.
        We then create a new FAISS index using the FAISS class from haystack and the “from_texts” method.
        """
        if self._faiss_index is None:
            # create a list of all the texts in our documents
            texts: List[str] = reduce(
                lambda x, y: x + y, [doc["texts"] for doc in self.docs.values()], []
            )
            # create a list of all the metadata in our documents
            metadatas: List[Dict[str, Any]] = reduce(
                lambda x, y: x + y, [doc["metadata"] for doc in self.docs.values()], []
            )
            try:
                # create a new FAISS index using the FAISS class from haystack and the “from_texts” method.
                self._faiss_index = FAISS.from_texts(
                    texts, OpenAIEmbeddings(), metadatas=metadatas
                )
            except TypeError:
                self._faiss_index = None


    def get_evidence(self, question: str, k: int = 3, max_sources: int = 5) -> Tuple[str, Dict[str, str]]:
        """
        A method to get the evidence for a given question.
        We first use the FAISS index to find the documents that are most relevant to the question.
        We then use the distill chain to extract the relevant sentences from each document.
        We then return the context string and the citations for the documents.
        
        Args:
            question (str): The question to be answered.
            k (int, optional): The number of documents to be returned. Defaults to 3.
            max_sources (int, optional): The maximum number of sources to be returned. Defaults to 5.
        """        
        # Step 1: Initialize context
        context = []
        # Step 2: If self._faiss_index is None, build the faiss index
        if self._faiss_index is None:
            self._build_faiss_index()
        # Step 3: Get the top k documents from the faiss index
        docs = self._faiss_index.max_marginal_relevance_search(
            question, k=k, fetch_k=5 * k
        )
        # Step 4: For each document, run distill chain
        for doc in docs:
            # want to work through indices but less k
            c = (
                doc.metadata["key"],
                doc.metadata["citation"],
                distill_chain.run(question=question, context_str=doc.page_content),
            )
            # Step 5: If the context is not empty, append it to context
            if "Not applicable" not in c[-1]:
                context.append(c)
            # Step 6: If the length of the context is equal to max_sources, break the loop
            if len(context) == max_sources:
                break
        # Step 7: Concatenate the context strings
        context_str = "\n\n".join(
            [f"{k}: {s}" for k, c, s in context if "Not applicable" not in s]
        )
        # Step 8: Create a list of valid keys
        valid_keys = [k for k, c, s in context if "Not applicable" not in s]
        # Step 9: If there are valid keys, add them to the context string
        if len(valid_keys) > 0:
            context_str += "\n\nValid keys: " + ", ".join(valid_keys)
        # Step 10: Return the context string and the context
        return context_str, {k: c for k, c, s in context}


    def query(
        self,
        query: str,
        k: int = 5,
        max_sources: int = 5,
        length_prompt: str = "about 100 words",
    ) -> Answer:
        """
        A method to query the collection.
        We first get the evidence for the question.
        We then use the QA chain to get the answer.
        We then use the edit chain to edit the answer if it is truncated.
        We then return the answer.
        
        Args:
            query (str): The question to be answered.
            k (int, optional): The number of documents to be returned. Defaults to 5.
            max_sources (int, optional): The maximum number of sources to be returned. Defaults to 5.
            length_prompt (str, optional): The length prompt to be used for the QA chain. Defaults to "about 100 words".
        """
        # Get the context and citations from the evidence.
        context_str, citations = self.get_evidence(query, k=k, max_sources=max_sources)
        bib = dict()
        # Check if there is enough information to answer the question.
        if len(context_str) < 10:
            answer = "I cannot answer this question due to insufficient information."
        # If there is enough information, get the answer.
        else:
            answer = qa_chain.run(
                question=query, context_str=context_str, length=length_prompt
            )[1:]
            # If the answer is truncated, edit it.
            if maybe_is_truncated(answer):
                answer = edit_chain.run(question=query, answer=answer)
        # Add the citations to the answer.
        for key, citation in citations.items():
            # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
            skey = key.split(" ")[0]
            if skey + " " in answer or skey + ")" in answer:
                bib[skey] = citation
        bib_str = "\n\n".join(
            [f"{i+1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )
        formatted_answer = f"Question: {query}\n\n{answer}\n"
        if len(bib) > 0:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"
        return Answer(
            answer=answer,
            question=query,
            formatted_answer=formatted_answer,
            context=context_str,
            references=bib_str,
        )
