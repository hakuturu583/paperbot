from paperbot.datatype import MetaData

from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import SpacyTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

# from sumy.summarizers.lsa import LsaSummarizer as Summarizer
# from sumy.summarizers.luhn import LuhnSummarizer as Summarizer
# from sumy.summarizers.reduction import ReductionSummarizer as Summarizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import spacy

import nltk

from typing import List, Optional
import langdetect
import logging
import os


class PaperBot:
    def __init__(self, log_level: int = logging.DEBUG):
        self.logger = logging.getLogger("paperbot")
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.basicConfig(level=log_level, handlers=[stream_handler])

    def show_sentences(self, sentences: List[Document]):
        for i, sentence in enumerate(sentences):
            if i == 0:
                self.logger.info("Loading 1st sentence.")
            elif i == 1:
                self.logger.info("Loading 2nd sentence.")
            elif i == 2:
                self.logger.info("Loading 3rd sentence.")
            else:
                self.logger.info("Loading " + str(i + 1) + " th sentence.")
            self.logger.info(
                "========================== Sentence Start ==================================="
            )
            self.logger.info(sentence.page_content)
            self.logger.info(
                "========================== Sentence End ===================================\n"
            )

    def load_pdf(self, pdf_path: Path, chunk_size: int = 256) -> List[Document]:
        loader = PyPDFLoader(pdf_path)
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size).split_documents(
            SpacyTextSplitter(
                chunk_size=chunk_size, pipeline="en_core_web_lg", separator=""
            ).split_documents(loader.load_and_split())
        )

    def load_string(self, text: str, chunk_size: int = 256) -> List[Document]:
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size).split_documents(
            SpacyTextSplitter(
                chunk_size=chunk_size, pipeline="en_core_web_lg", separator=""
            ).split_documents(
                [Document(page_content=text, metadata={"source": "local"})]
            )
        )

    def convert_to_text_list(self, sentences: List[Document]) -> List[str]:
        ret: List[str] = []
        for sentence in sentences:
            ret.append(sentence.page_content)
        return ret

    def concat_sentences(self, sentences: List[Document]) -> str:
        ret: str = ""
        for sentence in sentences:
            ret = ret + sentence.page_content
        return ret

    def query_metadata(self, sentences: List[Document]) -> Optional[MetaData]:
        output_parser = PydanticOutputParser(pydantic_object=MetaData)
        question_text = ""
        for sentence in sentences[:5]:
            question_text = question_text + sentence.page_content
        prompt = PromptTemplate(
            template="Please guess the names of title and authors of this paper. The begining of the paper is below.\n{format_instructions}\n{question}",
            input_variables=["question"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )
        llm = OpenAI(temperature=0)
        try:
            return output_parser.parse(
                llm(prompt.format_prompt(question=question_text).to_string())
            )
        except:
            return None

    def summary_by_sumy(
        self, sentences: List[Document], summary_sentences_count: int = 3
    ) -> str:
        language = "english"
        nltk.download("punkt")
        parser = PlaintextParser.from_string(
            self.concat_sentences(sentences), Tokenizer(language)
        )
        stemmer = Stemmer("english")
        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)
        summary: str = ""
        for sentence in summarizer(parser.document, summary_sentences_count):
            summary = summary + sentence.__str__()
        return summary

    def summary(self, sentences: List[Document]):
        chain = load_summarize_chain(
            ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"),
            chain_type="stuff",
        )
        return chain.run(sentences)

    def summary_pdf(self, pdf_path: Path, chunk_size: int = 256):
        return self.summary(
            self.load_string(self.summary_by_sumy(self.load_pdf(pdf_path, chunk_size)))
        )

    def answer(self, sentences: List[Document], question: str):
        lang = langdetect.detect(question)
        if lang == "ja":
            question = question + "回答は日本語でお願いします。"
        elif lang == "en":
            question = question + "Please ask in Englist."
        else:
            return "Unsupported language detected, did you ask in " + lang + "?"
        client = QdrantClient(
            url=os.environ["QDRANT_URI"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
        qdrant = self.load_qdrant(sentences)
        model = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",
            retriever=qdrant.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10},
            ),
            return_source_documents=False,
            verbose=True,
        )
        return model.run(question)

    def load_qdrant(self, sentences: List[Document]):
        client = QdrantClient(
            url=os.environ["QDRANT_URI"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        metadata = self.query_metadata(sentences)
        collection_name = metadata.title.replace(" ", "_").replace(":", ",")
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            if metadata != None:
                embeddings = OpenAIEmbeddings()
                db = Qdrant.from_documents(
                    sentences,
                    embeddings,
                    url=os.environ["QDRANT_URI"],
                    api_key=os.environ["QDRANT_API_KEY"],
                    collection_name=collection_name,
                )
        else:
            logging.info(
                "Collection "
                + collection_name
                + " was found. Skip calculate embeddings."
            )
        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=OpenAIEmbeddings(),
        )


if __name__ == "__main__":
    bot = PaperBot()
    # bot.construct_vector_database(bot.load_pdf("2309.17080.pdf"))
    # print(bot.answer(bot.load_pdf("2309.17080.pdf"), "この論文ではどのように拡散モデルが利用されていますか？"))
    print(bot.summary_pdf("2309.17080.pdf"))