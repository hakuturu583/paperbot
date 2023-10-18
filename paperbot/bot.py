from paperbot.datatype import MetaData, UserAction, Language

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.enum import EnumOutputParser
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

    def detect_language(self, text: str):
        lang = langdetect.detect(text)
        if lang == "ja":
            return Language.JAPANESE
        elif lang == "en":
            return Language.ENGLISH
        else:
            return None

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

    def summary(self, sentences: List[Document], language: Language) -> str:
        prompt_template = ""
        if language == Language.ENGLISH:
            prompt_template = (
                "Write a concise summary of the following in English :{text}"
            )
        elif language == Language.JAPANESE:
            prompt_template = "以下の英文の詳細な要約を日本語で行ってください :{text}"
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        return stuff_chain.run(sentences)

    def answer(
        self, sentences: List[Document], question: str, update_embedding: bool = False
    ):
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
        qdrant = self.load_qdrant(sentences, update_embedding)
        model = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",
            retriever=qdrant.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            ),
            return_source_documents=False,
            verbose=True,
        )
        print(
            "Summary : "
            + self.summary(self.load_string(self.summary_by_sumy(sentences)))
        )
        return model.run(question)

    def load_qdrant(self, sentences: List[Document], update_embedding: bool = False):
        client = QdrantClient(
            url=os.environ["QDRANT_URI"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        metadata = self.query_metadata(sentences)
        collection_name = metadata.title.replace(" ", "_").replace(":", ",")
        if collection_name not in collection_names or update_embedding:
            if collection_name in collection_names:
                client.delete_collection(collection_name)
            summary_text = (
                "The summary of the paper is below. If you ask the summary of this paper, please use sentences below."
                + self.summary(self.load_string(self.summary_by_sumy(sentences)))
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            if metadata != None:
                embeddings = OpenAIEmbeddings()
                db = Qdrant.from_documents(
                    sentences + self.load_string(summary_text),
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

    def interpret_action(self, question: str):
        output_parser = EnumOutputParser(enum=UserAction)
        prompt = PromptTemplate(
            template="User request is {question}. \nPlease choose which action is suitable from user request.\n{format_instructions}\n",
            input_variables=["question"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )
        llm = OpenAI(temperature=0)
        try:
            return output_parser.parse(
                llm(prompt.format_prompt(question=question).to_string())
            )
        except:
            return None


if __name__ == "__main__":
    bot = PaperBot()
    # print(bot.answer(bot.load_pdf("2309.17080.pdf"), "この論文ではどのように拡散モデルが利用されていますか？"))
    # print(
    #     bot.answer(bot.load_pdf("2309.17080.pdf"), "Please summary this paper.", False)
    # )
    print(bot.interpret_action("Please summary this paper."))
