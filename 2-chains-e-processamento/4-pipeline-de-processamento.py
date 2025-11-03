from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial-text"],
    template="Translate the following text to English:\n ``` {initial-text}``"
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Sumarize the following text in 4 words:\n ``` {text}``\n\n"
)


llm_en = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

result = pipeline.invoke({"initial-text": "LangChain é um framework para desenvolver aplicações com modelos de linguagem."})
print("***** Result:", result)