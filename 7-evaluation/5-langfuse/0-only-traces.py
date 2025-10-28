"""Demonstração de uso de traces no Langfuse com LangChain."""
from datetime import datetime
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Inicializa o callback handler do Langfuse
callback_handler = CallbackHandler()

# Inicializa o LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500)

# Define o prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é historiador especializado em linguagens de programação.
     Seu objetivo é contar a historia de linguagens de programação de forma interessante e educativa.
     
     Inclua em sua resposta:
     - Quando a linguagem foi criada e por quem
     - Qual problema ela veio resolver
     - Principais características e diferenciais
     - Evolução ao longo do tempo
     - Impacto no desenvolvimento de software
     
     Seja conciso, mas informativo. Use um tom narrativo envolvente."""),
    ("user", "Conte a história da linguagem de programação {language}.")
])

# Cria a chain
chain = prompt | llm

def generate_language_history(language):
    """Gera a historia de uma linguagem usando LLM com trace automático."""
    response = chain.invoke(
        {"language": language},
        config={"callbacks": [callback_handler]}
    )
    return response.content

if __name__ == "__main__":
    languages = ["Python", "Go", "Kotlin"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nExeutando traces - {timestamp}\n")
    print("=" * 60)

    for language in languages:
        print(f"\nGerando história para: {language}\n")
        generate_language_history(language)
     