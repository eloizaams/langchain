from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
load_dotenv()

long_text = """
In recent years, the integration of sustainable technology has fundamentally reshaped various aspects of modern society. From renewable energy sources like solar and wind power to advanced recycling systems and electric vehicles, these innovations offer a promising path towards mitigating climate change and reducing our collective carbon footprint. The shift towards these cleaner alternatives is driven by growing environmental awareness, as well as economic incentives and governmental policies aimed at promoting eco-friendly practices.
For example, the widespread adoption of solar panels on residential and commercial buildings has not only reduced reliance on fossil fuels but has also empowered individuals and businesses to generate their own clean energy. This trend, combined with breakthroughs in energy storage solutions, is making the transition to a decentralized and more resilient energy grid a tangible reality. Furthermore, smart home technologies are optimizing energy consumption by adjusting heating, lighting, and cooling systems based on user behavior and external conditions, leading to significant energy savings.
In the transportation sector, electric vehicles (EVs) are becoming increasingly common, thanks to improvements in battery technology and expanding charging infrastructure. Major automakers are investing heavily in EV production, and governments are offering tax credits and subsidies to encourage their purchase. This shift away from internal combustion engines is poised to dramatically reduce urban air pollution and decrease dependence on volatile oil markets.
Beyond energy and transportation, sustainable technology is also transforming industries like agriculture and manufacturing. Precision agriculture uses sensors and data analytics to optimize water and nutrient usage, minimizing waste and environmental impact. Similarly, "green" manufacturing processes are designed to reduce waste, conserve resources, and minimize pollution, demonstrating that economic growth and environmental stewardship can go hand in hand.
The long-term success of this transition, however, depends on continued innovation, supportive policies, and widespread public adoption. The challenges remain significant, including the high initial cost of some technologies and the need for new infrastructure, but the potential benefits for the planet and future generations are immense.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, # 250 caracteres por pedaço
    chunk_overlap=70,# 70 caracteres de sobreposição
)

parts = splitter.create_documents([long_text])

# for part in parts:
#    print(part.page_content)
#    print("-"*30)

llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0)

# Stuff: Junta todas as partes e faz um único resumo. Faz uma única chamada ao modelo.
chain_summarize = load_summarize_chain(llm, chain_type="stuff", verbose=False)

result = chain_summarize.invoke({"input_documents": parts})
print(result["output_text"])