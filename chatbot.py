from mdllama.rag import MdRAG

rag = MdRAG(llama_host='http://127.0.0.1:25565', chat_model='llama3.1', embed_model='mxbai-embed-large')
# rag.build_data_chain(['res/1_primer.md', 'res/2_point.md', 'res/3_system.md', 'res/4_analytical.md', 'res/5_central.md'], encoding=None)
rag.build_data_chain(['res/output.md'])
# rag.load_data_chain('res/vectorstore')
chatbot = rag.chatbot()

while True:
    prompt = input(">> ")
    if prompt == "/bye": break
    response = chatbot.chat(prompt).replace('\\\\', '\\')
    response.replace('\\\\', '\\')
    with open('res/response.md', 'a') as f:
        f.write(f'# User\n{prompt}\n# Assitant\n{response}\n')
    print('\n')
