import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import yaml
import pymssql
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Annotated, TypedDict
from langgraph.graph import END,StateGraph
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import PyPDFLoader
import nltk
from nltk.corpus import stopwords
import datetime
import random
import uuid
import altair as alt

# Cargar variables de entorno
load_dotenv()

# Configuración de la aplicación Streamlit
st.set_page_config(page_title="Field Feelings!", 
                   page_icon="Logo Beats AI.jpg", 
                   layout='wide')


st.title("Field Feelings 😊")
col1, col2 = st.columns([4, 1])

# Leer el archivo sentimientos.csv
sentimientos = pd.read_csv("sentimientos.csv")


# Comprobar si la columna 'Feeling' existe
if 'Feeling' in sentimientos.columns:
    # Agrupar por el campo 'Feeling' y contar la cantidad de ocurrencias de cada sentimiento
    grupos = sentimientos.groupby("Feeling").size().reset_index(name='Count')

    # Mostrar el resultado en Streamlit
    with col2:
        # Título de la sección
        st.subheader("Distribución de Sentimientos")

        # # Mostrar la tabla de grupos con estilo
        # st.dataframe(grupos.style.highlight_max(axis=0, color='lightgreen'))

        # Crear un gráfico de barras con Altair
        chart = alt.Chart(grupos).mark_bar().encode(
            x=alt.X('Feeling:N', title='Sentimiento'),
            y=alt.Y('Count:Q', title='Cantidad'),
            color=alt.Color('Feeling:N', legend=None),
            tooltip=['Feeling', 'Count']
        ).properties(
            title='Distribución de Sentimientos',
            width=600,
            height=400
        )

        # Mostrar el gráfico en Streamlit
        st.altair_chart(chart, use_container_width=True)
else:
    pass
    # st.error("La columna 'Feeling' no existe en el archivo 'sentimientos.csv'.")








# Asegúrate de que conversation_id está inicializado
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())


pdfs_links = [
    "BIENESTAR.pdf"
]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

pdfs_contents = []
for url in pdfs_links:
    loader = PyPDFLoader(url)
    pdf_pages = loader.load()
    pdf_text = ' '.join([clean_text(str(page.page_content)) for page in pdf_pages])
    pdfs_contents.append(pdf_text)

pdfs_df = pd.DataFrame({'Texto': pdfs_contents, 'Archivo': pdfs_links})

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

def text_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r',', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words).strip()

def text_to_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = ''
    word_count = 0
    for word in words:
        current_chunk += word + ' '
        word_count += 1
        if word_count >= chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ''
            word_count = 0
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks_list = []
for index, row in pdfs_df.iterrows():
    text = row['Texto']
    chunks = text_to_chunks(text)
    chunks_list.append(chunks)

index_repeated = []
for index, chunks in enumerate(chunks_list):
    index_repeated.extend([index]*len(chunks))

chunks_all = [chunk for chunks in chunks_list for chunk in chunks]

preprocessed_texts = []
for text in chunks_all:
    preprocessed_texts.append(text_preprocess(text))

chunks_df = pd.DataFrame({'Chunk': chunks_all, 'Texto preprocesado': preprocessed_texts, 'Índice': index_repeated})

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(chunks_df['Texto preprocesado'])

feature_names = tfidf_vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

chunks_df = pd.concat([chunks_df, tfidf_df], axis=1)

#########################################################################################################

class State(TypedDict):
    messages: Annotated[list[dict], "The messages in the conversation"]
    cliente_info: dict
    servicio: str
    info: dict
    base: dict
    creditos: dict
    pagos: dict
    chat_history:str
    user_query:str

def identificar_servicio(state: State):
    # Normalizar el contenido del último mensaje
    ultimo_mensaje = state['messages'][-1]['content'].lower()

    # Crear el DataFrame y reemplazar "NULL" y NaNs por 0
    CREDITO = pd.DataFrame(state['creditos']).replace("NULL", 0).fillna(0)

    # Obtener el saldo de capital
    saldo_capital = CREDITO['saldo_capital_dia'][0]

    # Normalizaciones para palabras clave en el mensaje
    liquidacion_keywords = ["liquidacion"]
    paz_y_salvo_keywords = ["paz y salvo"]

    # Identificar el servicio solicitado
    if any(keyword in ultimo_mensaje for keyword in liquidacion_keywords) and "radicar" in ultimo_mensaje:


        state['servicio'] = "liquidacion"
    elif any(keyword in ultimo_mensaje for keyword in paz_y_salvo_keywords) and "radicar" in ultimo_mensaje:


        state['servicio'] = "paz_y_salvo"
    else:
        state['servicio'] = "otro"

    return state


def generate_radicado():
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y%m%d")
    random_number = random.randint(1000, 9999)
    radicado = f"RAD-{date_string}-{random_number}"
    return radicado

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import json
import dateutil.parser

def get_feeling(user_query, chat_history):
    template = """
Eres un segmentador de sentimientos para el siguiente chat. Selecciona el sentimiento que el usuario tiene:

Historia del chat: `{chat_history}`   
Pregunta del usuario: `{user_query}`

Las opciones para "feeling" son:
- Feliz
- Tranquilo
- Annimado
- Amado
- Enojado
- Angustiado
- Estresado
- Deprimido

Siempre responde con el JSON bien formado sin comillas extra.

- Responde siempre al menos con la clase.
- Responde siempre con la llave "feeling" correctamente definida.
- Responde siempre con un JSON bien hecho, usando corchetes y comillas solo donde sea necesario.

Ejemplo:
```json
'''
    "feeling": "Feliz"

'''

    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY_2'], model_name="llama3-70b-8192")
    chain = prompt | llm | JsonOutputParser()

    ai_query = chain.invoke({
        "user_query": user_query, 
        "chat_history": chat_history[:3000]
    })
    return ai_query

def extract_liquidation_date(user_query: str) -> str:
    today = datetime.datetime.now()
    template = f"""
    Analiza el siguiente mensaje del cliente y extrae la fecha en la que desea la liquidación.
    Si no se menciona una fecha específica, asume que es para hoy{today}.
    Si se menciona una fecha relativa (como "mañana" o "en una semana"), calcula la fecha correspondiente.
    Devuelve la fecha en formato ISO (YYYY-MM-DD).

    Mensaje del cliente: {user_query}

    Responde solo con un JSON en este formato:
    "fecha": "YYYY-MM-DD"
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="llama3-70b-8192")
    chain = prompt | llm | JsonOutputParser()

    response = chain.invoke({"user_query": user_query})
    
    try:
        date_dict = json.loads(response)
        date_str = date_dict['fecha']
        parsed_date = dateutil.parser.parse(date_str).date()
        return parsed_date.isoformat()
    except:
        # Si hay algún error, devolvemos la fecha actual
        return datetime.date.today().isoformat()


def liquidacion_credito(state: State) -> dict:
    BASE = pd.DataFrame(state['base'])
    CREDITOS = pd.DataFrame(state['creditos'])
    
    nombre = BASE['Nombre'][0]
    credito = BASE['Credito'][0]
    saldo = CREDITOS['saldo_capital_dia'][0]
    radicado = generate_radicado()
    
    # Extraer la fecha de liquidación del mensaje del usuario
    user_query = state['messages'][-1]['content']
    liquidation_date = extract_liquidation_date(user_query)
    
    # Email for SAC
    sac_liquidacion = f"""
Solicitud de Liquidación

Radicado: {radicado}

Se ha recibido una solicitud de liquidación con los siguientes detalles:

Nombre del cliente: {nombre}
Número de crédito: {int(credito)}
Saldo a capital: {int(saldo)}
Fecha solicitada para la liquidación: {liquidation_date}

Por favor, procesar esta solicitud de liquidación y enviar el documento al cliente en un plazo de dos (2) días hábiles.

Atentamente,
ChatBot SAC DataPro
Sistema Automatizado de Atención al Cliente
    """

    # Email for client
    client_liquidacion = f"""
Estimado/a {nombre},

Hemos recibido su solicitud de liquidación para el crédito número {int(credito)}.

Número de radicado: {radicado}

Detalles de la solicitud:
Saldo a capital: {int(saldo)}
Fecha solicitada para la liquidación: {liquidation_date}

Nuestra área de Servicio al Cliente procesará su solicitud y le enviará la liquidación en un plazo estimado de dos (2) días hábiles a su correo electrónico registrado.

Si tiene alguna pregunta adicional, por favor mencione su número de radicado.

Atentamente,
ChatBot SAC DataPro
Finanzauto S.A. BIC
    """

    return {
        "sac_liquidacion": sac_liquidacion, 
        "client_liquidacion": client_liquidacion, 
        "radicado": radicado,
        "liquidation_date": liquidation_date
    }



def get_response(user_query, chat_history, relevant_docs):
    try:
        state = {
            "messages": [{"role": "user", "content": user_query}], # Ajuste basado en cómo se están pasando los datos
            "chat_history":chat_history,
            "user_query":user_query
        }
    except Exception as e:
        # st.write(e)
        pass
    
    # st.write(state)
    
    template = """
    Documentos relevantes: {relevant_docs}
    Eres el mejor agente Pscicologo del mundo y el contexto te va a ayudar a dar mejores respuestas con respecto a lo que las personas sienten.Siempre respondes con la intención de que el usuario se sienta mejor y pueda resolver sus problemas de sentirse mal a bien ese es tu objetivo, puedes usar emojis.:
    Chat histórico: {chat_history}
    Pregunta del usuario: {user_query}
    Sé breve en la respuesta.
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY_2'], model_name="llama3-70b-8192")
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_query": user_query,
        "relevant_docs": relevant_docs
    })

# Estado de la sesión
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola! ¿Como te sientes el dia de hoy?"),
    ]

# Estado de la sesión
if "chat_history" not in st.session_state:
    st.markdown("# Hola!👋🏻 como te sientes el dia de hoy?")

with col1:

    # Conversación
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
# Entrada del usuario
user_query = st.chat_input("Escriba acá sus intereses...")


def save_feeling_to_csv(chat_history, conversation_id, feeling):
    if len(chat_history) == 5:
        feeling_data = {
            "ConversationID": conversation_id,
            "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            "Feeling": feeling
        }

        feeling_df = pd.DataFrame([feeling_data])
        feeling_file = "sentimientos.csv"
        
        # Cargar el archivo existente si existe
        if os.path.exists(feeling_file):
            existing_df = pd.read_csv(feeling_file)
            # Concatenar y eliminar duplicados
            updated_df = pd.concat([existing_df, feeling_df]).drop_duplicates(subset=['ConversationID', 'Timestamp', 'Feeling'], keep='last')
            updated_df.to_csv(feeling_file, index=False)
        else:
            feeling_df.to_csv(feeling_file, index=False)

def save_conversation_to_csv(chat_history, conversation_id):
    conversation_data = []
    for message in chat_history:
        if isinstance(message, AIMessage):
            role = "AI"
        elif isinstance(message, HumanMessage):
            role = "Human"
        else:
            continue
        conversation_data.append({
            "ConversationID": conversation_id,
            "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            "Role": role,
            "Message": message.content
        })
    
    conversation_df = pd.DataFrame(conversation_data)
    conversation_file = "conversation_history.csv"
    
    # Cargar el archivo existente si existe
    if os.path.exists(conversation_file):
        existing_df = pd.read_csv(conversation_file)
        # Concatenar y eliminar duplicados
        updated_df = pd.concat([existing_df, conversation_df]).drop_duplicates(subset=['ConversationID', 'Timestamp', 'Role', 'Message'], keep='last')
        updated_df.to_csv(conversation_file, index=False)
    else:
        conversation_df.to_csv(conversation_file, index=False)



# En la parte donde procesas la entrada del usuario
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        query_processed = text_preprocess(user_query[:500])

        # Usa transform en lugar de fit_transform aquí
        reference_value_vector = tfidf_vectorizer.transform([query_processed])

        similarities = cosine_similarity(reference_value_vector, tfidf_matrix)[0]
        # Obtener los 2 documentos más relevantes
        top_2_indices = similarities.argsort()[-2:][::-1]
        
        chunks_df_top_2 = chunks_df.iloc[top_2_indices]
        
        relevant_docs = chunks_df_top_2["Chunk"].tolist()
            
        response = st.write_stream(get_response(
            user_query[:500], 
            str(st.session_state.chat_history[-5:])[:2000],
            relevant_docs=str(relevant_docs)
        ))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    if len(st.session_state.chat_history) == 5:
        try:
            feeling = get_feeling(user_query, st.session_state.chat_history)
            ai_feeling = feeling["feeling"]
            save_feeling_to_csv(st.session_state.chat_history, st.session_state.conversation_id, ai_feeling)
        except:
            st.write(feeling)
        
    # Guardar la conversación en un CSV sin duplicados
    save_conversation_to_csv(st.session_state.chat_history, st.session_state.conversation_id)