from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()

URLS = [
    "https://www.webmd.com/epilepsy/default.htm",
    "https://www.webmd.com/arthritis/default.htm",
    "https://www.webmd.com/osteoporosis/default.htm",
    "https://www.webmd.com/a-to-z-guides/malaria",
    "https://www.webmd.com/diabetes/type-1-diabetes",
    "https://www.webmd.com/diabetes/type-2-diabetes",
    "https://www.webmd.com/migraines-headaches/migraines-headaches-migraines",
    "https://www.webmd.com/asthma/guide/asthma",
    "https://www.webmd.com/heart-disease/heart-disease-default",
    "https://www.webmd.com/cancer/default.htm",
    "https://www.webmd.com/skin-problems-and-treatments/eczema/default.htm",
    "https://www.webmd.com/allergies/default.htm",
    "https://www.webmd.com/cold-and-flu/cold-guide/default.htm",
    "https://www.webmd.com/cold-and-flu/flu-guide/default.htm",
    "https://www.webmd.com/hepatitis/hepatitis-c/default.htm",
    "https://www.webmd.com/hiv-aids/default.htm",
    "https://www.webmd.com/mental-health/addiction/default.htm",
    "https://www.webmd.com/depression/default.htm",
    "https://www.webmd.com/mental-health/anxiety-disorders/default.htm",
    "https://www.webmd.com/alzheimers/default.htm",
    "https://www.webmd.com/parkinsons-disease/default.htm",
    "https://www.webmd.com/stroke/default.htm",
    "https://www.webmd.com/pain-management/default.htm",
    "https://www.webmd.com/sexual-conditions/default.htm",
]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME="Llama3-70b-8192"
EMBEDDING_FUNCTION = HuggingFaceEmbeddings()
PERSIST_DIR = "db/chroma_store" 