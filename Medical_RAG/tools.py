from pydantic import BaseModel

class VectorStore(BaseModel):
    """
    A vectorstore contains information about various medical conditions 
    such as malaria, type 1 and type 2 diabetes, migraines, asthma, heart disease, 
    cancer, skin problems (eczema), allergies, cold and flu, hepatitis, HIV/AIDS, 
    mental health disorders (depression, anxiety, addiction), Alzheimer's, 
    Parkinson's, epilepsy, arthritis, osteoporosis, stroke, pain management, 
    and sexual health, among others.
    """
    query: str


class SearchEngine(BaseModel):
    """
    A search engine for retrieving additional medical information 
    from the web beyond what is stored in the vector database.
    """
    query: str
