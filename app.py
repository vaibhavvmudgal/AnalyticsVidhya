import os
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
import pandas as pd

# Pinecone setup
PINECONE_API_KEY = "pcsk_73PRpV_CgWMY7AHHNkEg9fbvMuLVS2G85A22t8K4QPBw3UBd1ZvmMe472YJzKBB9XQCPX5"
INDEX_NAME = "drishti"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists, otherwise create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Adjust according to your embeddings' dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

# Access the Pinecone index
pc_index = pc.Index(INDEX_NAME)

# Streamlit app
st.title("Course Analytics with Pinecone")
st.markdown("---")

# Default database file (ensure this file exists in the same directory)
DEFAULT_DATABASE = "dataset.csv"

@st.cache_data
def process_file(file):
    """Process the default file and prepare it for vectorstore creation."""
    data = pd.read_csv(file)
    required_columns = {"Course_Name", "Course_Link", "Course_Description", "Course_Curriculum"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV file must contain these columns: {required_columns}")
    
    # Fill missing values with placeholders to avoid incomplete metadata
    data.fillna("N/A", inplace=True)

    # Break course curriculum into chunks
    data["chunks"] = data["Course_Curriculum"].apply(
        lambda text: [text[i:i + 200] for i in range(0, len(text), 200)] if isinstance(text, str) else []
    )
    return data

@st.cache_resource
def create_vectorstore(data):
    """Create a Pinecone vectorstore from the course data."""
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    metadata = []
    texts = []

    for _, row in data.iterrows():
        for chunk in row['chunks']:
            texts.append(chunk)
            metadata.append({
                'Course_Name': row.get('Course_Name', 'N/A'),
                'Course_Link': row.get('Course_Link', 'N/A'),
                'Course_Description': row.get('Course_Description', 'N/A'),
                'Course_Curriculum': chunk
            })

    st.write("Metadata Example:", metadata[0] if metadata else "No Metadata Found")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = LangchainPinecone.from_texts(
        texts=texts,
        embedding=embeddings.embed_query,
        index_name=INDEX_NAME,
        metadatas=metadata
    )
    return vectorstore

# Load and process the default database
try:
    processed_data = process_file(DEFAULT_DATABASE)

    # Data Preview with Customizable Row Limit
    st.write("**Preview Course Data**")
    num_rows = st.slider("Number of rows to display", min_value=5, max_value=len(processed_data), value=5, step=1)
    st.write(processed_data.head(num_rows))
    
    vectorstore = create_vectorstore(processed_data)
    st.success("Vectorstore created successfully!")

    query = st.text_input("Search courses")
    if query:
        # Check if user explicitly requests multiple courses
        user_wants_multiple = any(
            keyword in query.lower() for keyword in ["courses", "multiple", "list", "top"]
        )
        
        # Determine the number of results to fetch
        k = 3 if user_wants_multiple else 1  # Default to 1 for the best course
        
        # Perform similarity search
        search_results = vectorstore.similarity_search_with_score(query, k=k)
        
        if search_results:
            seen = set()
            for result, score in search_results:
                metadata = result.metadata
                # Ensure unique results
                if metadata.get('Course_Name') not in seen:
                    seen.add(metadata.get('Course_Name'))
                    st.write(f"**Course Name:** {metadata.get('Course_Name', 'N/A')}")
                    st.write(f"**Course Link:** {metadata.get('Course_Link', 'N/A')}")
                    st.write(f"**Course Description:** {metadata.get('Course_Description', 'N/A')}")
                    st.write(f"**Course Curriculum:** {metadata.get('Course_Curriculum', 'N/A')}")
                    st.markdown("---")
        else:
            st.warning("No relevant results found.")
except Exception as e:
    st.error(f"An error occurred: {e}")
