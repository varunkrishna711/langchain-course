from dotenv import load_dotenv
import os

load_dotenv()


if __name__ == "__main__":
    print("Ingesting....")
    print(os.getenv("PINECONE_API_KEY"))
