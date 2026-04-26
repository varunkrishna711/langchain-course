from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

def main():
    print("Hello from langchain-course!")
    print(f"Your GOOGLE_API_KEY is: {os.getenv('GOOGLE_API_KEY')}")

    information = """
        Lata Dinanath Mangeshkar (Hindi pronunciation: [ləˈt̪aː məŋˈɡeːʃkəɾ] ⓘ; born Hema Dinanath Mangeshkar; 28 September 1929 – 6 February 2022)[9] was an Indian playback singer and occasional music composer. She is considered to be one of the greatest and most influential singers of the Indian subcontinent.[10][11][12][13] Her contribution to the Indian music industry in a career spanning eight decades gained her honorific titles such as the "Queen of Melody" and "Voice of the Millennium".[14]

        Mangeshkar recorded songs in over thirty-six Indian languages and a few foreign languages, though primarily in Hindustani, Bengali and Marathi.[14][15]

        Known for her discipline, Mangeshkar always recorded her songs barefoot as a mark of respect for the recording studio, which she considered a temple. Despite her perfectionism, she famously never listened to her own songs after their release, stating in interviews that she would only find mistakes in her singing.[16]
        """

    summary_template = """
        Given the information {information}, summarize these points:
        1. A short summary of the information.
        2. Two interesting facts about the person mentioned in the information.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    chain = summary_prompt_template | llm
    response = chain.invoke(input = {"information": information})
    print(response)

if __name__ == "__main__":
    main()
