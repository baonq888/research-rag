from loader.pdf_loader import UnstructuredPDFLoader
import os
import base64
from IPython.display import Image, display
from loader.summarizer import Summarizer  
from dotenv import load_dotenv


def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))

def main():
    pdf_path = "./data/attention.pdf"
    image_output_dir = "./data"

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    loader = UnstructuredPDFLoader(
        file_path=pdf_path,
        image_output_dir=image_output_dir,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts, tables, images_b64 = loader.process_pdf_content()

    # Display document chunks
    print(f"Text chunks (Documents): {len(texts)}")
    for i, doc in enumerate(texts[:2]):
        print(f"\n--- Document Chunk {i+1} ---")
        print(doc.page_content[:500])


    # Display table summaries
    print(f"\nTables Extracted: {len(tables)}")
    for i, table in enumerate(tables[:1]):
        print(f"\n--- Table {i+1} ---")
        print(str(table)[:300])

    # Display image previews
    print(f"\nBase64 Images Extracted: {len(images_b64)}")
    if images_b64:
        print("Displaying first extracted image...")
        display_base64_image(images_b64[0])

    # Summarization
    summarizer = Summarizer()

    print("\nSummarizing text...")
    text_summaries = summarizer.summarize_text(texts)

    # print("\nSummarizing tables...")
    # tables_html = [table.metadata.text_as_html for table in tables]


    print("\nSummarizing images...")
    image_summaries = summarizer.summarize_images(images_b64)

    # Display summaries (optional)
    for i, summary in enumerate(text_summaries):
        print(f"\nText Summary {i + 1}:\n{summary}\n{'-'*50}")

    # for i, summary in enumerate(table_summaries[:1]):
    #     print(f"\nTable Summary {i+1}:\n{summary}")

    for i, summary in enumerate(image_summaries):
        print(f"\nImage Summary {i+1}:\n{summary}\n{'-'*50}")

if __name__ == "__main__":
    main()