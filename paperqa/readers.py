from langchain.text_splitter import RecursiveCharacterTextSplitter

TextSplitter = RecursiveCharacterTextSplitter


def parse_pdf(path, citation, key, chunk_chars=4000, overlap=50):
    """
    Parse a PDF file into chunks.

    Parameters
    ----------
    path : str
        The path to the PDF file.
    citation : str
        The citation for the document.
    key : str
        The key for the document.
    chunk_chars : int, optional
        The number of characters to include in each chunk.
    overlap : int, optional
        The number of characters to overlap between chunks.
        
    """
    # Import the pypdf library
    import pypdf

    # Open the pdf file
    pdfFileObj = open(path, "rb")
    # Read the pdf file
    pdfReader = pypdf.PdfReader(pdfFileObj)
    # Initialize the split
    splits = []
    # Initialize the split
    split = ""
    # Initialize the page list
    pages = []
    # Initialize the metadata
    metadatas = []
    # Loop through the pages in the pdf
    for i, page in enumerate(pdfReader.pages):
        # Add the text from the page to the split
        split += page.extract_text()
        # Add the page to the page list
        pages.append(str(i + 1))
        # If the length of the split is greater than the chunk size
        if len(split) > chunk_chars:
            # Add the split to the list of splits
            splits.append(split[:chunk_chars])
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            # Create a string for the page range
            pg = "-".join([pages[0], pages[-1]])
            # Add the citation, key, and page range to the metadata
            metadatas.append(
                dict(
                    citation=citation,
                    dockey=key,
                    key=f"{key} pages {pg}",
                )
            )
            # Set the split to the last overlap characters of the split
            split = split[chunk_chars - overlap:]
            # Reset the page list
            pages = [str(i + 1)]
    # Close the pdf file
    pdfFileObj.close()
    # Return the splits and metadata
    return splits, metadatas


def parse_txt(path, citation, key, chunk_chars=4000, overlap=50):

    try:
        with open(path) as f:
            doc = f.read()
    except UnicodeDecodeError as e:
        with open(path, encoding="utf-8", errors="ignore") as f:
            doc = f.read()
    # yo, no idea why but the texts are not split correctly
    text_splitter = TextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    texts = text_splitter.split_text(doc)
    return texts, [dict(citation=citation, dockey=key, key=key)] * len(texts)


def read_doc(path, citation, key, chunk_chars=4000, overlap=50, disable_check=False):
    """Parse a document into chunks."""
    if path.endswith(".pdf"):
        # Parse the PDF file into chunks
        return parse_pdf(path, citation, key, chunk_chars, overlap)
    elif path.endswith(".txt"):
        # Parse the text file into chunks
        return parse_txt(path, citation, key, chunk_chars, overlap)
    else:
        # Raise an error if the file type is not supported
        raise ValueError(f"Unknown file type: {path} (expected .pdf or .txt).")
