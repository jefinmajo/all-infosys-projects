import streamlit as st

def main():
    # Sidebar with radio buttons
    st.sidebar.title("Options")
    option = st.sidebar.radio(
        "Select an input type:",
        ("PDF", "URL", "None")
    )

    st.title("Data Input Interface")

    if option == "PDF":
        st.subheader("Upload a PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if pdf_file is not None:
            st.success("PDF file uploaded successfully!")
            # Process the PDF here
    elif option == "URL":
        st.subheader("Enter a URL")
        url = st.text_input("Paste a URL below:")
        if url:
            st.success(f"URL entered: {url}")
            # Process the URL here
    else:
        st.subheader("Using Default Data")
        st.write("Default data will be processed.")

if __name__ == "__main__":
    main()
