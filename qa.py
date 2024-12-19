import streamlit as st
from transformers import pipeline

def load_qa_model():
    """
    Initialize and load a pre-trained Question Answering model.
    
    This function sets up a machine learning pipeline that can extract 
    answers from a given context based on a specific question.
    
    Returns:
        A question-answering pipeline model or None if loading fails
    """
    try:
        # Use a pre-trained model specifically fine-tuned for question answering
        # 'deepset/bert-base-cased-squad2' is trained on the Stanford Question Answering Dataset (SQuAD)
        qa_pipeline = pipeline(
            "question-answering", 
            model="deepset/bert-base-cased-squad2"
        )
        return qa_pipeline
    except Exception as e:
        # If model loading fails, display an error message
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Configure the Streamlit page with a title and icon
    st.set_page_config(page_title="Question Answering System")
    
    # Main title of the application
    st.title("🤖 Question Answering Assistant")
    
    # Load the Question Answering model
    # This is the core AI component that will extract answers
    qa_pipeline = load_qa_model()
    
    # Stop the app if model loading fails
    if qa_pipeline is None:
        st.error("Failed to load the Question Answering model. Please check your installation.")
        st.stop()
    
    # Sidebar with usage instructions
    st.sidebar.header("How to Use")
    st.sidebar.info("""
    1. Enter a context paragraph in the text area
    2. Ask a specific question about the context
    3. Click 'Get Answer' to see the results
    """)
    
    # Context input section
    # Provides a default context about Albert Einstein
    st.header("📝 Context")
    context = st.text_area(
        "Enter your context paragraph:", 
        value="Albert Einstein was a renowned physicist who developed the theory of relativity. "
               "Born in Germany in 1879, he is best known for his mass-energy equivalence formula E = mc². "
               "Einstein won the Nobel Prize in Physics in 1921 for his services to theoretical physics.",
        height=200
    )
    
    # Question input section
    st.header("❓ Your Question")
    question = st.text_input("What would you like to know?", 
                              placeholder="E.g., When did Einstein win the Nobel Prize?")
    
    # Answer generation button
    if st.button("🔍 Get Answer", type="primary"):
        # Check if both context and question are provided
        if context and question:
            try:
                # Use the AI model to find the answer in the context
                result = qa_pipeline(question=question, context=context)
                
                # Display the extracted answer
                st.subheader("📌 Answer:")
                st.success(result['answer'])
                
                # Show confidence score of the answer
                confidence = result.get('score', 0)
                st.metric(label="Confidence", value=f"{confidence:.2%}")
                
                # Provide a warning for low-confidence answers
                if confidence < 0.5:
                    st.warning("The confidence is relatively low. The answer might not be entirely accurate.")
            
            except Exception as e:
                # Handle any errors during answer extraction
                st.error(f"Error processing your question: {e}")
        else:
            # Prompt user to provide both context and question
            st.warning("Please provide both a context and a question.")
    
    # Footer with attribution
    st.markdown("---")
    st.markdown("*Presented to you  by Shamitha Reddy N and Team (Powered by Hugging Face Transformers*)")

# Entry point of the application
if __name__ == "__main__":
    main()