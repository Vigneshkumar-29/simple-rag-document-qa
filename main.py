import gradio as gr

from rag import (
    create_rag,
    retrieval_chunk,
    generate_answer
)


# Create RAG

vectordb = create_rag()


# Generate Answer

def generate_response(query):

    if not query.strip():
        return "", "Please enter a question."

    retrieval_text = retrieval_chunk(
        vectordb,
        query,
        k=3
    )

    response = generate_answer(
        retrieval_text,
        query
    )

    return response, "Answer generated successfully."


# Clear

def clear_all():

    return "", "", "Cleared."


# Gradio UI

with gr.Blocks(
    title="Document Question Answering"
) as demo:

    gr.Markdown(
        """
        # Document Question Answering

        Ask questions based on the information available
        in the document.
        """
    )


    query_box = gr.Textbox(
        label="Question",
        placeholder="Ask a question about the document...",
        lines=4
    )


    with gr.Row():

        generate_button = gr.Button(
            "Generate",
            variant="primary"
        )

        clear_button = gr.Button(
            "Clear"
        )

        stop_button = gr.Button(
            "Break"
        )


    answer_box = gr.Textbox(
        label="Answer",
        lines=12,
        interactive=False
    )


    status_box = gr.Textbox(
        label="Status",
        interactive=False
    )


    # Generate

    generate_event = generate_button.click(
        fn=generate_response,
        inputs=query_box,
        outputs=[
            answer_box,
            status_box
        ]
    )


    # Clear

    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            query_box,
            answer_box,
            status_box
        ]
    )


    # Break / Stop

    stop_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[generate_event]
    )


# Launch

if __name__ == "__main__":

    demo.launch(
        share=True
    )