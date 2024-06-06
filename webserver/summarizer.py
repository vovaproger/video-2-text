from transformers import pipeline

def summarize_text(transcript, max_chunk_length=512):

    assert transcript != None, "Please trascribe the video first."

    # Split the text into chunks to avoid IndexError
    chunks = [transcript[i:i + max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]
    summaries = []

    # Load the summarization pipeline
    summarizer = pipeline("summarization", model="google/pegasus-xsum")

    for chunk in chunks:
        # Generate summary for each chunk
        summary = summarizer(chunk, max_length=60, min_length=20, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Output the summary
    return " ".join(summaries)


