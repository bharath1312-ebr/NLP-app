import streamlit as st

# NLP Packages
import spacy
import textblob 
from textblob import TextBlob
import gensim
from gensim.summarization.summarizer import summarize
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#Summaryfn
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer()
    summary_list = [str(sentence) for sentence in summary]
    result = ''.join(summary_list)
    return result

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)

    tokens = [token.text for token in docx]
    all_data = [('"Tokens":{},\n "Lemma":{}'.format(token.text, token.lemma_)) for token in docx]
    return all_data

def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)

    tokens = [token.text for token in docx]
    entities = [(entity.text,entity.label_) for entity in docx.ents]
    all_data = ['"Tokens":{}, \n "Entities:{}'.format(tokens,entities)]
    return all_data

#Pkgs


def main():
    """ Creating NLP App using Streamlit"""
    st.title("NLPiffy with Streamlit")
    st.subheader("Natural Processing Language on the Go")

    # Tokenization
    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize your text")
        message = st.text_area("Enter the text","Type Here.. ")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # Named Entity
    if st.checkbox("Show Named Entites"):
        st.subheader("Extract Entities From the Text")
        message = st.text_area("Enter the text","Type Here.. ")
        if st.button("Extract"):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)


    # Sentiment Analysis
    if st.checkbox("Sentiment Analysis"):
        st.subheader("Sentiment of Your Text")
        message = st.text_area("Enter the text","Type here.. ")
        if st.button("Analyze"):
           blob = TextBlob(message)
           result_sentiment = blob.sentiment
           st.success(result_sentiment)

    # Text Summarzation
    if st.checkbox("Text Summarization"):
        st.subheader("Summarize Your Text")
        message = st.text_area("Enter the text","type here.. ")
        summary_options = st.selectbox("Choice Your Summarizer",("gensim","sumy"))
        if st.button("Summarize"):
            if summary_options == "gensim" :
                st.text("Using Gensim")
                summary_result = summarize(message)
            elif summary_options == 'sumy' :
                st.text("Using Sumy")
                summary_result = sumy_summarizer(message)
            
            else:
                st.warning("Using Default Summarizer")
                st.text("Using Genism")
                summary_result = summarize(message)
            st.success(summary_result)
           



if __name__ == '__main__':
    main()