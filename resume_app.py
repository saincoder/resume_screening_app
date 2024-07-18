import streamlit as st
import pickle
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# loading the model
model = pickle.load(open('model.pkl', 'rb'))

# loading the vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


# cleaning (urls, hashtags, mentions, special letters, punctuation) from text
def cleanResume(text):
  cleanText = re.sub('http\S+\s',' ',text)
  cleanText = re.sub('RT|cc',' ',cleanText)
  cleanText = re.sub('@\S+', ' ', cleanText)
  cleanText = re.sub('#\S+', ' ', cleanText)
  cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
  cleanText = re.sub(r'[^x00-x7f]', ' ', cleanText)
  cleanText = re.sub('\s+', ' ', cleanText)
  return cleanText


# webapp UI
# function
def main():
    st.title('Resume Screening App')
    st.write('Developed by Shahid')
    st.write('Upload your resume to see the predicted catogery')

    # upload the resume file
    upload_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

    # checks
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')

        except UnicodeDecodeError:
            # if utf-8 decoding fails, try decode with latin-1
            resume_text = resume_bytes.decode('latine-1')





# python main
if __name__ == "__main__":
    main()

