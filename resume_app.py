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
            resume_text = resume_bytes.decode('latin-1')
        
        # function calling
        cleaned_resume = cleanResume(resume_text)

        # vectorize the text
        vectorizer_resume = vectorizer.transform([cleaned_resume])

        # calling the model for prediction
        prediction = model.predict(vectorizer_resume)[0]

        # show predicited value
        # st.write(prediction)

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            18: "ETL Developer",
            10: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
         }

        category_name = category_mapping.get(prediction, "Unknown")
        st.success(f"Predicted Category: {category_name}")






# python main
if __name__ == "__main__":
    main()

