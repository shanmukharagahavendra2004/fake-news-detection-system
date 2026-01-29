import pandas as pd
import joblib
from src.preprocess import clean_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model=joblib.load("models/fake_news_model.pkl")
vectorizer=joblib.load("models/tfidf_vectorizer.pkl")

fake_df=pd.read_csv("./dataset/fake_news.csv")
true_df=pd.read_csv("./dataset/true_news.csv")

fake_df["label"]=0
true_df["label"]=1

df=pd.concat([fake_df,true_df],axis=0)
df=df.sample(frac=1,random_state=42).reset_index(drop=True)

df["cleaned_text"]=df["text"].apply(clean_text)

x=vectorizer.transform(df["cleaned_text"])
y=df["label"]

y_pred=model.predict(x)

print("Accuracy:",accuracy_score(y,y_pred))
print("\nClassification Report:\n",classification_report(y,y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y,y_pred))
