#Core Packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import river
matplotlib.use('Agg')
import seaborn as sns
import altair as alt
import ast
from datetime import datetime
from utils.get_text import get_text
from PIL import Image

#Welcome Banner
display = image = Image.open("img/valutico_teal.png")
display = np.array(display)
st.image(display, width=250)
st.title(":teal[NLP Sales Qualifier v2]")

#Create Subheader
st.subheader('''App created by Raj Shah''')
st.caption('''Predicts with English, Spanish and German languages''')

#Online ML Pkgs
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords,TFIDF
from river.compose import Pipeline

#Training Data

data = [("company valuation","Business Valuation Expert"),("valoración de la empresa","Business Valuation Expert"),("Unternehmensbewertung","Business Valuation Expert"),("business valuations","Business Valuation Expert"),("valoraciones de empresas","Business Valuation Expert"),("Unternehmensbewertungen","Business Valuation Expert"),("business mergers","M&A"),("fusiones de negocios","M&A"),("Unternehmensfusionen","M&A"),("business acquisitions","M&A"),("adquisiciones de empresas","M&A"),("Unternehmensakquisitionen","M&A"),("merger acquisitions","M&A"),("mergers acquisitions","M&A"),("fusiones adquisiciones","M&A"),("Fusionen Übernahmen","M&A"),("M&A","M&A"),("selling business","M&A"),("buying business","M&A"),("sell-side","M&A"),("buy-side","M&A"),("transaction advisory","M&A"),("transacción","M&A"),("Transaktion","M&A"),("M&A transactions","M&A"),("Divestures","M&A"),("Value Enhancement","M&A"),("buyouts","M&A"),("buyer","M&A"),("Business Broker","Business Broker"),("Corredora","Business Broker"),("Corredor","Business Broker"),("Maklerin","Business Broker"),("Makler","Business Broker"),("Investment bank","Investment Banking"),("Banco de inversiones","Investment Banking"),("Investmentbank","Investment Banking"),("Raise capital","Investment Banking"),("Kapital beschaffen","Investment Banking"),("Aumentar el capital","Investment Banking"),("Investment Banking","Investment Banking"),("business appraisal","Business Appraiser"),("tasación de negocios","Business Appraiser"),("Würdigung","Business Appraiser"),("private equity","Private Equity"),("Privates Eigenkapital","Private Equity"),("Capital privado","Private Equity"),("certified public accountants","Tax Advisors"),("contadora","Tax Advisors"),("contador","Tax Advisors"),("buchhalterin","Tax Advisors"),("buchhalter","Tax Advisors"),("Buchhaltung","Tax Advisors"),("Steuer","Tax Advisors"),("impuesto","Tax Advisors"),("Accounting and tax","Tax Advisors"),("tax planning","Tax Advisors"),("tax return preparation","Tax Advisors"),("organization tax","Tax Advisors"),("family office","Family Office Investor"),("Oficina familiar","Family Office Investor"),("Familienbüro","Family Office Investor"),("public equity","Public Equity"),("patrimonio publico","Public Equity"),("öffentliches","Public Equity"),("world equity","Public Equity"),("late stage VC","Venture Capital"),("Risikokapital","Venture Capital"),("capital de riesgo","Venture Capital"),("late stage venture","Venture Capital"),("commercial real estate","Real Estate Valuation Expert"),("Gewerbeimmobilien","Real Estate Valuation Expert"),("bienes raíces comerciales","Real Estate Valuation Expert"),("bienes raíces","Real Estate Valuation Expert"),("Immobilie","Real Estate Valuation Expert"),("real estate","Real Estate Valuation Expert"),("property","Real Estate Valuation Expert"),("propiedad comercial","Real Estate Valuation Expert"),("Eigentum","Real Estate Valuation Expert")]

#Model Building
model = Pipeline(('vectorizer',BagOfWords(lowercase=True)),('nv',MultinomialNB()))
for x,y in data:
    model = model.learn_one(x,y)
    
#Storage in a database
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

#Create Fxn from SQL
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS predictionTable(message TEXT,prediction TEXT,probability NUMBER, businessvaluationexpert_proba NUMBER, mna_proba NUMBER, businessbroker_proba NUMBER, investmentbanking_proba NUMBER, businessappraiser_proba NUMBER, privateequity_proba NUMBER, taxadvisors_proba NUMBER, familyofficeinvestor_proba NUMBER, publicequity_proba NUMBER, venturecapital_proba NUMBER, realestate_proba NUMBER,postdate DATE)')
    
def add_data(message,prediction,probability,businessvaluationexpert_proba,mna_proba,businessbroker_proba,investmentbanking_proba,businessappraiser_proba,privateequity_proba,taxadvisors_proba,familyofficeinvestor_proba,publicequity_proba,venturecapital_proba,realestate_proba,postdate):
    c.execute('INSERT INTO predictionTable(message,prediction,probability,businessvaluationexpert_proba,mna_proba,businessbroker_proba,investmentbanking_proba,businessappraiser_proba,privateequity_proba,taxadvisors_proba,familyofficeinvestor_proba,publicequity_proba,venturecapital_proba,realestate_proba,postdate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(message,prediction,probability,businessvaluationexpert_proba,mna_proba,businessbroker_proba,investmentbanking_proba,businessappraiser_proba,privateequity_proba,taxadvisors_proba,familyofficeinvestor_proba,publicequity_proba,venturecapital_proba,realestate_proba,postdate))
    conn.commit()
    
def view_all_data():
    c.execute("SELECT * FROM predictionTable")
    data = c.fetchall()
    return data


def main():
    menu = ["Home","Manage","Web Scraper","About"]
    create_table()
    
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        with st.form(key='mlform'):
            col1,col2 = st.columns([2,1])
            with col1:
                message = st.text_area("Message")
                submit_message = st.form_submit_button(label='Predict')
                
            with col2:
                st.write("Web-Based Machine Learning Qualifier")
                st.write("Predict Text as per defined Valutico Personas to Qualify Leads")
                
                
        if submit_message:
            prediction = model.predict_one(message)
            prediction_proba = model.predict_proba_one(message)
            probability = max(prediction_proba.values())
            postdate = datetime.now()
            #Add data to database
            add_data(message,prediction,probability,prediction_proba['Business Valuation Expert'],prediction_proba['M&A'],prediction_proba['Business Broker'],prediction_proba['Investment Banking'],prediction_proba['Business Appraiser'],prediction_proba['Private Equity'],prediction_proba['Tax Advisors'],prediction_proba['Family Office Investor'],prediction_proba['Public Equity'],prediction_proba['Venture Capital'],prediction_proba['Real Estate Valuation Expert'],postdate)
            st.success("Data Submitted")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("Original Text")
                st.write(message)
                
                st.success("Prediction")
                st.write(prediction)
                
                st.info("Persona")
                if prediction == "Business Valuation Expert":
                    st.markdown(":green[Green]")
                elif prediction == "M&A":
                    st.markdown(":green[Green]")
                elif prediction == "Tax Advisors":
                    st.markdown(":green[Green]")
                elif prediction == "Business Broker":
                    st.markdown(":orange[Orange]")
                elif prediction == "Business Appraiser":
                    st.markdown(":orange[Orange]")
                elif prediction == "Investment Banking":
                    st.markdown(":orange[Orange]")
                elif prediction == "Private Equity":
                    st.markdown(":orange[Orange]")
                elif prediction == "Family Office Investor":
                    st.markdown(":red[Red]")
                elif prediction == "Public Equity":
                    st.markdown(":red[Red]")
                elif prediction == "Venture Capital":
                    st.markdown(":red[Red]")
                elif prediction == "Real Estate Valuation Expert":
                    st.markdown(":red[Out of Scope]")
                
                st.success("Verdict")
                if prediction == "Business Valuation Expert":
                    st.markdown(":green[Qualified]")
                elif prediction == "M&A":
                    st.markdown(":green[Qualified]")
                elif prediction == "Tax Advisors":
                    st.markdown(":green[Qualified]")
                elif prediction == "Business Broker":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Business Appraiser":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Investment Banking":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Private Equity":
                    st.markdown(":orange[Qualified]")
                elif prediction == "Family Office Investor":
                    st.markdown(":red[Difficult Lead]")
                elif prediction == "Public Equity":
                    st.markdown(":red[Difficult Lead]")
                elif prediction == "Venture Capital":
                    st.markdown(":red[Difficult Lead]")
                elif prediction == "Real Estate Valuation Expert":
                    st.markdown(":red[Disqualified]")
                
            with res_col2:
                st.info("Probability")
                st.write(prediction_proba)
                
                #Plot of Probability
                df_proba = pd.DataFrame({'label':prediction_proba.keys(),'probability':prediction_proba.values()})
                st.dataframe(df_proba)
                #visualization
                fig = alt.Chart(df_proba).mark_bar().encode(x='label',y='probability')
                st.altair_chart(fig,use_container_width=True)
    
    elif choice == "Manage":
        st.subheader("Manage & Monitor Results")
        stored_data = view_all_data()
        new_df = pd.DataFrame(stored_data,columns=['message','prediction','probability','businessvaluationexpert_proba','mna_proba','businessbroker_proba','investmentbanking_proba','businessappraiser_proba','privateequity_proba','taxadvisors_proba','familyofficeinvestor_proba','publicequity_proba','venturecapital_proba','realestate_proba','postdate'])
        st.dataframe(new_df)
        new_df['postdate'] = pd.to_datetime(new_df['postdate'])
        
        #c = alt.Chart(new_df).mark_line().encode(x='minutes(postdate)',y='probability') #For Minutes
        c = alt.Chart(new_df).mark_line().encode(x='postdate',y='probability')
        st.altair_chart(c)
        
        c_businessvaluationexpert_proba = alt.Chart(new_df['businessvaluationexpert_proba'].reset_index()).mark_line().encode(x='businessvaluationexpert_proba',y='index')
        c_mna_proba = alt.Chart(new_df['mna_proba'].reset_index()).mark_line().encode(x='mna_proba',y='index')
        c_businessbroker_proba = alt.Chart(new_df['businessbroker_proba'].reset_index()).mark_line().encode(x='businessbroker_proba',y='index')
        c_investmentbanking_proba = alt.Chart(new_df['investmentbanking_proba'].reset_index()).mark_line().encode(x='investmentbanking_proba',y='index')
        c_businessappraiser_proba = alt.Chart(new_df['businessappraiser_proba'].reset_index()).mark_line().encode(x='businessappraiser_proba',y='index')
        c_privateequity_proba = alt.Chart(new_df['privateequity_proba'].reset_index()).mark_line().encode(x='privateequity_proba',y='index')
        c_taxadvisors_proba = alt.Chart(new_df['taxadvisors_proba'].reset_index()).mark_line().encode(x='taxadvisors_proba',y='index')
        c_familyofficeinvestor_proba = alt.Chart(new_df['familyofficeinvestor_proba'].reset_index()).mark_line().encode(x='familyofficeinvestor_proba',y='index')
        c_publicequity_proba = alt.Chart(new_df['publicequity_proba'].reset_index()).mark_line().encode(x='publicequity_proba',y='index')
        c_venturecapital_proba = alt.Chart(new_df['venturecapital_proba'].reset_index()).mark_line().encode(x='venturecapital_proba',y='index')
        c_realestate_proba = alt.Chart(new_df['realestate_proba'].reset_index()).mark_line().encode(x='realestate_proba',y='index')
        
        
        c1,c2 = st.columns(2)
        with c1:
            with st.expander("Business Valuation Expert Probability"):
                st.altair_chart(c_businessvaluationexpert_proba,use_container_width=True)
                
        with c2:
            with st.expander("M&A Probability"):
                st.altair_chart(c_mna_proba,use_container_width=True)
        
        c3,c4 = st.columns(2)
        with c3:
            with st.expander("Business Broker Probability"):
                st.altair_chart(c_businessbroker_proba,use_container_width=True)
                
        with c4:
            with st.expander("Investment Banking Probability"):
                st.altair_chart(c_investmentbanking_proba,use_container_width=True)
        
        c5,c6 = st.columns(2)
        with c5:
            with st.expander("Business Appraiser Probability"):
                st.altair_chart(c_businessappraiser_proba,use_container_width=True)
                
        with c6:
            with st.expander("Private Equity Probability"):
                st.altair_chart(c_privateequity_proba,use_container_width=True)
                
        c7,c8 = st.columns(2)
        with c7:
            with st.expander("Tax Advisors Probability"):
                st.altair_chart(c_taxadvisors_proba,use_container_width=True)
                
        with c8:
            with st.expander("Family Office Investor Probability"):
                st.altair_chart(c_familyofficeinvestor_proba,use_container_width=True)
                
        c9,c10 = st. columns(2)
        with c9:
            with st.expander("Public Equity Probability"):
                st.altair_chart(c_publicequity_proba,use_container_width=True)
        with c10:
            with st.expander("Venture Capital Probability"):
                st.altair_chart(c_venturecapital_proba,use_container_width=True)
                
        c11,c12 = st.columns(2)
        with c11:
            with st.expander("Real Estate Valuation Expert Probability"):
                st.altair_chart(c_realestate_proba,use_container_width=True)
                
        #with st.expander("Prediction Distribution"):
            #fig2 = plt.figure()
            #ax = sns.countplot(y='probability',data=new_df)
            #ax.bar_label(ax.containers[0],label_type='edge')
            #st.pyplot(fig2)
    #Web Scraper
    elif choice == "Web Scraper":
        st.subheader("Web Scraper")
        st.markdown('##### This option helps you to scrape, and extract the text and show only first 10 lines')
        URL = st.text_input("Enter the URL of the webpage you want to scrape")
        if URL is not None:
            if st.button("Scrape"):
                text = get_text(URL)
                df = pd.DataFrame(text.splitlines(),columns=["Webpage_text"],index=None)
                st.markdown('## Showing the first ten lines of the text')
                st.dataframe(df.head(10))
                st.info('''Download the text as a csv file if you like.''')
                st.download_button(label="Download the text as a csv file", data=df.to_csv(index=False, encoding='utf-8'),file_name='webpage_text.csv',mime='text/csv')
        else:
            st.warning("Please enter a valid URL")
        
    else:
        st.subheader("About")
        st.caption("NLP Sales Qualifier can be used for Web Scraping for Lead Sourcing. When you have identified potential leads, you can access their URL and web scrape their website using this program. The program instantly downloads a csv file, which you can copy-paste into the message section in Home screen - so it can start an analysis if this client Qualifies for sales outreach or sequencing. This program uses Naive Bayes to predict and classify the lead. This program also stores the data which was produced in the form of output and utilizes it as training data for new queries. For further information on this program, contact: r.shah@valutico.com")
    
if __name__ == '__main__':
    main()
    
