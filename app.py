import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Prédiction Prêt Bancaire",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)


model = joblib.load("logistic_model.pkl")


df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

categorical_cols = ["Employment_Status", "Marital_Status", "Property_Ownership", "Loan_Purpose", "Previous_Defaults"]


encoders = {}
category_mappings = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    category_mappings[col] = le.classes_


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stNumberInput, .stSelectbox {
        margin-bottom: 1.2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }
    
    .result-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .approved {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
    }
    
    .rejected {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }
    
    .info-box {
        padding: 1rem;
        background: #fff;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="header">
        <h1>🏦 Prédiction d'Approbation de Prêt</h1>
        <p>Analysez la solvabilité de vos clients en temps réel</p>
    </div>
""", unsafe_allow_html=True)


with st.expander("ℹ️ Comment utiliser cette application"):
    st.markdown("""
        **Remplissez simplement le formulaire avec les informations du client :**
        1. Entrez les données personnelles et financières
        2. Sélectionnez les caractéristiques du prêt et cliquer sur 'Prédire' pour obtenir le résultat
        3. Pour en  savoir plus sur la méthodologie vous pouvez cliquez sur ce lien: (https://github.com/kadidiatousow/Pret_Bancaire/blob/main/loan_approve.ipynb)
    """)


with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Informations Personnelles")
        age = st.number_input("Âge", min_value=18, max_value=100, value=30, help="Âge du demandeur")
        income = st.number_input("Revenu annuel ($)", min_value=0, value=50000, step=1000, format="%d")
        marital_status = st.selectbox("État civil", category_mappings["Marital_Status"])
        number_of_dependents = st.number_input("Personnes à charge", min_value=0, max_value=10, value=0)
        
        st.subheader("📊 Historique Crédit")
        credit_score = st.number_input("Score de crédit", min_value=300, max_value=850, value=650)
        previous_defaults = st.selectbox("Défauts de paiement", category_mappings["Previous_Defaults"])

    with col2:
        st.subheader("💼 Détails du Prêt")
        loan_amount = st.number_input("Montant du prêt ($)", min_value=0, value=100000, step=1000, format="%d")
        loan_term = st.number_input("Durée (mois)", min_value=6, max_value=360, value=120)
        interest_rate = st.number_input("Taux d'intérêt (%)", min_value=0.0, max_value=100.0, value=5.0)
        debt_to_income_ratio = st.slider("Ratio dette/revenu", 0.0, 1.0, 0.3, 0.01)
        
        st.subheader("🏢 Situation Professionnelle")
        employment_status = st.selectbox("Statut d'emploi", category_mappings["Employment_Status"])
        property_ownership = st.selectbox("Propriété", category_mappings["Property_Ownership"])
        loan_purpose = st.selectbox("But du prêt", category_mappings["Loan_Purpose"])

    submitted = st.form_submit_button("🚀 Lancer l'analyse de crédit")


if submitted:
    
    employment_status_value = encoders["Employment_Status"].transform([employment_status])[0]
    marital_status_value = encoders["Marital_Status"].transform([marital_status])[0]
    property_ownership_value = encoders["Property_Ownership"].transform([property_ownership])[0]
    loan_purpose_value = encoders["Loan_Purpose"].transform([loan_purpose])[0]
    previous_defaults_value = encoders["Previous_Defaults"].transform([previous_defaults])[0]

    
    input_data = pd.DataFrame([[
        age, income, credit_score, loan_amount, loan_term, interest_rate,
        employment_status_value, debt_to_income_ratio, marital_status_value,
        number_of_dependents, property_ownership_value, loan_purpose_value,
        previous_defaults_value
    ]], columns=[
        "Age", "Income", "Credit_Score", "Loan_Amount", "Loan_Term", "Interest_Rate",
        "Employment_Status", "Debt_to_Income_Ratio", "Marital_Status",
        "Number_of_Dependents", "Property_Ownership", "Loan_Purpose", "Previous_Defaults"
    ])

    prediction = model.predict(input_data)[0]
    result = "✅ Prêt Approuvé" if prediction == 1 else "❌ Prêt Refusé"
    
   
    result_class = "approved" if prediction == 1 else "rejected"
    st.markdown(f"""
        <div class="result-box {result_class}">
            <h3>{result}</h3>
            <p>Probabilité d'approbation: {model.predict_proba(input_data)[0][1]*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

    
    with st.expander("📊 Comprendre la décision"):
        st.markdown("""
            **Facteurs clés influençant la décision :**
            - Score de crédit
            - Ratio dette/revenu
            - Historique de remboursement
            - Stabilité financière
            
            *Les résultats sont basés sur un modèle d'apprentissage automatique entraîné sur des données historiques.*
        """)

