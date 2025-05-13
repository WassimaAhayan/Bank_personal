import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Charger le modèle entraîné ---
model = joblib.load('Bank_Personal_Loan.pkl')

st.set_page_config(page_title="Bank Loan App", layout="wide")

# --- Interface en deux onglets ---
tab1, tab2 = st.tabs(["📈 Prédiction", "📊 Visualisation"])

# === ONGLET 1 : Prédiction ===
with tab1:
    st.title("🏦 Prédiction d'un Prêt Bancaire")

    st.markdown("**Veuillez remplir les informations suivantes :**")

    # Champs utilisateur : toutes les colonnes
    id_val = st.number_input("ID du client", min_value=0)
    age = st.slider("Âge", 18, 75, 30)
    experience = st.slider("Expérience (en années)", 0, 50, 5)
    income = st.number_input("Revenu annuel (en milliers $)", min_value=0)
    zip_code_input = st.text_input("Code postal (ZIP)", max_chars=10)
    try:
        zip_code = int(zip_code_input)
    except ValueError:
        st.error("❌ دخّل كود بوستال صحيح (بالأرقام).")
    family = st.selectbox("Taille de la famille", [1, 2, 3, 4])
    ccavg = st.number_input("Dépense moyenne carte de crédit (en milliers $)", min_value=0.0)
    education = st.selectbox("Niveau d'éducation", [1, 2, 3])  # 1=Undergrad, 2=Graduate, 3=Advanced
    mortgage = st.number_input("Montant de l'hypothèque", min_value=0)
    securities_account = st.checkbox("Compte titres")
    cd_account = st.checkbox("Compte CD")
    online = st.checkbox("Bancaire en ligne")
    credit_card = st.checkbox("Carte de crédit")

    if st.button("🔍 Prédire"):
        # Créer un DataFrame avec toutes les 13 colonnes
        input_data = pd.DataFrame([[
            id_val, age, experience, income, zip_code, family, ccavg, education,
            mortgage, int(securities_account), int(cd_account), int(online), int(credit_card)
        ]], columns=[
            'ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
            'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard'
        ])

        # Prédiction
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Ce client est **susceptible d'accepter** le prêt.")
        else:
            st.warning("❌ Ce client est **peu susceptible d'accepter** le prêt.")

# === ONGLET 2 : Visualisation ===
with tab2:
    st.title("📊 Visualisation des Données")

    try:
        df = pd.read_csv("Bank_Personal_Loan.csv")
        st.success("✅ Données chargées avec succès.")

        st.subheader("🧾 Aperçu complet des données")
        st.dataframe(df)  # عرض جميع البيانات

        st.subheader("📌 Statistiques descriptives")
        st.write(df.describe())  # إحصائيات عامة

        st.subheader("📊 Sélectionner une colonne pour afficher l'histogramme")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if numeric_cols:
            selected_col = st.selectbox("🧮 Choisissez une colonne numérique :", numeric_cols)

            fig, ax = plt.subplots()
            df[selected_col].hist(ax=ax, bins=20, color="skyblue", edgecolor="black")
            ax.set_title(f"Histogramme de {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Aucune colonne numérique trouvée pour générer un histogramme.")

        st.subheader("📈 Corrélation entre les variables")
        corr = df.corr(numeric_only=True)
        fig_corr, ax_corr = plt.subplots()
        im = ax_corr.matshow(corr, cmap="coolwarm")
        fig_corr.colorbar(im)
        ax_corr.set_xticks(range(len(corr.columns)))
        ax_corr.set_yticks(range(len(corr.columns)))
        ax_corr.set_xticklabels(corr.columns, rotation=90)
        ax_corr.set_yticklabels(corr.columns)
        ax_corr.set_title("Matrice de Corrélation", pad=20)
        st.pyplot(fig_corr)

    except FileNotFoundError:
        st.error("❌ Le fichier 'Bank_Personal_Loan.csv' est introuvable.")
