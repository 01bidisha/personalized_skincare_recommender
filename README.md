# 🧴 Personalized Skincare Recommender

Why do some skincare products work for certain skin types while others cause reactions? I built a recommendation engine across 1000 skincare products to determine if the right product match is driven by skin type, product category, or ingredient safety.

---

## 🎯 The Core Mission

The goal was to move beyond basic filtering and build a system that actually understands a user's skin profile. This project focuses on recommending the most compatible skincare products while actively protecting users from allergens hidden across 261 unique ingredient strings.

---

## 🧠 Key Insights (The "So What?")

**The Allergen Problem:** Skincare datasets don't standardize allergen names — `Fragrance`, `Parfum`, `Perfume` and `Aroma` are all the same thing written differently. I grouped 261 raw allergen strings into 35 canonical categories so the filter actually works the way a user expects it to.

**The Similarity Trap:** Combining all features into one TF-IDF vector causes product type to dominate the similarity score — a Toner for Dry skin would rank above a Toner for Oily skin because the word "Toner" outweighed the skin type mismatch. I solved this by repeating Skin Type 4× and Product Type 2× in the feature vector to enforce the right priority.

**Content Over Popularity:** This is a pure content-based system — no ratings, no purchase history, no collaborative filtering. The recommendation is driven entirely by what the product is made of and who it is for.

---

## 🛠️ The Toolkit

**Data Processing:** Systematic renaming and null handling across 13 columns to ensure the TF-IDF pipeline received clean, consistent input.

**Recommendation Engine:** TF-IDF Vectorizer to convert product profiles into numeric vectors and Cosine Similarity to rank products by closeness to the user's skin profile.

**Allergen Logic:** A keyword-matching dictionary that maps canonical allergen group names to all their known variants — so selecting `Lavender Oil` catches `Lavandula Angustifolia`, `Lavender Extract` and everything in between.

**Deployment:** Streamlit app with pickle-based model loading for fast startup, product images, and a multiselect allergen filter covering all 35 groups.

---

## 📈 Future Roadmap

To take this further, the next step is incorporating user feedback loops — if a user marks a recommendation as unhelpful, the system should learn to down-rank that product profile. Adding ingredient function filtering (e.g. show only Anti-Acne or Brightening products) would also significantly improve precision for users who know what benefit they are targeting.

---

## 📂 Repository Contents

- `app4.py` — Streamlit web application with full UI and allergen filtering
- `recommendation_system_final.ipynb` — Complete ML pipeline from EDA to recommendation engine
- `FINAL_SKINCARE_DATASET_MAY_2026.xlsx` — 1000 skincare products across 10 product types and 13 feature columns
- `allergen_groups.pkl` — 35 grouped allergen categories mapped from 261 raw strings
- `skincare_df.pkl` — Cleaned and renamed dataframe ready for inference
- `tfidf.pkl` — Fitted TF-IDF vectorizer trained on the combined feature column
- `vectors.pkl` — Precomputed product feature matrix for fast similarity scoring
- `images/` — Product images mapped by index to each product in the dataset

---

## 🌐 Live Demo
👉 [personalized-skincare-recommender.streamlit.app](https://personalized-skincare-recommender.streamlit.app)

---

**Note:** This project isn't just about writing code — it's about understanding that a skincare recommendation without allergen awareness is incomplete. The real challenge was not building the similarity model, it was making the safety filter robust enough to catch every variant of an allergen hiding in plain sight.

## 👩‍💻 Author

**Bidisha Bhaduri**
GitHub: [@01bidisha](https://github.com/01bidisha)
