import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare Recommender",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 Skincare Recommender")
st.markdown("Find the right product for your skin — filtered by allergens you want to avoid.")
st.divider()

# ── LOAD PICKLES (data only — not the function) ───────────────
@st.cache_resource
def load_all():
    skincare        = pickle.load(open('skincare_df.pkl',     'rb'))
    tfidf           = pickle.load(open('tfidf.pkl',           'rb'))
    vectors         = pickle.load(open('vectors.pkl',         'rb'))
    ALLERGEN_GROUPS = pickle.load(open('allergen_groups.pkl', 'rb'))
    return skincare, tfidf, vectors, ALLERGEN_GROUPS

with st.spinner("Loading model..."):
    skincare, tfidf, vectors, ALLERGEN_GROUPS = load_all()

# ── RECOMMEND FUNCTION (defined here — same as your notebook) ─
def recommend3(skin_type, product_type, avoid_allergens=[], top_n=10):
    count      = 0
    user_query = (skin_type + ' ') * 4 + (product_type + ' ') * 2
    query_vec  = tfidf.transform([user_query])
    scores     = cosine_similarity(query_vec, vectors).flatten()
    ranked     = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        row = skincare.loc[idx]

        product_allergens = str(row['Allergens']).lower()
        is_safe = True

        for grp in avoid_allergens:
            keywords = ALLERGEN_GROUPS.get(grp, [])
            for kw in keywords:
                if kw in product_allergens:
                    is_safe = False

        if is_safe == False:
            continue

        results.append({
            'name':        row['Name'],
            'brand':       row['Brand'],
            'skin':        row['Skin_Types'],
            'type':        row['Product_Type'],
            'description': row['Description'],
            'ingredients': row['Key_Ingredients'],
            'allergens':   row['Allergens'],
            'time':        row['Time'],
            'price':       row['Price'],
            'score':       round(score, 4),
            'image':       str(row['image_name']),
        })

        count += 1
        if count >= top_n:
            break

    return results

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Skin Profile")

    skin_type = st.selectbox(
        "Skin Type",
        ["Dry", "Oily", "Combination", "Sensitive", "All skin types"]
    )

    product_type = st.selectbox(
        "Product Type",
        ["Serum", "Moisturiser", "Cleanser", "Toner/Essence",
         "Mask", "Sunscreen", "Exfoliant", "Eye Cream", "Face Oil"]
    )

    st.subheader("Allergens to Avoid")
    avoid_allergens = st.multiselect(
        label="Choose allergens",
        options=list(ALLERGEN_GROUPS.keys()),
        default=[],
        placeholder="None selected"
    )

    top_n = st.slider("Number of results", min_value=5, max_value=20, value=10)
    go    = st.button("Find Products", type="primary", use_container_width=True)

# ── RESULTS ───────────────────────────────────────────────────
if go:
    results = recommend3(skin_type, product_type, avoid_allergens, top_n)

    st.subheader(f"Results for **{skin_type}** skin · **{product_type}**")
    if avoid_allergens:
        st.caption(f"Avoiding: {', '.join(avoid_allergens)}")
    st.write(f"Showing **{len(results)}** products")
    st.divider()

    if not results:
        st.warning("No products found. Try removing some allergen filters.")
    else:
        for i, r in enumerate(results, 1):

            with st.container(border=True):

                st.markdown(f"### #{i} — {r['name']}")
                st.markdown(
                    f"**{r['brand']}** &nbsp;|&nbsp; {r['price']} &nbsp;|&nbsp; ⏰ {r['time']}",
                    unsafe_allow_html=True
                )
                st.divider()

                col_img, col1, col2 = st.columns([1, 1, 2])

                with col_img:
                    image_path = f"images/{r['image']}"
                    try:
                        st.image(image_path, width=150)
                    except:
                        st.caption("No image")

                with col1:
                    st.markdown("**Skin Type**")
                    st.write(r['skin'])

                    st.markdown("**Product Type**")
                    st.write(r['type'])

                    st.markdown("**Similarity Score**")
                    st.progress(float(r['score']), text=str(r['score']))

                with col2:
                    st.markdown("**Description**")
                    st.write(r['description'])

                    st.markdown("**Key Ingredients**")
                    ingredients = r['ingredients'].split('|')
                    clean_ingredients = ', '.join([ing.strip() for ing in ingredients])
                    st.write(clean_ingredients)

                    st.markdown("**Allergens in this product**")
                    st.caption(r['allergens'])

                st.write("")

# ── DEFAULT SCREEN ────────────────────────────────────────────
else:
    st.info("👈 Select your skin type and product type in the sidebar, then click **Find Products**.")

    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Pick your skin type**")
        st.write("Choose whether your skin is Dry, Oily, Combination or Sensitive.")
    with col2:
        st.markdown("**2. Pick a product type**")
        st.write("Choose what kind of product you're looking for.")
    with col3:
        st.markdown("**3. Filter allergens**")
        st.write("Select ingredients you're allergic to — those products will be excluded.")