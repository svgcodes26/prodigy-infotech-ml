# 🛒 Customer Segmentation — K-Means Clustering
**Prodigy Infotech ML Internship · Task-02**

Groups retail store customers into segments based on their purchase history using K-Means clustering.

---

## 📁 Project Structure

```
customer_segmentation/
├── streamlit_app.py     ← Main app (pure Python)
├── requirements.txt     ← Dependencies
├── README.md
└── data/                ← Place Mall_Customers.csv here
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at: **http://localhost:8501**

---

## 📊 Using Real Kaggle Data

Download from:
> https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Place `Mall_Customers.csv` inside the `data/` folder.
The app will automatically detect and use it.

---

## 🧠 What the App Does

- Loads customer data (real or synthetic)
- Lets you choose which features to cluster on
- Lets you pick number of clusters (K)
- Shows the **Elbow Method** chart to find best K
- Plots **cluster scatter chart** with centroids
- Shows **cluster summary table**
- Lets you input a new customer and predict their segment
