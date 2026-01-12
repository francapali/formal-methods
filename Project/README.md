# 🎮 Steam Retention Analytics
### User Behavior, Purchasing Patterns, and Retention Analysis

![Project Status](https://img.shields.io/badge/Status-In_Development-yellow)
![Python](https://img.shields.io/badge/Language-Python-blue)
![Focus](https://img.shields.io/badge/Focus-E--commerce_%26_Retention-green)

## 📋 Project Description
This project analyzes the "Steam Video Games" dataset using a management and E-commerce approach. Instead of simply ranking popular games, the focus is on understanding the **User Lifecycle** within the platform.

We examine the relationship between the act of purchasing and actual product usage (playtime) to identify patterns of **Customer Churn**, **User Engagement**, and **Buying Behavior**.

## 🎯 Business Objectives
The project aims to answer the following key business questions:
* **Purchase-to-Play Ratio:** What is the conversion rate between buying a game and actually playing it?
* **Retention Analysis:** Is there a critical playtime threshold (e.g., "The Golden Hour") that predicts long-term loyalty?
* **Value Segmentation:** Can we segment users based on their value (e.g., "Collectors" who buy but don't play vs. "Hardcore Gamers")?

## 🗂️ Dataset
* **Source:** [Steam Video Games Dataset (Kaggle)](https://www.kaggle.com/datasets/tamber/steam-video-games)
* **Content:** Data regarding User IDs, Game Titles, Behaviors (purchase/play), and Total Playtime hours (where applicable).

## 🛠️ Tech Stack (Provisional)
* **Language:** Python
* **Core Libraries:** Pandas, NumPy
* **Data Viz:** Matplotlib, Seaborn, Plotly (for interactive dashboards)
* **Analytics:** Cohort Analysis, Modified RFM Analysis (Recency, Frequency, Monetary - adapted for playtime)


## 🚀 Roadmap
1.  **Data Ingestion & Cleaning:** Handling null values and normalizing game titles.
2.  **Feature Engineering:** Creating metrics such as "Average Playtime per User," "Purchase-to-Play Ratio," and "Engagement Score."
3.  **Exploratory Analysis (EDA):** Visualizing playtime distributions and purchase habits.
4.  **Advanced Analysis:** User Segmentation (K-Means Clustering) and Churn probability.
5.  **Reporting:** Final dashboard with operational insights.

---
*Project created for Formal Methods in CS - A.Y. 2025/2026*
