import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 불러오기
df = pd.read_csv("data/diabetes.csv")

# Step1: BMI 이상치 제거 (IQR)
Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 2.0 * IQR
upper_bound = Q3 + 2.0 * IQR
df = df[(df['BMI'] >= lower_bound) & (df['BMI'] <= upper_bound)]

# Step2: 특성과 타겟 변수 분리
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

# Step3: 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step5: 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step6: 특성 중요도
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# ==========================
# 🎨 Streamlit UI 디자인
# ==========================
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

# 사이드바 메뉴
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Go to", ["Dashboard", "Intro", "EDA", "Model Performance"])

# 대시보드 페이지
def dashboard():
    st.title("Diabetes Prediction Dashboard")
    st.markdown("""
    **이 대시보드는 당뇨병 예측 데이터를 분석하는 웹 앱입니다.**
    - 📌 **변수 설명**
      - **Pregnancies**: 임신 횟수
      - **Glucose**: 혈당 수치
      - **BloodPressure**: 혈압(mmHg)
      - **SkinThickness**: 피부 두께(mm)
      - **Insulin**: 인슐린 수치
      - **BMI**: 체질량지수
      - **DiabetesPedigreeFunction**: 가족력 지수
      - **Age**: 나이
      - **Outcome**: 당뇨병 여부 (0: 정상, 1: 당뇨병)
    """)

# 📖 소개 페이지
def intro():
    st.title("📖 소개 (Introduction)")
    st.markdown("""
    ## 당뇨병이란?
    당뇨병(Diabetes)은 혈액 내 혈당(Glucose) 수치가 정상보다 높아지는 만성 질환입니다. 
    이는 인슐린 생산 부족 또는 인슐린 저항성으로 인해 발생합니다. 
    당뇨병은 크게 두 가지 유형으로 나뉘며, 제1형 당뇨병(인슐린 의존성)과 제2형 당뇨병(인슐린 저항성)이 있습니다.

    ### 당뇨병이 중요한 이유
    - 전 세계적으로 빠르게 증가하는 질병으로, 조기 발견과 예방이 중요합니다.
    - 혈당 조절 실패 시 심혈관 질환, 신부전, 신경 손상 등의 합병증을 유발할 수 있습니다.
    - 생활 습관 개선과 조기 진단을 통해 당뇨병 예방 및 관리가 가능합니다.

    ## 이 웹 애플리케이션의 목적
    본 웹 애플리케이션은 당뇨병 예측을 위한 데이터 분석과 모델 성능 평가를 통해, 
    당뇨병 위험 요인을 이해하고 예측할 수 있도록 돕는 것을 목표로 합니다.

    **주요 기능:**
    - 📊 탐색적 데이터 분석 (EDA): 데이터의 특성을 시각적으로 분석
    - 🚀 머신러닝 기반 예측 모델: 랜덤 포레스트 모델을 활용한 당뇨병 예측
    - 📈 모델 성능 평가: 정확도, 분류 리포트 및 특성 중요도를 확인

    """)

# 📊 EDA (탐색적 데이터 분석)
def eda():
    st.title("📊 데이터 시각화")
    tab1, tab2, tab3 = st.tabs(["Histogram", "Boxplot", "Heatmap"])

    with tab1:
        st.subheader("📌 변수별 분포 (Histogram)")
        selected_col = st.selectbox("📊 분석할 변수를 선택하세요", 
                                    ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"])
        fig = px.histogram(df, x=selected_col, nbins=30, color="Outcome",
                           title=f"{selected_col} Histogram", barmode="overlay", marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📌 당뇨병 여부에 따른 Glucose Boxplot")
        fig = px.box(df, x="Outcome", y="Glucose", color="Outcome", 
                     title="당뇨병 여부에 따른 Glucose 분포", points="all")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📌 변수 간 상관관계 (Heatmap)")
        corr_matrix = df.corr(numeric_only=True)
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=np.round(corr_matrix.values, 2),
            colorscale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

# 🚀 모델 성능 평가
def model_performance():
    st.title("🚀 모델 성능 평가")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="🎯 모델 정확도 (Accuracy)", value=f"{accuracy:.2%}")
    st.subheader("📊 분류 리포트 (Classification Report)")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    st.subheader("📌 특성 중요도 (Feature Importance)")
    fig = px.bar(feature_importances, x="Feature", y="Importance", 
                 title="랜덤 포레스트 모델의 특성 중요도", color="Importance")
    st.plotly_chart(fig, use_container_width=True)

# 메뉴 선택에 따라 함수 실행
if menu == "Dashboard":
    dashboard()
elif menu == "Intro":
    intro()
elif menu == "EDA":
    eda()
elif menu == "Model Performance":
    model_performance()
