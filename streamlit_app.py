import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="제주감귤 당도 예측기", layout="wide")

# 제목
st.title("🍊 제주감귤 당도 예측기")
st.write("슬라이더를 조정하고 예측하기 버튼을 누르면 당도(Brix)를 예측해드립니다.")

# 모델 로드
@st.cache_resource
def load_model():
    model = joblib.load('brix_model.joblib')
    return model

@st.cache_resource
def load_poly():
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(np.array([[0, 0, 0]]))
    return poly

model = load_model()
poly = load_poly()

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# 메인 레이아웃: 왼쪽 슬라이더, 오른쪽 결과
left_col, right_col = st.columns([1, 1], gap="large")

# ========== 왼쪽: 슬라이더 입력 ==========
with left_col:
    st.subheader("📊 환경 변수 조정")
    
    # 최저기온 슬라이더
    min_temp = st.slider(
        "최저기온 (℃)",
        min_value=-6.9,
        max_value=35.3,
        value=10.0,
        step=0.1
    )
    
    # 최고기온 슬라이더 - 최저기온보다 커야 함
    max_temp_min = min(min_temp + 0.1, 35.3)  # 최저기온보다 0.1도 이상 높아야 함
    max_temp = st.slider(
        "최고기온 (℃)",
        min_value=-6.9,
        max_value=35.3,
        value=min(max(max_temp_min, 20.0), 35.3),
        step=0.1
    )
    
    # 가조시간 슬라이더
    sunshine_hours = st.slider(
        "가조시간 (시간)",
        min_value=9.9,
        max_value=14.4,
        value=12.0,
        step=0.1
    )
    
    # 예측 버튼
    if st.button("🔮 예측하기", use_container_width=True, key="predict_btn"):
        # 입력값 검증
        if min_temp > max_temp:
            st.error("⚠️ 최저기온이 최고기온보다 클 수 없습니다!")
        else:
            try:
                # 입력 데이터 변환 (2차 다항식)
                input_data = np.array([[max_temp, min_temp, sunshine_hours]])
                input_data_transformed = poly.transform(input_data)
                
                # 모델 예측
                prediction = model.predict(input_data_transformed)[0]
                
                # 당도 등급 판정
                if prediction >= 13:
                    grade = "최고급"
                elif prediction >= 11:
                    grade = "우수"
                elif prediction >= 9:
                    grade = "양호"
                else:
                    grade = "보통"
                
                # 히스토리에 추가
                history_item = {
                    "시간": datetime.now().strftime("%H:%M:%S"),
                    "최저기온": f"{min_temp}℃",
                    "최고기온": f"{max_temp}℃",
                    "가조시간": f"{sunshine_hours}시간",
                    "예상 Brix": f"{prediction:.2f}",
                    "등급": grade
                }
                st.session_state.history.insert(0, history_item)
                st.success("✅ 예측이 완료되었습니다!")
                
            except Exception as e:
                st.error(f"❌ 예측 중 오류가 발생했습니다: {str(e)}")

# ========== 오른쪽: 예측 결과 ==========
with right_col:
    st.subheader("📈 예측 결과")
    
    # 마지막 예측 결과 표시
    if st.session_state.history:
        latest = st.session_state.history[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="예상 당도 (Brix)",
                value=latest["예상 Brix"]
            )
        
        with col2:
            # 등급에 맞는 이모지
            grade = latest["등급"]
            if grade == "최고급":
                grade_display = "🌟 최고급"
            elif grade == "우수":
                grade_display = "⭐ 우수"
            elif grade == "양호":
                grade_display = "✅ 양호"
            else:
                grade_display = "⚪ 보통"
            
            st.metric(
                label="등급",
                value=grade_display
            )
        
        st.divider()
        st.caption("📋 현재 설정")
        st.write(f"- **최저기온**: {latest['최저기온']}")
        st.write(f"- **최고기온**: {latest['최고기온']}")
        st.write(f"- **가조시간**: {latest['가조시간']}")
    else:
        st.info("💡 슬라이더를 조정하고 '예측하기' 버튼을 누르세요.")

# ========== 아래쪽: 히스토리 ==========
st.divider()
st.subheader("📜 예측 히스토리")

if st.session_state.history:
    # 히스토리 데이터프레임 생성
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    # 초기화 버튼
    if st.button("🗑️ 히스토리 초기화", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    # ========== 산점도 ==========
    st.divider()
    st.subheader("📊 예측 결과 분석")
    
    # 데이터 전처리
    plot_data = []
    for item in st.session_state.history:
        min_temp_val = float(item["최저기온"].replace("℃", ""))
        plot_data.append({
            "최저기온": min_temp_val,
            "최고기온": float(item["최고기온"].replace("℃", "")),
            "가조시간": float(item["가조시간"].replace("시간", "")),
            "예상 Brix": float(item["예상 Brix"]),
            "등급": item["등급"]
        })
    
    df_plot = pd.DataFrame(plot_data)
    # y축 범위(예상 Brix) 확장: 최소/최대에 패딩 추가
    if not df_plot.empty:
        brix_min = df_plot["예상 Brix"].min()
        brix_max = df_plot["예상 Brix"].max()
        padding = max(1.0, (brix_max - brix_min) * 0.2)
        y_range = [max(0, brix_min - padding), brix_max + padding]
    else:
        y_range = [0, 15]
    
    # 3개의 산점도를 한 줄에 표시
    col1, col2, col3 = st.columns(3)
    
    # 1. 최저기온 vs 당도
    with col1:
        fig1 = px.scatter(
            df_plot,
            x="최저기온",
            y="예상 Brix",
            color="예상 Brix",
            title="최저기온 vs 당도",
            labels={
                "최저기온": "최저기온 (℃)",
                "예상 Brix": "예상 Brix"
            },
            color_continuous_scale="Viridis"
        )
        fig1.update_layout(
            height=400,
            hovermode="closest",
            font=dict(size=10),
            showlegend=False
        )
        fig1.update_yaxes(range=y_range)
        st.plotly_chart(fig1, use_container_width=True)
    
    # 2. 최고기온 vs 당도
    with col2:
        fig2 = px.scatter(
            df_plot,
            x="최고기온",
            y="예상 Brix",
            color="예상 Brix",
            title="최고기온 vs 당도",
            labels={
                "최고기온": "최고기온 (℃)",
                "예상 Brix": "예상 Brix"
            },
            color_continuous_scale="Viridis"
        )
        fig2.update_layout(
            height=400,
            hovermode="closest",
            font=dict(size=10),
            showlegend=False
        )
        fig2.update_yaxes(range=y_range)
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. 가조시간 vs 당도
    with col3:
        fig3 = px.scatter(
            df_plot,
            x="가조시간",
            y="예상 Brix",
            color="예상 Brix",
            title="가조시간 vs 당도",
            labels={
                "가조시간": "가조시간 (시간)",
                "예상 Brix": "예상 Brix"
            },
            color_continuous_scale="Viridis"
        )
        fig3.update_layout(
            height=400,
            hovermode="closest",
            font=dict(size=10),
            showlegend=False
        )
        fig3.update_yaxes(range=y_range)
        st.plotly_chart(fig3, use_container_width=True)
    
else:
    st.info("💡 값을 조정하고 '예측하기' 버튼을 눌러 히스토리에 기록하세요.")

# 하단 안내
st.divider()
st.caption("💡 이 예측기는 다항선형회귀 모델을 기반으로 합니다.")
