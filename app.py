import streamlit as st
import pandas as pd
import joblib

# 1. تحميل الموديل والأدوات المحفوظة
model = joblib.load('weather_model.pkl')
imputer = joblib.load('imputer.pkl')
le_summary = joblib.load('le_summary.pkl')
le_precip = joblib.load('le_precip.pkl')

st.title("☀️ تطبيق توقع درجة الحرارة البيئية")

# 2. إنشاء واجهة المستخدم لأخذ المدخلات
humidity = st.slider("نسبة الرطوبة (Humidity)", 0.0, 1.0, 0.5)
wind_speed = st.number_input("سرعة الرياح (Wind Speed km/h)", min_value=0.0, max_value=100.0, value=10.0)
pressure = st.number_input("الضغط الجوي (Pressure millibars)", min_value=0.0, max_value=1100.0, value=1015.0)

# المدخلات النصية يتم عرضها كقائمة خيارات بناءً على الأشكال التي تدرب عليها الموديل
summary_options = list(le_summary.classes_)
summary_choice = st.selectbox("وصف الطقس (Summary)", summary_options)

precip_options = list(le_precip.classes_)
precip_choice = st.selectbox("نوع الهطول (Precip Type)", precip_options)

# 3. زر التوقع
if st.button("توقع درجة الحرارة"):
    # تحويل المدخلات النصية إلى أرقام كما فعلنا في التدريب
    summary_encoded = le_summary.transform([summary_choice])[0]
    precip_encoded = le_precip.transform([precip_choice])[0]
    
    # تجميع المدخلات في مصفوفة ممررة للموديل بنفس ترتيب الأعمدة
    user_data = pd.DataFrame([[humidity, wind_speed, pressure, summary_encoded, precip_encoded]], 
                             columns=['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Summary', 'Precip Type'])
    
    # التوقع وعرض النتيجة
    prediction = model.predict(user_data)
    st.success(f"🌡️ درجة الحرارة المتوقعة هي: {prediction[0]:.2f} درجة مئوية")
