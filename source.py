import streamlit as st
import pandas as pd
import numpy as np

def load_features():
    final_data = pd.read_csv('final_data.csv')
    X=final_data.drop("DEPARTURE_DELAY",axis=1)
    # st.info(X.columns)
    return X

def accept_data():
    month = st.number_input("Enter month ",min_value=1,max_value=12)
    day = st.number_input("Enter day",min_value=1,max_value=31)
    sch_dept = st.number_input("Enter scheduled departure")
    distance = st.number_input("Enter distance in miles")
    arrival_delay = st.number_input("Enter arrival delay (Enter negative value if arrival is delayed else enter positive value)")
    airline = st.text_area("Enter airline code in place of XX","AIRLINE_XX")
    origin = st.text_area("Enter origin airport code in place of XXX","ORIGIN_AIRPORT_XX")
    destination = st.text_area("Enter destination airport code in place of XXX","DESTINATION_AIRPORT_XX")
    day_of_week = st.text_area("Enter day of week in place of XX","XXDAY")
    return month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week

def predict(MONTH, DAY,SCHEDULED_DEPARTURE,DISTANCE, ARRIVAL_DELAY,AIRLINE,ORIGIN_AIRPORT,DESTINATION_AIRPORT,DAY_OF_WEEK,model):
    X = load_features()

    AIRLINE_index = np.where(X.columns==AIRLINE)[0][0]
    ORIGIN_index = np.where(X.columns==ORIGIN_AIRPORT)[0][0]
    DESTINATION_index = np.where(X.columns==DESTINATION_AIRPORT)[0][0]
    DAY_OF_WEEK_index = np.where(X.columns==DAY_OF_WEEK)[0][0]
    x= np.zeros(len(X.columns))
    x[0] = MONTH
    x[1] = DAY
    x[2] = SCHEDULED_DEPARTURE
    x[3] = DISTANCE
    x[4] = ARRIVAL_DELAY
    if AIRLINE_index >=0:
        x[AIRLINE_index] = 1
    if ORIGIN_index >=0:
        x[ORIGIN_index] = 1
    if DESTINATION_index >=0:
        x[DESTINATION_index] = 1
    if  DAY_OF_WEEK_index >= 0:
        x[ DAY_OF_WEEK_index] = 1

    return model.predict([x])[0]

def main():
    st.title("Flight Delay Prediction")
    st.subheader("Prediction using Machine Learning Algorithm")


    import joblib
    rf_model = joblib.load('1_rf_model.pkl')
    dt_model = joblib.load('2_dt_model.pkl')
    xgb_model = joblib.load('3_xgb_model.pkl')
    lr_model = joblib.load('4_lr_model.pkl')

    month, day, sch_dept, distance, arrival_delay, airline, origin, destination, day_of_week = accept_data()

    if st.button("Predict Delay"):
        tab1, tab2, tab3, tab4 = st.tabs(['Random Forest','Decision Tree','XGBoost','Linear Regression'])

        with tab1:
            # res = prediction(X,month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week,rf_model)
            res = predict(5, 6, 1515, 328, -8.0, 'AIRLINE_OO', 'ORIGIN_AIRPORT_PHX', 'DESTINATION_AIRPORT_ABQ',
                          'DAY_OF_WEEK_TUESDAY', rf_model)
            st.subheader("Random Forest MODEL PREDICTION")
            if (res >= 0):
                text1 = "Flight is **not delayed**. It will depart for next flight at scheduled time."
                st.success(text1)
            elif (res >= -15):
                text2 = "Flight is only delayed by " + str(
                    abs(res)) + ". Delays upto 15 minutes are considered as not delay. **FLIGHT IS NOT DELAYED**"
                st.warning(text2)
            else:
                text3 = "**FLIGHT IS DELAYED** by " + str(
                    res) + ". Delays by more than 15 minutes are considered to be actual delays. **FLIGHT IS DELAYED**"
                st.error(text3)

            st.subheader('Random Forest MODEL PARAMETERS')
            st.info("MAE : 6.10")
            st.error("MSE : 103.91")
            st.success("RMSE : 10.19")
            st.warning("r2 score : 0.92")

        with tab2:
            # res = prediction(X,month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week,rf_model)
            res = predict(5, 6, 1515, 328, -8.0, 'AIRLINE_OO', 'ORIGIN_AIRPORT_PHX', 'DESTINATION_AIRPORT_ABQ',
                          'DAY_OF_WEEK_TUESDAY', dt_model)
            st.subheader("Decision Tree MODEL PREDICTION")
            if (res >= 0):
                text1 = "Flight is **not delayed**. It will depart for next flight at scheduled time."
                st.success(text1)
            elif (res >= -15):
                text2 = "Flight is only delayed by " + str(
                    abs(res)) + ". Delays upto 15 minutes are considered as not delay. **FLIGHT IS NOT DELAYED**"
                st.warning(text2)
            else:
                text3 = "**FLIGHT IS DELAYED** by " + str(
                    res) + ". Delays by more than 15 minutes are considered to be actual delays. FLIGHT IS DELAY"
                st.error(text3)

            st.subheader('Decision Tree MODEL PARAMETERS')
            st.info("MAE : 8.14")
            st.error("MSE : 183.81")
            st.success("RMSE : 13.55")
            st.warning("r2 score : 0.867")

        with tab3:
            # res = prediction(X,month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week,rf_model)
            res = predict(5, 6, 1515, 328, -8.0, 'AIRLINE_OO', 'ORIGIN_AIRPORT_PHX', 'DESTINATION_AIRPORT_ABQ',
                          'DAY_OF_WEEK_TUESDAY', xgb_model)
            st.subheader("XGBoost MODEL PREDICTION")
            if (res >= 0):
                text1 = "Flight is **not delayed**. It will depart for next flight at scheduled time."
                st.success(text1)
            elif (res >= -15):
                text2 = "Flight is only delayed by " + str(
                    abs(res)) + ". Delays upto 15 minutes are considered as not delay. **FLIGHT IS NOT DELAYED**"
                st.warning(text2)
            else:
                text3 = "**FLIGHT IS DELAYED** by " + str(
                    res) + ". Delays by more than 15 minutes are considered to be actual delays. FLIGHT IS DELAY"
                st.error(text3)

            st.subheader('Decision Tree MODEL PARAMETERS')
            st.info("MAE : 6.41")
            st.error("MSE : 211.92")
            st.success("RMSE : 14.55")
            st.warning("r2 score : 0.84")

        with tab4:
            # res = prediction(X,month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week,rf_model)
            res = predict(5, 6, 1515, 328, -8.0, 'AIRLINE_OO', 'ORIGIN_AIRPORT_PHX', 'DESTINATION_AIRPORT_ABQ',
                          'DAY_OF_WEEK_TUESDAY', lr_model)
            st.subheader("XGBoost MODEL PREDICTION")
            if (res >= 0):
                text1 = "Flight is **not delayed**. It will depart for next flight at scheduled time."
                st.success(text1)
            elif (res >= -15):
                text2 = "Flight is only delayed by " + str(
                    abs(res)) + ". Delays upto 15 minutes are considered as not delay. **FLIGHT IS NOT DELAYED**"
                st.warning(text2)
            else:
                text3 = "**FLIGHT IS DELAYED** by " + str(
                    res) + ". Delays by more than 15 minutes are considered to be actual delays. FLIGHT IS DELAY"
                st.error(text3)

            st.subheader('Decision Tree MODEL PARAMETERS')
            st.info("MAE : 8.16")
            st.error("MSE : 140.12")
            st.success("RMSE : 11.83")
            st.warning("r2 score : 0.899")

        import matplotlib.pyplot as plt

        models = ['Random Forest', 'Decision Tree', 'XGBoost', 'Linear Regression']
        r2_scores = [0.92, 0.867, 0.84, 0.899]

        fig, ax = plt.subplots()
        ax.bar(models, r2_scores, color=['blue', 'green', 'red', 'orange'])
        ax.set_xlabel('Models')
        ax.set_ylabel('R^2 Scores')
        ax.set_title('R^2 Scores of Different Models')

        st.subheader("MODEL PERFORMANCE COMPARISION")
        st.pyplot(fig)# Display plot on Streamlit page

        st.info('By comparison, it is evident that **Random Forest** regressor is the best performing model'
                ' with highest r2 score value.')



if __name__=='__main__':
    main()