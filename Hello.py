import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

def main():
    
    st.markdown("""
    <style>
        body {text-align: center;}
    </style>
    """, unsafe_allow_html=True)

    st.header(":orange[DIABETES DETECTION]", divider='orange')
 
    # Widget to select XLSX file
    uploaded_file = st.file_uploader("FILE XLSX:", type="xlsx")

    # Handling uploaded file
    if uploaded_file is not None:
        # PROCESS CSV
        try:
            # Read the uploaded Excel file into a DataFrame
            df = pd.read_excel(uploaded_file)

            # Normalize 'Frequency' column
            df['Frequency'] = df['Frequency'] / 1000000000

            # Save modified DataFrame to a temporary file
            file_path = "temp_modified_data.xlsx"  # Adjust path if needed
            df.to_excel(file_path, index=False)

            # Read back specific columns
            kolom = ['Frequency', 's11-magnitude (db)']
            df = pd.read_excel(file_path, usecols=kolom)

            # Calculate return loss and related values
            nilaiterkecil = df[df['s11-magnitude (db)'] < 0]['s11-magnitude (db)']
            returnloss = nilaiterkecil.min()
            baris_rl = df[df['s11-magnitude (db)'] == returnloss].index[0] + 1
            frekuensi = df.at[baris_rl, 'Frequency']
            list_rl_atas = []  
            list_rl_bawah = []

             # Deret Nilai Batas Atas
            NA = baris_rl
            while NA < len(df):
                nilai = df.iloc[NA]['s11-magnitude (db)']
                if nilai < -10:
                    list_rl_atas.append(nilai)
                else:
                    break
                NA += 1

            # Deret Nilai Batas Bawah
            NB = baris_rl
            while NB >= 0:
                nilai = df.iloc[NB]['s11-magnitude (db)']
                if nilai < -10:
                    list_rl_bawah.append(nilai)
                else:
                    break
                NB -= 1

            #Nilai Return Loss Batas Bawah dan Atas
            nilai_rl_atas = list_rl_atas[-1]
            baris_rl_atas = df[df['s11-magnitude (db)'] == nilai_rl_atas].index[0] + 1
            nilai_rl_bawah = list_rl_bawah[-1]
            baris_rl_bawah = df[df['s11-magnitude (db)'] == nilai_rl_bawah].index[0] + 1
            rentan = nilai_rl_atas - nilai_rl_bawah
            bandwidth = abs(rentan*100).round(5)
            
            # PROSES PREDIKSI
            interpreter = tf.lite.Interpreter(model_path="data.tflite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_data = np.array([[frekuensi, returnloss, bandwidth]], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            selected_output = np.argmax(output_data)

            
            st.header(":orange[HASIL PERHITUNGAN]", divider=('orange'))
            

            col1, col2 = st.columns(2)

            with col1:
                st.latex(r"\text{KADAR GULA DARAH}") 
                st.subheader(f"`{selected_output}` ` MG/DL`")


           
            with col2:
                st.latex(r"\text{GOLONGAN}") 
                if selected_output < 100 :
                    st.subheader("`PRA DIABETES`")
                elif selected_output > 199 :
                    st.subheader("`DIABETES`")
                else :
                    st.subheader("`NORMAL`")

        except Exception as e:
            st.error("Terjadi kesalahan saat pemrosesan CSV:")
            st.error(e)

if __name__ == "__main__":
    main()
