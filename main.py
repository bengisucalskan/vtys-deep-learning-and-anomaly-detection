import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from st_pages import Page, show_pages,add_page_title
from streamlit_option_menu import option_menu
import time
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from io import StringIO
import pandas as pd
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report


col1, col2, col3 = st.columns([1,2,1])
with col2:
    logo_path = 'C://Users//90544//PycharmProjects//pythonProject11//marmara.png'
    st.image(logo_path, width=300)


st.sidebar.header("VERİ ANALİZ PORTALI")

with st.sidebar:
    selected=option_menu(
        menu_title = "Anasayfa",
        options = ["Veri Analizi","ML algorithm"],
        icons = ["house-heart-fill"],
        default_index = 0,
    )

if selected == "ML algorithm":
    upload_file = st.file_uploader("Upload your CSV File", type=['csv'])
    if upload_file is not None:
        with st.spinner('Veriler yükleniyor...'):
            # Uzun süren işlem simülasyonu
            time.sleep(5)
        st.success('Yükleme tamamlandı!')

        # Veri setini yükle

        st.write("""# KNN algorithm""")
        df = pd.read_csv(upload_file)
        veri = df.copy()
        df1 = df.copy()
        df2 = df.copy()



        # İlk iki sütunu atma

        def categorize_page_life_expectancy(value):
            if value < 301:
                return 'Dusuk'
            elif 301 <= value < 5000:
                return 'Orta'
            else:
                return 'Yuksek'


        # Yeni bir sütun oluşturarak her bir örneği bu kategorilere göre etiketleme
        df['page_life_category'] = df['page life expectancy'].apply(categorize_page_life_expectancy)

        # Her bir satır için 'alışveriş sepeti' oluşturma
        transactions = df.groupby(['InstanceID'])['page_life_category'].apply(list).tolist()
        df.drop(columns=['InstanceID', 'CreateDate'], inplace=True)
        # Kategorik değişkenleri sayısallaştırma
        label_encoder = LabelEncoder()
        df['page_life_category'] = label_encoder.fit_transform(df['page_life_category'])
        # Kategorik verileri sayısallaştırma
        label_encoder = LabelEncoder()
        df['page_life_category'] = label_encoder.fit_transform(df['page_life_category'])

        # Tüm verileri ölçeklendirme
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)

        # Elbow metodu ile optimal küme sayısını belirleme
        inertia = []
        for i in range(1, 11):  # 1'den 10'a kadar küme sayısı deniyoruz
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data_scaled)
            inertia.append(kmeans.inertia_)

        # Elbow grafiğini çizdirme
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 11), inertia, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        st.pyplot()

        # Optimal küme sayısıyla modeli eğitme


        # Veri setini yükleme





        veri['etiket'] = veri['page life expectancy'].apply(lambda x: 1 if x <= 300 else 0)

        label_counts = veri['etiket'].value_counts()
        df_majority = veri[veri['etiket'] == label_counts.idxmax()]
        df_minority = veri[veri['etiket'] == label_counts.idxmin()]

        # Resample the dataset
        df_balanced = resample(df_majority,
                               replace=False,
                               n_samples=label_counts.min(),
                               random_state=42)
        df_balanced = pd.concat([df_balanced, df_minority])

        X = df_balanced.iloc[:, 2:-1]  # Öznitelikler
        y = df_balanced['etiket']  # Hedef

        # Sonuçları saklamak için bir DataFrame oluşturun
        results = pd.DataFrame(columns=['K', 'Random State', 'Accuracy', 'F1 Score', 'Recall', 'Confusion Matrix'])

        # Farklı k değerleri ve random_state değerleri için döngü
        for k in [3, 4, 5, 6]:
            for random_state in [42, 84]:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                # Metrikleri hesapla
                conf_matrix = confusion_matrix(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Sonuçları DataFrame'e ekle
                new_row = pd.DataFrame({'K': [k], 'Random State': [random_state], 'Accuracy': [accuracy],
                                        'F1 Score': [f1], 'Recall': [recall], 'Confusion Matrix': [conf_matrix]})
                results = pd.concat([results, new_row], ignore_index=True)

        # Sonuçları kaydet
        results.to_csv('model_results.csv', index=False)
        results = pd.read_csv('model_results.csv')
        st.write("""  # KNN Model Performans Karşılaştırması""")
        # DataFrame'i göster
        st.dataframe(results)
        ###################################################################

        # İlk iki sütunu atma ve kategorilere ayırma
        df1['page_life_category'] = df1['page life expectancy'].apply(categorize_page_life_expectancy)
        df1.drop(columns=['InstanceID', 'CreateDate'], inplace=True)

        # Label encoding
        label_encoder = LabelEncoder()
        df1['page_life_category'] = label_encoder.fit_transform(df1['page_life_category'])

        # Veriyi ayrıştırma
        X = df1.drop('page_life_category', axis=1)
        y = df1['page_life_category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sonuçları saklamak için bir DataFrame
        results1 = pd.DataFrame(columns=['n_estimators', 'max_depth', 'Accuracy', 'F1 Score', 'Recall'])

        # Farklı parametre değerleri için modeli eğit
        for n in [1, 10, 25]:
            for depth in [1, 3, 5]:
                rf_model = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)

                # Performans metriklerini hesapla
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')

                # Sonuçları DataFrame'e ekle
                new_row1 = pd.DataFrame({'n_estimators': [n], 'max_depth': [depth], 'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall]})
                results1 = pd.concat([results1, new_row1], ignore_index=True)

        # Sonuçları kaydet
        results1.to_csv('random_forest_results1.csv', index=False)
        st.title('Random Forest Model Performans Karşılaştırması')

        # Sonuçları yükle
        results1 = pd.read_csv('random_forest_results1.csv')

        # DataFrame'i göster
        st.dataframe(results1)


        ############################################

        X = df2.iloc[:, 2:7]  # 2 ila 6 arasındaki sütunlar
        y = df2['page life expectancy'] > 300  # Hedef etiket

        # Veriyi eğitim ve test kümelerine bölelim
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sonuçları saklamak için bir DataFrame
        results2 = pd.DataFrame(columns=['Kernel', 'Accuracy', 'F1 Score', 'Recall', 'Confusion Matrix'])

        # Farklı kernel türleri için modeli eğit
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in kernels:
            svm_model = SVC(kernel=kernel)
            svm_model.fit(X_train, y_train)
            y_pred = svm_model.predict(X_test)

            # Performans metriklerini hesapla
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Sonuçları DataFrame'e ekle
            new_row2 = pd.DataFrame({'Kernel': [kernel], 'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall],
                                    'Confusion Matrix': [conf_matrix]})
            results2 = pd.concat([results2, new_row2], ignore_index=True)

        # Sonuçları yazdır
        print(results2)
        results2.to_csv('svm-algo.csv', index=False)


        # Sonuçları yükle


        # Her bir kernel için confusion matrix görselleştirme
        for index, row in results2.iterrows():
            plt.figure(figsize=(3, 3))
            sns.heatmap(row['Confusion Matrix'], annot=True, fmt="d", cbar=False)
            plt.title(f"Kernel: {row['Kernel']}")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot()

        st.title('Support Vector Machine (SVM) Model Performans Karşılaştırması')
        results2 = pd.read_csv('svm-algo.csv')

        # DataFrame'i göster
        st.dataframe(results2)
elif selected =="Veri Analizi":
    add_page_title()

    show_pages(
        [
            Page("main.py", "Melih BALTA - Bengisu ÇALIŞKAN"),
        ]
    )
    st.header("Single File Upload")
    upload_file = st.file_uploader("Upload your CSV File",type=['csv'])

    # Uzun süren işlem simülasyonu


    if upload_file is not None:
        with st.spinner('Veriler yükleniyor...'):
            # Uzun süren işlem simülasyonu
            time.sleep(5)
        st.success('Yükleme tamamlandı!')

        # Veri setini yükle
        df = pd.read_csv(upload_file)
        veriler = df.copy()
        veriler_1 = df.copy()
        fpgrowth_veriler = df.copy()
        # 'page life expectancy' kolonunu etiketleme

        st.write( """# df.head()""")
        st.write(df.head())
        st.write("""# df.info()""" )
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.write(""" # df.describe().T""")
        st.write(df.describe().T)
        st.write( """# df.describe(include=object).T""")
        st.write(df.describe(include=object).T)


        def categorize_page_life_expectancy(value):
            if value < 301:
                return 'Dusuk'
            elif 301 <= value < 5000:
                return 'Orta'
            else:
                return 'Yuksek'


        # Yeni bir sütun oluşturarak her bir örneği bu kategorilere göre etiketleme
        df['page_life_category'] = df['page life expectancy'].apply(categorize_page_life_expectancy)

        # Her bir satır için 'alışveriş sepeti' oluşturma
        transactions = df.groupby(['InstanceID'])['page_life_category'].apply(list).tolist()
        df.drop(columns=['InstanceID', 'CreateDate'], inplace=True)
        # Kategorik değişkenleri sayısallaştırma
        label_encoder = LabelEncoder()
        df['page_life_category'] = label_encoder.fit_transform(df['page_life_category'])

        # Kategori frekanslarını hesaplama
        labels = df['page_life_category'].value_counts()
        w = (list(labels.index), list(labels.values))

        # Pasta grafiği için explode ayarları
        explode = [0, 0, 0.1]  # Yüksek kategorisini biraz öne çıkar

        # Pasta grafiği çizme
        fig = plt.figure(figsize=(7, 5))
        plt.pie(w[1], explode=explode, labels=w[0], shadow=True, startangle=90,
                colors=['red', 'green', 'blue'], autopct='%1.1f%%', textprops={'fontsize': 15})
        plt.axis('equal')  # Bu sayede pasta grafiği yuvarlak görünür
        plt.legend(title='[page life expectancy] değerleri', loc='upper left')
        st.pyplot()

        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.utils import resample

        # Veri setini yükle


        # 'page life expectancy' kolonunu etiketleme
        df['etiket'] = df['page life expectancy'].apply(lambda x: 1 if x <= 300 else 0)

        # Orijinal sınıf dağılımını görselleştirme
        plt.figure(figsize=(8, 4))
        df['etiket'].value_counts().plot(kind='bar', color=['blue', 'red'])
        plt.title('Orijinal Sınıf Dağılımı')
        plt.xlabel('Etiket')
        plt.ylabel('Frekans')
        plt.xticks([0, 1], ['page life expectancy (>300)', 'page life expectancy (0-300)'], rotation=0)
        st.pyplot()
        print("Orijinal Veri Seti Bilgileri:")
        print("Toplam veri sayısı:", len(df))
        print("Sınıf dağılımı:\n", df['etiket'].value_counts())
        # Azınlık ve çoğunluk sınıflarını ayır
        df_majority = df[df['etiket'] == 0]
        df_minority = df[df['etiket'] == 1]

        # Azınlık sınıfını çoğaltma (Oversampling)
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])

        # Çoğunluk sınıfını azaltma (Undersampling)
        df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=123)
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])

        # Güncellenmiş sınıf dağılımlarını yazdır ve görselleştir
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        df_upsampled['etiket'].value_counts().plot(kind='bar', ax=ax[0], color=['blue', 'red'])
        ax[0].set_title('Oversampling Sonrası Sınıf Dağılımı')
        ax[0].set_xlabel('Etiket')
        ax[0].set_ylabel('Frekans')
        ax[0].set_xticklabels(['page life expectancy (>300)', 'page life expectancy (0-300)'], rotation=0)
        print("\nOversampling Yapıldıktan Sonra:")
        print("Toplam veri sayısı:", len(df_upsampled))
        print("Sınıf dağılımı:\n", df_upsampled['etiket'].value_counts())

        df_downsampled['etiket'].value_counts().plot(kind='bar', ax=ax[1], color=['blue', 'red'])
        ax[1].set_title('Undersampling Sonrası Sınıf Dağılımı')
        ax[1].set_xticklabels(['page life expectancy (>300)', 'page life expectancy (0-300)'], rotation=0)
        ax[1].set_ylabel('Frekans')
        ax[1].set_xlabel('Etiket')
        st.pyplot()
        print("\nUndersampling Yapıldıktan Sonra:")
        print("Toplam veri sayısı:", len(df_downsampled))
        print("Sınıf dağılımı:\n", df_downsampled['etiket'].value_counts())









        st.dataframe(veriler,width=900,height=600)


        # Verileri yükleme


        # Korelasyon matrisini hesaplama
        Y = veriler.iloc[:,2:7]
        korelasyon_matrisi = Y.corr()

        # Excel dosyasını pandas DataFrame olarak yükle

        veriler = veriler.iloc[:, 2:7]

        # İstatistiksel özetleri saklamak için boş bir DataFrame oluştur
        istatistikler = pd.DataFrame(columns=['Özellik', 'Mod', 'Medyan', 'Standart Sapma', 'Ortalama', 'Minimum', 'Maksimum'])

        for kolon in veriler.columns:
            # Mod hesaplama (birden fazlamod olabilir)
            mod = veriler[kolon].mode().values
            mod_str = ', '.join(map(str, mod))

            # Diğer istatistikleri hesapla
            medyan = veriler[kolon].median()
            standart_sapma = veriler[kolon].std()
            ortalama = veriler[kolon].mean()
            minimum = veriler[kolon].min()
            maksimum = veriler[kolon].max()

            # Hesaplanan istatistikleri DataFrame'e ekle
            istatistikler = pd.concat([istatistikler, pd.DataFrame({'Özellik': [kolon],
                                                  'Mod': [mod_str],
                                                  'Medyan': [medyan],
                                                  'Standart Sapma': [standart_sapma],
                                                  'Ortalama': [ortalama],
                                                  'Minimum': [minimum],
                                                  'Maksimum': [maksimum]})], ignore_index=True)

        # Streamlit uygulamasını oluşturma
        st.write("""
        # İstatistik Tablosu
        """)
        st.write(istatistikler)




        # Streamlit uygulamasını oluşturma
        st.write("""
        # Korelasyon Matrisi
        """)

        st.write(korelasyon_matrisi)

        # Isı haritasını oluşturma
        plt.figure(figsize=(10,8))
        sns.heatmap(korelasyon_matrisi,annot=True,fmt=".2f",cmap='coolwarm')
        st.pyplot()
        st.write("""
                # Page Life Expectancy Değerine Göre Isı Haritası""")


        def categorize_page_life_expectancy(value):
            if value < 301:
                return 'Dusuk'
            elif 301 <= value < 5000:
                return 'Orta'
            else:
                return 'Yuksek'


        # Yeni bir sütun oluşturarak her bir örneği bu kategorilere göre etiketleme
        fpgrowth_veriler['page_life_category'] = fpgrowth_veriler['page life expectancy'].apply(categorize_page_life_expectancy)

        # Her bir satır için 'alışveriş sepeti' oluşturma
        transactions = fpgrowth_veriler.groupby(['InstanceID'])['page_life_category'].apply(list).tolist()

        # Veri setini FP-Growth için uygun formata dönüştürme
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_ready_for_fp = pd.DataFrame(te_ary, columns=te.columns_)

        # FP-Growth algoritmasını kullanarak sık kullanılan item setlerini bulma
        frequent_itemsets = fpgrowth(df_ready_for_fp, min_support=0.1, use_colnames=True)

        # Kuralları çıkarma
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        grouped = fpgrowth_veriler.groupby(['InstanceID', 'page_life_category']).size().unstack(fill_value=0)


        plt.figure(figsize=(10, 8))
        sns.heatmap(grouped, annot=True, fmt="d", cmap="YlGnBu")
        plt.title('Page Life Expectancy Kategorilerine Göre Isı Haritası')
        plt.xlabel('Page Life Kategorisi')
        plt.ylabel('InstanceID')
        st.pyplot(plt)

        df['page_life_expectancy_etiket'] = df['page life expectancy'].apply(lambda x: 1 if x <= 300 else 0)

        # Altıncı indeksten son indekse kadar olan sütunlar
        altinci_sutundan_sonra = df.columns[6:]  # Altıncı indeksten itibaren sütun isimlerini al

        # İlişkileri saklamak için boş bir DataFrame oluştur
        iliskiler_df = pd.DataFrame(columns=['Sütun', '1 için Ortalama', '0 için Ortalama'])

        for sutun in altinci_sutundan_sonra:
            # 1 ve 0 değerlerinin ortalamalarını hesapla
            bir_ortalama = df[df[sutun] == 1]['page_life_expectancy_etiket'].mean()
            sifir_ortalama = df[df[sutun] == 0]['page_life_expectancy_etiket'].mean()

            # Yeni verileri DataFrame'e ekle
            if not pd.isna(bir_ortalama) and bir_ortalama != 0:
                yeni_satir = pd.DataFrame(
                    {'Sütun': [sutun], '1 için Ortalama': [bir_ortalama], '0 için Ortalama': [sifir_ortalama]})
                iliskiler_df = pd.concat([iliskiler_df, yeni_satir], ignore_index=True)

        # Sonuçları göster
        print(iliskiler_df)

        # Grafik boyutunu ayarla
        plt.figure(figsize=(12, 8))

        # Her bir sütun için, 1 ve 0 değerlerinin ortalamalarını bar grafik olarak çizdir
        bar_genisligi = 0.35
        index = range(len(iliskiler_df))

        plt.bar(index, iliskiler_df['1 için Ortalama'], bar_genisligi, label='1 için Ortalama', color='blue')
        plt.bar([p + bar_genisligi for p in index], iliskiler_df['0 için Ortalama'], bar_genisligi,
                label='0 için Ortalama', color='red')

        # Grafik başlığı ve eksen etiketlerini ekle
        plt.xlabel('Kolonlar')
        plt.ylabel('Ortalama Etiket Değeri')
        plt.title('Diğer Kolon Değerlerinin "Page Life Expectancy" Üzerindeki Etkisi')
        plt.xticks([p + bar_genisligi / 2 for p in index], iliskiler_df['Sütun'], rotation=90)

        # Efsaneyi ekle
        plt.legend()

        # Grafik gösterimi
        plt.tight_layout()
        st.pyplot()



        st.write("""
        # FP Growth Algorithm""")

        categories = {
            'page life expectancy': [('Yuksek', 0, 300), ('Orta', 300, 5000), ('Dusuk', 5000, 50000)],
            '% privileged time': [('Dusuk', 0, 15), ('Orta', 15, 25), ('Yuksek', 25, 50)],
            'transactions/sec': [('Dusuk', 0, 300), ('Orta', 300, 5000), ('Yuksek', 5000, 20000)],
            'write transactions/sec': [('Dusuk', 0, 500), ('Orta', 500, 1500), ('Yuksek', 1500, 5000)],
            'logical connections': [('Dusuk', 0, 500), ('Orta', 500, 750), ('Yuksek', 750, 2000)]
        }
        for column, bins in categories.items():
            bin_edges = [b[1] for b in bins]  # Artık ilk -float('inf') kullanmıyoruz
            # Sadece listeye başlangıç olarak 0 ekliyoruz
            bin_edges.append(float('inf'))  # Listenin sonuna pozitif sonsuzluk ekliyoruz
            labels = [b[0] for b in bins]
            veriler_1[column] = pd.cut(veriler_1[column], bins=bin_edges, labels=labels, right=False)
        transactions = veriler_1[categories.keys()].apply(lambda x: x.dropna().tolist(), axis=1).tolist()

        # FP-Growth için uygun formata dönüştürme
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_ready_for_fp = pd.DataFrame(te_ary, columns=te.columns_)

        # FP-Growth algoritmasını kullanarak sık kullanılan item setlerini bulma
        frequent_itemsets = fpgrowth(df_ready_for_fp, min_support=0.1, use_colnames=True)

        # Kuralları çıkarma
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        # Sık kullanılan öğe setlerinin destek değerlerini görselleştirme
        plt.figure(figsize=(12, 8))
        sns.barplot(x=frequent_itemsets['support'], y=frequent_itemsets['itemsets'].astype(str))
        plt.title('Support of Frequent Itemsets')
        plt.xlabel('Support')
        plt.ylabel('Itemsets')
        st.pyplot()

        # İlişkilendirme kurallarının güven değerlerini görselleştirme
        plt.figure(figsize=(12, 8))
        sns.barplot(x=rules['confidence'], y=rules['antecedents'].astype(str) + '=>' + rules['consequents'].astype(str))
        plt.title('Confidence of Association Rules')
        plt.xlabel('Confidence')

        plt.ylabel('Association Rules')
        st.pyplot(plt)
        # İlişkilendirme kurallarının kaldıraç değerlerini görselleştirme
        plt.figure(figsize=(12, 8))
        sns.barplot(x=rules['leverage'], y=rules['antecedents'].astype(str) + '=>' + rules['consequents'].astype(str))
        plt.title('Leverage of Association Rules')
        plt.xlabel('Leverage')
        plt.ylabel('Association Rules')
        st.pyplot()

