# Phishing Website Detection - Comprehensive Analysis and Machine Learning Project

## Project Overview
This project employs machine learning techniques to detect whether websites are phishing attacks or legitimate. The analysis is conducted on 10,000 website samples with 49 different features.

### Objectives:
- Analyze website characteristics and patterns
- Identify discriminating factors between phishing and legitimate sites
- Develop high-accuracy classification models
- Comprehensively evaluate model performance and reliability
- Provide actionable insights for cybersecurity applications



```python
# Essential Library Imports
"""
This section imports all necessary libraries for data analysis, visualization, 
and machine learning model development.
"""

# Core data manipulation and analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries for model development and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Visualization configuration for consistent and professional plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("All required libraries have been successfully imported.") 
```

    All required libraries have been successfully imported.


## Data Loading and Initial Exploration



```python
# Data Loading and Basic Information Extraction
"""
Load the phishing detection dataset and perform initial data exploration.
This includes examining data structure, dimensions, and basic statistics.
"""

# Load the dataset from CSV file
df = pd.read_csv('Phishing_Legitimate_full.csv')

print("DATASET GENERAL INFORMATION")
print("=" * 50)
print(f"Total number of samples: {len(df):,}")
print(f"Number of features: {df.shape[1]-1}")
print(f"Target variable: CLASS_LABEL (0: Legitimate, 1: Phishing)")
print(f"Dataset dimensions: {df.shape}")

# Display first few rows to understand data structure
print("\nFirst 5 rows of the dataset:")
display(df.head())

# Comprehensive data information
print("\nDETAILED DATASET INFORMATION")
print("=" * 50)
df.info()

# Statistical summary of all features
print("\nSTATISTICAL SUMMARY")
print("=" * 50)
display(df.describe().round(3))
```

    DATASET GENERAL INFORMATION
    ==================================================
    Total number of samples: 10,000
    Number of features: 49
    Target variable: CLASS_LABEL (0: Legitimate, 1: Phishing)
    Dataset dimensions: (10000, 50)
    
    First 5 rows of the dataset:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NumDots</th>
      <th>SubdomainLevel</th>
      <th>PathLevel</th>
      <th>UrlLength</th>
      <th>NumDash</th>
      <th>NumDashInHostname</th>
      <th>AtSymbol</th>
      <th>TildeSymbol</th>
      <th>NumUnderscore</th>
      <th>...</th>
      <th>IframeOrFrame</th>
      <th>MissingTitle</th>
      <th>ImagesOnlyInForm</th>
      <th>SubdomainLevelRT</th>
      <th>UrlLengthRT</th>
      <th>PctExtResourceUrlsRT</th>
      <th>AbnormalExtFormActionR</th>
      <th>ExtMetaScriptLinkRT</th>
      <th>PctExtNullSelfRedirectHyperlinksRT</th>
      <th>CLASS_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>72</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>144</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>58</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>79</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>


    
    DETAILED DATASET INFORMATION
    ==================================================
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 50 columns):
     #   Column                              Non-Null Count  Dtype  
    ---  ------                              --------------  -----  
     0   id                                  10000 non-null  int64  
     1   NumDots                             10000 non-null  int64  
     2   SubdomainLevel                      10000 non-null  int64  
     3   PathLevel                           10000 non-null  int64  
     4   UrlLength                           10000 non-null  int64  
     5   NumDash                             10000 non-null  int64  
     6   NumDashInHostname                   10000 non-null  int64  
     7   AtSymbol                            10000 non-null  int64  
     8   TildeSymbol                         10000 non-null  int64  
     9   NumUnderscore                       10000 non-null  int64  
     10  NumPercent                          10000 non-null  int64  
     11  NumQueryComponents                  10000 non-null  int64  
     12  NumAmpersand                        10000 non-null  int64  
     13  NumHash                             10000 non-null  int64  
     14  NumNumericChars                     10000 non-null  int64  
     15  NoHttps                             10000 non-null  int64  
     16  RandomString                        10000 non-null  int64  
     17  IpAddress                           10000 non-null  int64  
     18  DomainInSubdomains                  10000 non-null  int64  
     19  DomainInPaths                       10000 non-null  int64  
     20  HttpsInHostname                     10000 non-null  int64  
     21  HostnameLength                      10000 non-null  int64  
     22  PathLength                          10000 non-null  int64  
     23  QueryLength                         10000 non-null  int64  
     24  DoubleSlashInPath                   10000 non-null  int64  
     25  NumSensitiveWords                   10000 non-null  int64  
     26  EmbeddedBrandName                   10000 non-null  int64  
     27  PctExtHyperlinks                    10000 non-null  float64
     28  PctExtResourceUrls                  10000 non-null  float64
     29  ExtFavicon                          10000 non-null  int64  
     30  InsecureForms                       10000 non-null  int64  
     31  RelativeFormAction                  10000 non-null  int64  
     32  ExtFormAction                       10000 non-null  int64  
     33  AbnormalFormAction                  10000 non-null  int64  
     34  PctNullSelfRedirectHyperlinks       10000 non-null  float64
     35  FrequentDomainNameMismatch          10000 non-null  int64  
     36  FakeLinkInStatusBar                 10000 non-null  int64  
     37  RightClickDisabled                  10000 non-null  int64  
     38  PopUpWindow                         10000 non-null  int64  
     39  SubmitInfoToEmail                   10000 non-null  int64  
     40  IframeOrFrame                       10000 non-null  int64  
     41  MissingTitle                        10000 non-null  int64  
     42  ImagesOnlyInForm                    10000 non-null  int64  
     43  SubdomainLevelRT                    10000 non-null  int64  
     44  UrlLengthRT                         10000 non-null  int64  
     45  PctExtResourceUrlsRT                10000 non-null  int64  
     46  AbnormalExtFormActionR              10000 non-null  int64  
     47  ExtMetaScriptLinkRT                 10000 non-null  int64  
     48  PctExtNullSelfRedirectHyperlinksRT  10000 non-null  int64  
     49  CLASS_LABEL                         10000 non-null  int64  
    dtypes: float64(3), int64(47)
    memory usage: 3.8 MB
    
    STATISTICAL SUMMARY
    ==================================================



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>NumDots</th>
      <th>SubdomainLevel</th>
      <th>PathLevel</th>
      <th>UrlLength</th>
      <th>NumDash</th>
      <th>NumDashInHostname</th>
      <th>AtSymbol</th>
      <th>TildeSymbol</th>
      <th>NumUnderscore</th>
      <th>...</th>
      <th>IframeOrFrame</th>
      <th>MissingTitle</th>
      <th>ImagesOnlyInForm</th>
      <th>SubdomainLevelRT</th>
      <th>UrlLengthRT</th>
      <th>PctExtResourceUrlsRT</th>
      <th>AbnormalExtFormActionR</th>
      <th>ExtMetaScriptLinkRT</th>
      <th>PctExtNullSelfRedirectHyperlinksRT</th>
      <th>CLASS_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>...</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.00</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.000</td>
      <td>10000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5000.500</td>
      <td>2.445</td>
      <td>0.587</td>
      <td>3.300</td>
      <td>70.264</td>
      <td>1.818</td>
      <td>0.139</td>
      <td>0.000</td>
      <td>0.013</td>
      <td>0.323</td>
      <td>...</td>
      <td>0.340</td>
      <td>0.032</td>
      <td>0.030</td>
      <td>0.957</td>
      <td>0.02</td>
      <td>0.353</td>
      <td>0.793</td>
      <td>0.173</td>
      <td>0.314</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2886.896</td>
      <td>1.347</td>
      <td>0.751</td>
      <td>1.863</td>
      <td>33.370</td>
      <td>3.106</td>
      <td>0.546</td>
      <td>0.017</td>
      <td>0.114</td>
      <td>1.115</td>
      <td>...</td>
      <td>0.474</td>
      <td>0.177</td>
      <td>0.172</td>
      <td>0.248</td>
      <td>0.82</td>
      <td>0.889</td>
      <td>0.521</td>
      <td>0.756</td>
      <td>0.898</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>12.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>-1.000</td>
      <td>-1.00</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>-1.000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2500.750</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>48.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>-1.00</td>
      <td>-1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>-1.000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5000.500</td>
      <td>2.000</td>
      <td>1.000</td>
      <td>3.000</td>
      <td>62.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7500.250</td>
      <td>3.000</td>
      <td>1.000</td>
      <td>4.000</td>
      <td>84.000</td>
      <td>2.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10000.000</td>
      <td>21.000</td>
      <td>14.000</td>
      <td>18.000</td>
      <td>253.000</td>
      <td>55.000</td>
      <td>9.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>18.000</td>
      <td>...</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 50 columns</p>
</div>



```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(34, 34))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](analysis_summary_files/analysis_summary_4_0.png)
    


##  Hedef Değişken Analizi



```python
# Hedef değişkenin dağılımı
target_counts = df['CLASS_LABEL'].value_counts()
target_percentages = df['CLASS_LABEL'].value_counts(normalize=True) * 100

print("🎯 HEDEF DEĞİŞKEN DAĞILIMI")
print("=" * 50)
print(f"Meşru Siteler (0): {target_counts[0]:,} ({target_percentages[0]:.1f}%)")
print(f"Phishing Siteler (1): {target_counts[1]:,} ({target_percentages[1]:.1f}%)")

# Görselleştirme
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Çubuk grafik
target_counts.plot(kind='bar', ax=ax1, color=['#2E8B57', '#DC143C'])
ax1.set_title('Sınıf Dağılımı - Sayı', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sınıf (0: Meşru, 1: Phishing)')
ax1.set_ylabel('Örnek Sayısı')
ax1.tick_params(axis='x', rotation=0)

# Pasta grafik
ax2.pie(target_counts.values, labels=['Meşru Siteler', 'Phishing Siteler'], 
        autopct='%1.1f%%', colors=['#2E8B57', '#DC143C'], startangle=90)
ax2.set_title('Sınıf Dağılımı - Yüzde', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f" Veri seti dengeli: Her iki sınıfta da eşit sayıda örnek bulunuyor.")

```

    🎯 HEDEF DEĞİŞKEN DAĞILIMI
    ==================================================
    Meşru Siteler (0): 5,000 (50.0%)
    Phishing Siteler (1): 5,000 (50.0%)



    
![png](analysis_summary_files/analysis_summary_6_1.png)
    


     Veri seti dengeli: Her iki sınıfta da eşit sayıda örnek bulunuyor.


## 🧹 Veri Temizleme ve Ön İşleme



```python
# Eksik değerlerin kontrolü
print("🔍 EKSİK DEĞER ANALİZİ")
print("=" * 50)
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_info = pd.DataFrame({
    'Eksik Değer Sayısı': missing_values,
    'Eksik Değer Yüzdesi': missing_percentage
})

missing_info = missing_info[missing_info['Eksik Değer Sayısı'] > 0]

if len(missing_info) > 0:
    print("Eksik değerler tespit edildi:")
    display(missing_info)
else:
    print("Hiç eksik değer bulunmadı!")

# Duplicate kontrol
print(f" DUPLICATE KAYIT KONTROLÜ")
print("=" * 50)
duplicate_count = df.duplicated().sum()
print(f"Duplicate kayıt sayısı: {duplicate_count}")

if duplicate_count > 0:
    print("Duplicate kayıtlar kaldırılıyor...")
    df = df.drop_duplicates()
    print(f" {duplicate_count} duplicate kayıt kaldırıldı.")
    print(f"Yeni veri boyutu: {df.shape}")
else:
    print(" Duplicate kayıt bulunmadı!")

# Veri tiplerinin kontrolü
print(f" VERİ TİPLERİ")
print("=" * 50)
print(df.dtypes.value_counts())

```

    🔍 EKSİK DEĞER ANALİZİ
    ==================================================
    Hiç eksik değer bulunmadı!
     DUPLICATE KAYIT KONTROLÜ
    ==================================================
    Duplicate kayıt sayısı: 0
     Duplicate kayıt bulunmadı!
     VERİ TİPLERİ
    ==================================================
    int64      47
    float64     3
    Name: count, dtype: int64


## Keşifsel Veri Analizi (EDA)



```python
# En önemli özelliklerin analizi
important_features = ['NumDots', 'UrlLength', 'NumDash', 'HostnameLength', 
                     'PathLength', 'NoHttps', 'RandomString', 'IpAddress']

# Özellik dağılımlarının sınıflara göre karşılaştırılması
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, feature in enumerate(important_features):
    phishing_data = df[df['CLASS_LABEL'] == 1][feature]
    legitimate_data = df[df['CLASS_LABEL'] == 0][feature]
    
    axes[i].hist(legitimate_data, alpha=0.7, label='Meşru', bins=30, color='#2E8B57')
    axes[i].hist(phishing_data, alpha=0.7, label='Phishing', bins=30, color='#DC143C')
    axes[i].set_title(f'{feature} Dağılımı', fontweight='bold')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frekans')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('🔍 Önemli Özelliklerin Sınıflara Göre Dağılımı', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

```


    
![png](analysis_summary_files/analysis_summary_10_0.png)
    



```python
# 🔗 Korelasyon Analizi - Özellikler arası ilişkilerin incelenmesi
"""
Bu bölümde özelliklerin birbiriyle olan korelasyonlarını analiz ederek:
1. Hangi özelliklerin birbirine benzer bilgiler içerdiğini
2. Hedef değişkenle en yüksek korelasyona sahip özellikleri
3. Multicollinearity problemini tespit ederiz
"""

# 📊 Korelasyon matrisinin hesaplanması
correlation_matrix = df.corr()

# 🎯 Hedef değişkenle en yüksek korelasyona sahip özelliklerin belirlenmesi
target_correlations = correlation_matrix['CLASS_LABEL'].abs().sort_values(ascending=False)
top_features = target_correlations.head(15)  # İlk 15 özellik

print("🎯 HEDEF DEĞİŞKENLE EN YÜKSEK KORELASYONLU ÖZELLİKLER")
print("=" * 60)
for feature, corr in top_features.items():
    if feature != 'CLASS_LABEL':
        print(f"📌 {feature:<30}: {corr:.4f}")

# 🌡️ Korelasyon haritasının görselleştirilmesi (top 20 özellik)
plt.figure(figsize=(16, 14))
top_20_features = target_correlations.head(21).index  # CLASS_LABEL dahil 21
correlation_subset = correlation_matrix.loc[top_20_features, top_20_features]

# 🎨 Heatmap oluşturma
sns.heatmap(correlation_subset, 
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'label': 'Korelasyon Katsayısı'})

plt.title('🔍 En Önemli Özelliklerin Korelasyon Haritası', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print(f"\n💡 Analiz: En yüksek korelasyonlu özellikler model eğitiminde priorite alacak")

```

    🎯 HEDEF DEĞİŞKENLE EN YÜKSEK KORELASYONLU ÖZELLİKLER
    ============================================================
    📌 id                            : 0.8660
    📌 PctExtNullSelfRedirectHyperlinksRT: 0.5405
    📌 FrequentDomainNameMismatch    : 0.4640
    📌 NumDash                       : 0.3722
    📌 SubmitInfoToEmail             : 0.3576
    📌 PctNullSelfRedirectHyperlinks : 0.3428
    📌 InsecureForms                 : 0.3164
    📌 NumDots                       : 0.2941
    📌 PctExtHyperlinks              : 0.2597
    📌 NumSensitiveWords             : 0.2552
    📌 IframeOrFrame                 : 0.2352
    📌 PathLevel                     : 0.2295
    📌 AbnormalExtFormActionR        : 0.1858
    📌 UrlLengthRT                   : 0.1695



    
![png](analysis_summary_files/analysis_summary_11_1.png)
    


    
    💡 Analiz: En yüksek korelasyonlu özellikler model eğitiminde priorite alacak


## 🔬 Özellik Mühendisliği ve Seçimi



```python
# 🔬 Özellik Mühendisliği ve Seçimi 
"""
🎯 Bu bölümde amacımız:
1. 📊 En önemli özellikleri belirlemek
2. 🧬 Yeni özellikler oluşturmak  
3. ⚡ Model performansını artıracak özellikleri seçmek
4. 🗑️ Gereksiz özellikleri elemek
"""

# Özellik önemlilik skorlarının hesaplanması (Random Forest ile)
print("🌳 RANDOM FOREST İLE ÖZELLİK ÖNEMLİLİK ANALİZİ")
print("=" * 60)

#  Verinin hazırlanması
X = df.drop(['CLASS_LABEL', 'id'], axis=1)  # 🗑️ ID sütununu da çıkarıyoruz
y = df['CLASS_LABEL']

#  Random Forest modeli ile özellik önemliliği
rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_importance.fit(X, y)

#  Özellik önemlilik skorları
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_importance.feature_importances_
}).sort_values('Importance', ascending=False)

#  En önemli 15 özellik
top_15_features = feature_importance.head(15)

print("🏆 EN ÖNEMLİ 15 ÖZELLİK:")
print("-" * 40)
for idx, row in top_15_features.iterrows():
    print(f"🔹 {row['Feature']:<25}: {row['Importance']:.4f}")

# Görselleştirme
plt.figure(figsize=(14, 8))
top_20 = feature_importance.head(20)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_20)))

bars = plt.barh(range(len(top_20)), top_20['Importance'], color=colors)
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('📊 Özellik Önemlilik Skoru')
plt.title('🏆 En Önemli 20 Özellik - Random Forest Analizi', 
          fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()

#  Değerleri çubukların üzerine yazma
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\n💡 Analiz Sonucu: İlk 15 özellik toplam varyansın {top_15_features['Importance'].sum():.2%}'ini açıklıyor")

```

    🌳 RANDOM FOREST İLE ÖZELLİK ÖNEMLİLİK ANALİZİ
    ============================================================
    🏆 EN ÖNEMLİ 15 ÖZELLİK:
    ----------------------------------------
    🔹 PctExtHyperlinks         : 0.1931
    🔹 PctExtNullSelfRedirectHyperlinksRT: 0.1906
    🔹 PctExtResourceUrls       : 0.0777
    🔹 FrequentDomainNameMismatch: 0.0663
    🔹 PctNullSelfRedirectHyperlinks: 0.0613
    🔹 NumDash                  : 0.0475
    🔹 NumNumericChars          : 0.0366
    🔹 ExtMetaScriptLinkRT      : 0.0300
    🔹 InsecureForms            : 0.0296
    🔹 PathLevel                : 0.0267
    🔹 SubmitInfoToEmail        : 0.0230
    🔹 NumDots                  : 0.0225
    🔹 PathLength               : 0.0188
    🔹 NumSensitiveWords        : 0.0171
    🔹 UrlLength                : 0.0168



    
![png](analysis_summary_files/analysis_summary_13_1.png)
    


    
    💡 Analiz Sonucu: İlk 15 özellik toplam varyansın 85.77%'ini açıklıyor


## 🤖 Makine Öğrenmesi Modelleri



```python
# Model Development and Training
"""
This section implements multiple machine learning algorithms for phishing detection.
We'll train and evaluate different models to find the best performing approach.
"""

# Data Preparation for Machine Learning
print("DATA PREPARATION FOR MACHINE LEARNING")
print("=" * 50)

# Prepare features and target variable
X = df.drop(['CLASS_LABEL', 'id'], axis=1)  # Remove target and ID columns
y = df['CLASS_LABEL']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Feature Scaling for algorithms that require it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed successfully.")

```

    DATA PREPARATION FOR MACHINE LEARNING
    ==================================================
    Feature matrix shape: (10000, 48)
    Target vector shape: (10000,)
    Training set size: 8000 samples
    Testing set size: 2000 samples
    Feature scaling completed successfully.



```python
# Multiple Algorithm Implementation and Comparison
"""
Implement and train multiple machine learning algorithms to compare their performance
on the phishing detection task. This includes both traditional ML and ensemble methods.
"""

# Initialize multiple machine learning models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Support Vector Machine': SVC(random_state=42, probability=True)
}

# Store results for comparison
results = {}
print("MODEL TRAINING AND EVALUATION")
print("=" * 50)

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Use scaled data for SVM and Logistic Regression, original for tree-based models
    if model_name in ['Support Vector Machine', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Store results
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Model': model,
        'Predictions': y_pred,
        'Probabilities': y_prob
    }
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

print("\nAll models have been successfully trained and evaluated.")

```

    MODEL TRAINING AND EVALUATION
    ==================================================
    
    Training Random Forest...
    Random Forest - Accuracy: 0.9845, F1-Score: 0.9845, ROC-AUC: 0.9989
    
    Training Gradient Boosting...
    Gradient Boosting - Accuracy: 0.9790, F1-Score: 0.9790, ROC-AUC: 0.9970
    
    Training Logistic Regression...
    Logistic Regression - Accuracy: 0.9520, F1-Score: 0.9524, ROC-AUC: 0.9869
    
    Training Support Vector Machine...
    Support Vector Machine - Accuracy: 0.9690, F1-Score: 0.9692, ROC-AUC: 0.9953
    
    All models have been successfully trained and evaluated.


## Model Performance Analysis and Visualization



```python
# Comprehensive Performance Comparison and Visualization
"""
Create detailed performance comparisons and visualizations to identify 
the best performing model and understand model behavior.
"""

# Create performance comparison dataframe
performance_df = pd.DataFrame.from_dict(
    {name: {metric: value for metric, value in model_results.items() 
            if metric not in ['Model', 'Predictions', 'Probabilities']} 
     for name, model_results in results.items()}, 
    orient='index'
)

print("DETAILED PERFORMANCE COMPARISON")
print("=" * 60)
print(performance_df.round(4))

# Identify best performing model
best_model_name = performance_df['F1-Score'].idxmax()
best_model_f1 = performance_df.loc[best_model_name, 'F1-Score']
print(f"\nBest performing model: {best_model_name} (F1-Score: {best_model_f1:.4f})")

# Create comprehensive visualization of model performance
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Performance metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x_pos = np.arange(len(performance_df.index))
width = 0.15

for i, metric in enumerate(metrics):
    axes[0, 0].bar(x_pos + i * width, performance_df[metric], width, 
                   label=metric, alpha=0.8)

axes[0, 0].set_xlabel('Models')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Model Performance Comparison - All Metrics')
axes[0, 0].set_xticks(x_pos + width * 2)
axes[0, 0].set_xticklabels(performance_df.index, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. ROC Curves for all models
for model_name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[model_name]['Probabilities'])
    auc_score = results[model_name]['ROC-AUC']
    axes[0, 1].plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curves Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix for best model
best_predictions = results[best_model_name]['Predictions']
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')

# 4. F1-Score comparison (focused view)
f1_scores = performance_df['F1-Score'].sort_values(ascending=True)
colors = ['red' if x == best_model_name else 'skyblue' for x in f1_scores.index]
axes[1, 1].barh(range(len(f1_scores)), f1_scores.values, color=colors)
axes[1, 1].set_yticks(range(len(f1_scores)))
axes[1, 1].set_yticklabels(f1_scores.index)
axes[1, 1].set_xlabel('F1-Score')
axes[1, 1].set_title('F1-Score Comparison (Best Model Highlighted)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nModel evaluation completed. {best_model_name} achieved the highest F1-Score.")

```

    DETAILED PERFORMANCE COMPARISON
    ============================================================
                            Accuracy  Precision  Recall  F1-Score  ROC-AUC
    Random Forest             0.9845     0.9850   0.984    0.9845   0.9989
    Gradient Boosting         0.9790     0.9809   0.977    0.9790   0.9970
    Logistic Regression       0.9520     0.9449   0.960    0.9524   0.9869
    Support Vector Machine    0.9690     0.9616   0.977    0.9692   0.9953
    
    Best performing model: Random Forest (F1-Score: 0.9845)



    
![png](analysis_summary_files/analysis_summary_18_1.png)
    


    
    Model evaluation completed. Random Forest achieved the highest F1-Score.


## Detailed Classification Analysis and Insights



```python
# Detailed Classification Report and Feature Importance Analysis
"""
Generate comprehensive classification reports and analyze feature importance
to understand what characteristics make websites suspicious.
"""

# Generate detailed classification report for best model
print("DETAILED CLASSIFICATION REPORT")
print("=" * 50)
print(f"Model: {best_model_name}")
print(classification_report(y_test, best_predictions, 
                          target_names=['Legitimate', 'Phishing']))

# Feature Importance Analysis (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    best_model = results[best_model_name]['Model']
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTOP 15 MOST IMPORTANT FEATURES - {best_model_name}")
    print("=" * 60)
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['Feature']:<30}: {row['Importance']:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top 20 Most Important Features - {best_model_name}')
    plt.gca().invert_yaxis()
    
    # Add importance values to bars
    for i, v in enumerate(top_features['Importance']):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# Cross-validation analysis for model reliability
print(f"\nCROSS-VALIDATION ANALYSIS - {best_model_name}")
print("=" * 50)

# Perform 5-fold cross-validation
if best_model_name in ['Support Vector Machine', 'Logistic Regression']:
    cv_scores = cross_val_score(results[best_model_name]['Model'], 
                               X_train_scaled, y_train, cv=5, scoring='f1')
else:
    cv_scores = cross_val_score(results[best_model_name]['Model'], 
                               X_train, y_train, cv=5, scoring='f1')

print(f"Cross-validation F1-scores: {cv_scores}")
print(f"Mean CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Model stability: {'High' if cv_scores.std() < 0.02 else 'Moderate' if cv_scores.std() < 0.05 else 'Low'}")

print("\nAnalysis completed successfully.")

```

    DETAILED CLASSIFICATION REPORT
    ==================================================
    Model: Random Forest
                  precision    recall  f1-score   support
    
      Legitimate       0.98      0.98      0.98      1000
        Phishing       0.98      0.98      0.98      1000
    
        accuracy                           0.98      2000
       macro avg       0.98      0.98      0.98      2000
    weighted avg       0.98      0.98      0.98      2000
    
    
    TOP 15 MOST IMPORTANT FEATURES - Random Forest
    ============================================================
    PctExtHyperlinks              : 0.2074
    PctExtNullSelfRedirectHyperlinksRT: 0.1662
    FrequentDomainNameMismatch    : 0.0782
    PctExtResourceUrls            : 0.0686
    PctNullSelfRedirectHyperlinks : 0.0645
    NumDash                       : 0.0449
    ExtMetaScriptLinkRT           : 0.0358
    NumNumericChars               : 0.0336
    InsecureForms                 : 0.0334
    PathLevel                     : 0.0289
    SubmitInfoToEmail             : 0.0209
    NumDots                       : 0.0206
    PathLength                    : 0.0188
    UrlLength                     : 0.0168
    NumSensitiveWords             : 0.0162



    
![png](analysis_summary_files/analysis_summary_20_1.png)
    


    
    CROSS-VALIDATION ANALYSIS - Random Forest
    ==================================================
    Cross-validation F1-scores: [0.98434565 0.98186366 0.98073337 0.98561601 0.96992481]
    Mean CV F1-score: 0.9805 (+/- 0.0111)
    Model stability: High
    
    Analysis completed successfully.



```python

```


```python

```
