# Databricks notebook source
#Carregar os dados
#tabela custos intercambio
df_table = spark.read.table("prod.bases_custo.intercambio")
#limitar os dados em 10k linhas
df_table = df_table.limit(100000)
#transformar em dataframe pandas
df = df_table.toPandas()

# COMMAND ----------

display(df)

# COMMAND ----------

df.head(3)

# COMMAND ----------

df.shape

# COMMAND ----------

df.info()

# COMMAND ----------

'''
importar bibliotecas
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# COMMAND ----------

df.info()

# COMMAND ----------



# COMMAND ----------

df['VR_NOVO'] = df['VR_NOVO'].astype('float')
df['VR_DESP_REAL'] = df['VR_DESP_REAL'].astype('float')
df['VR_COBRADO'] = df['VR_COBRADO'].astype('float')
df['QT_INF'] = df['QT_INF'].astype('float')

# COMMAND ----------

df.info()

# COMMAND ----------

df.describe()

# COMMAND ----------

# Na representação gráfica

sns.histplot(df['VR_DESP_REAL'])

# plotando a mediana e os quartis
plt.axvline(df['VR_DESP_REAL'].quantile(0.25), color="orange", label="q1")
plt.axvline(df['VR_DESP_REAL'].quantile(0.5), color="green", label="q2=mediana")
plt.axvline(df['VR_DESP_REAL'].quantile(0.75), color="pink", label="q3")

plt.title('Histograma de quantidade informada')

# Cria uma legenda
plt.legend()

plt.show()

# COMMAND ----------

sns.boxplot(data=df, x='VR_DESP_REAL')
plt.show()

# COMMAND ----------

df['VR_DESP_REAL'].median()

# COMMAND ----------

df1 = df[df['VR_DESP_REAL'] < 150]

# COMMAND ----------

df['VR_DESP_REAL'].median()

# COMMAND ----------

sns.boxplot(data=df1, x='VR_DESP_REAL')
plt.axvline(np.quantile(df1['VR_DESP_REAL'], 0.80), color="yellow", label="Primeiro Quartil")
plt.title('Gráfico boxplot DESPESA REAL')
plt.legend()
plt.show()

# COMMAND ----------

sns.boxplot(data=df, x='VR_DESP_REAL')
plt.show()

# COMMAND ----------

df.describe()

# COMMAND ----------

df_filtrado = df[df['VR_DESP_REAL'] < 500]

# COMMAND ----------

sns.boxplot(data=df_filtrado, x='VR_DESP_REAL')

IQR = np.quantile(df_filtrado['VR_DESP_REAL'], 0.75) - np.quantile(df_filtrado['VR_DESP_REAL'], 0.25)
plt.axvline(np.quantile(df_filtrado['VR_DESP_REAL'], 0.25), color="magenta", label="Q1", alpha=0.4)
plt.axvline(np.quantile(df_filtrado['VR_DESP_REAL'], 0.75), color="orange", label="Q3", alpha=0.4)
plt.axvline(np.mean(df_filtrado['VR_DESP_REAL']), color="green", label="Média", alpha=0.4)
plt.axvline(np.median(df_filtrado['VR_DESP_REAL']), color="red", label="Mediana", alpha=0.4)
plt.axvline(np.quantile(df_filtrado['VR_DESP_REAL'], 0.75) + 1.5 * IQR, color = "pink", label = "Upper")

plt.legend()
plt.show()

# COMMAND ----------

df.info()

# COMMAND ----------

df.head()

# COMMAND ----------

df.describe()

# COMMAND ----------

df_correlacao = df.corr(method = 'pearson')

# COMMAND ----------

df_corr_spearmann = df.corr(method = 'spearman')

# COMMAND ----------

df_corr_spearmann

# COMMAND ----------

df_correlacao

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt


# COMMAND ----------

plt.figure(figsize=(7,5))
sns.heatmap(df_correlacao, annot=True, cmap='seismic')
plt.title ('Mapa de calor de correlação')
plt.show()

# COMMAND ----------


