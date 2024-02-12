#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import style
style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')


# # Importing DataSets

# In[2]:


Bukit_Sagu_Gray=pd.read_csv("Bukit_Sagu_Gray.csv")

Bukit_Ubi_Gray=pd.read_csv("Bukit_Ubi_Gray.csv")

Panching_Gray=pd.read_csv("Panching_Gray.csv")

Semambu_Gray=pd.read_csv("Semambu_Gray.csv")

Sg_Lembing_Gray=pd.read_csv("Sg_lembing_Gray.csv")



Bukit_Sagu_Blue=pd.read_csv("Bukit_Sagu_Blue.csv")

Bukit_Ubi_Blue=pd.read_csv("Bukit_Ubi_Blue.csv")

Panching_Blue=pd.read_csv("Panching_Blue.csv")

Semambu_Blue=pd.read_csv("Semambu_Blue.csv")

Sg_Lembing_Blue=pd.read_csv("Sg_Lembing_Blue.csv")


# In[3]:


Semambu_Gray.head(3)


# # Bukit Sagu Data Visualization

# ## Gray Water

# In[4]:


Bukit_Sagu_Gray['Year'] = pd.to_datetime(Bukit_Sagu_Gray['DATE']).dt.year
Bukit_Sagu_Gray['Month'] = pd.to_datetime(Bukit_Sagu_Gray['DATE']).dt.month_name()


# In[5]:


def MonthlySumGroup(df,year,Target):
    results=df[df['Year']==year].groupby('Month').sum()[Target]
    return results


# In[6]:


Results_2018=MonthlySumGroup(Bukit_Sagu_Gray,2018,'Total Grey')
Results_2019=MonthlySumGroup(Bukit_Sagu_Gray,2019,'Total Grey')
Results_2020=MonthlySumGroup(Bukit_Sagu_Gray,2020,'Total Grey')
Results_2017=MonthlySumGroup(Bukit_Sagu_Gray,2017,'Total Grey')

fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2017'))
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Monthly sum of water per (m^3) of Bakit Sagu Gray water footprint'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# This graph shows that there are a lot of Gray water production happens in July , May and October.
# <br><br><br>

# # Water Discharge analysis

# In[7]:


Results_2018=MonthlySumGroup(Bukit_Sagu_Gray,2018,'WATER DISCHARGE')
Results_2019=MonthlySumGroup(Bukit_Sagu_Gray,2019,'WATER DISCHARGE')
Results_2020=MonthlySumGroup(Bukit_Sagu_Gray,2020,'WATER DISCHARGE')

fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Monthly sum of water per (m^3) of Bakit Sagu Gray water footprint'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Blue Water Analysis(Bukit Sagu)

# In[8]:


Bukit_Sagu_Blue['Year'] = pd.to_datetime(Bukit_Sagu_Gray['DATE']).dt.year
Bukit_Sagu_Blue['Month'] = pd.to_datetime(Bukit_Sagu_Gray['DATE']).dt.month_name()


# In[9]:


Results_2018=MonthlySumGroup(Bukit_Sagu_Blue,2018,'TOTAL BWF')
Results_2019=MonthlySumGroup(Bukit_Sagu_Blue,2019,'TOTAL BWF')
Results_2020=MonthlySumGroup(Bukit_Sagu_Blue,2020,'TOTAL BWF')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Monthly sum of  water per (m^3) of Bakit Sagu Blue water footprint'
                  ,yaxis_title="Water per volume liter",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# ## Rain Analysis (Bukit Sagu)

# In[10]:


Results_2015 = MonthlySumGroup(Bukit_Sagu_Blue,2018,'TOTAL RAINFALL (m3)')
Results_2016 = MonthlySumGroup(Bukit_Sagu_Blue,2019,'TOTAL RAINFALL (m3)')
Results_2017 = MonthlySumGroup(Bukit_Sagu_Blue,2020,'TOTAL RAINFALL (m3)')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Total Raining by month'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# ##  EVAPORATION Analysis (Bukit Sagu)

# In[11]:


Results_2015 = MonthlySumGroup(Bukit_Sagu_Blue,2018,'TOTAL EVAPORATION (m3)')
Results_2016 = MonthlySumGroup(Bukit_Sagu_Blue,2019,'TOTAL EVAPORATION (m3)')
Results_2017 = MonthlySumGroup(Bukit_Sagu_Blue,2020,'TOTAL EVAPORATION (m3)')
fig = go.Figure()
fig.add_trace(go.Scatter(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Scatter(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Scatter(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Water Evaporation by month (Bukit Sagu)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# ## Water Intake Analysis (Bukit Sagu):

# In[12]:


Results_2015 = MonthlySumGroup(Bukit_Sagu_Blue,2018,'WATER INTAKE (m3)')
Results_2016 = MonthlySumGroup(Bukit_Sagu_Blue,2019,'WATER INTAKE (m3)')
Results_2017 = MonthlySumGroup(Bukit_Sagu_Blue,2020,'WATER INTAKE (m3)')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Total Water Intake Monthly sum'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Temperature Analysis (Bukit Sagu)

# In[13]:


results = Bukit_Sagu_Blue.groupby(['Month','TEMPERATURE']).sum()['TOTAL BWF'].to_frame().reset_index()
px.bar(results,x=results['Month'],y=results['TOTAL BWF'],color="TEMPERATURE",title="Blue Water by monthly sum in a specific temperature (Bukit Sagu)")


# # Bukit Ubi Data Visualization

# ## Gray Water FootPrint (Buki Ubi)

# In[14]:


Bukit_Ubi_Gray['Year'] = pd.to_datetime(Bukit_Ubi_Gray['DATE']).dt.year
Bukit_Ubi_Gray['Month'] = pd.to_datetime(Bukit_Ubi_Gray['DATE']).dt.month_name()


# In[48]:


Results_2018=MonthlySumGroup(Bukit_Ubi_Gray,2018,'Total Grey')
Results_2019=MonthlySumGroup(Bukit_Ubi_Gray,2019,'Total Grey')
Results_2020=MonthlySumGroup(Bukit_Ubi_Gray,2020,'Total Grey')
Results_2017=MonthlySumGroup(Bukit_Ubi_Gray,2017,'Total Grey')

fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2017'))
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2019.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Monthly sum of water per (m^3) of Bukit Ubi Gray water footprint'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Blue Water Analysis(Bukit Ubi)

# In[16]:


Bukit_Ubi_Blue['Year'] = pd.to_datetime(Bukit_Ubi_Blue['DATE']).dt.year
Bukit_Ubi_Blue['Month'] = pd.to_datetime(Bukit_Ubi_Blue['DATE']).dt.month_name()


# In[17]:


Results_2018=MonthlySumGroup(Bukit_Ubi_Blue,2018,'TOTAL BWF')
Results_2019=MonthlySumGroup(Bukit_Ubi_Blue,2019,'TOTAL BWF')
Results_2020=MonthlySumGroup(Bukit_Ubi_Blue,2020,'TOTAL BWF')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Monthly Blue Water Foot Print sum per (m^3) of BUkit Ubi'
                  ,yaxis_title="Water per volume liter",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Rain Analysis (Bukit Ubi):

# In[18]:


Results_2015 = MonthlySumGroup(Bukit_Ubi_Blue,2018,'TOTAL RAINFALL (m3)')
Results_2016 = MonthlySumGroup(Bukit_Ubi_Blue,2019,'TOTAL RAINFALL (m3)')
Results_2017 = MonthlySumGroup(Bukit_Ubi_Blue,2020,'TOTAL RAINFALL (m3)')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Total Raining by month (Bukit Ubi)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))


# # Water Evaporation Analysis (Bukit Ubi):

# In[19]:


Results_2015 = MonthlySumGroup(Bukit_Ubi_Blue,2018,'TOTAL EVAPORATION (m3)')
Results_2016 = MonthlySumGroup(Bukit_Ubi_Blue,2019,'TOTAL EVAPORATION (m3)')
Results_2017 = MonthlySumGroup(Bukit_Ubi_Blue,2020,'TOTAL EVAPORATION (m3)')
fig = go.Figure()
fig.add_trace(go.Scatter(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Scatter(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Scatter(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Water Evaporation by month (Bukit Ubi)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# ## Water Intake Analysis (Bukit Ubi):

# In[20]:


Results_2015 = MonthlySumGroup(Bukit_Ubi_Blue,2018,'WATER INTAKE (m3)')
Results_2016 = MonthlySumGroup(Bukit_Ubi_Blue,2019,'WATER INTAKE (m3)')
Results_2017 = MonthlySumGroup(Bukit_Ubi_Blue,2020,'WATER INTAKE (m3)')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Total Water Intake Monthly sum (Bukit Ubi)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Temperature Analysis (Bukit Ubi)

# In[21]:


results = Bukit_Ubi_Blue.groupby(['Month','TEMPERATURE']).sum()['TOTAL BWF'].to_frame().reset_index()
px.bar(results,x=results['Month'],y=results['TOTAL BWF'],color="TEMPERATURE",title="Blue Water by monthly sum in a specific temperature (Bukit Ubi)")


# # Panching Data Visualization

# # Gray Water FootPrint (Panching)

# In[22]:


Panching_Gray['Year'] = pd.to_datetime(Panching_Gray['DATE']).dt.year
Panching_Gray['Month'] = pd.to_datetime(Panching_Gray['DATE']).dt.month_name()


# In[23]:


Results_2018=MonthlySumGroup(Panching_Gray,2018,'Total Grey')
Results_2019=MonthlySumGroup(Panching_Gray,2019,'Total Grey')
Results_2020=MonthlySumGroup(Panching_Gray,2020,'Total Grey')
Results_2017=MonthlySumGroup(Panching_Gray,2017,'Total Grey')

fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2017'))
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Gray Water Footprint Monthly sum  per (m^3) of (Panching)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Water Discharge analysis (Panching)

# In[24]:


Results_2018=MonthlySumGroup(Panching_Gray,2018,'WATER DISCHARGE')
Results_2019=MonthlySumGroup(Panching_Gray,2019,'WATER DISCHARGE')
Results_2020=MonthlySumGroup(Panching_Gray,2020,'WATER DISCHARGE')

fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Gray Water footprint Discharge Monthly sum of  per (m^3) of (Panching)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Blue Water Analysis(Panching)

# In[25]:


Panching_Blue['Year'] = pd.to_datetime(Panching_Blue['DATE']).dt.year
Panching_Blue['Month'] = pd.to_datetime(Panching_Blue['DATE']).dt.month_name()


# In[26]:


Results_2018=MonthlySumGroup(Panching_Blue,2018,'TOTAL BWF')
Results_2019=MonthlySumGroup(Panching_Blue,2019,'TOTAL BWF')
Results_2020=MonthlySumGroup(Panching_Blue,2020,'TOTAL BWF')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Monthly Blue Water Foot Print sum per (m^3) of Panching'
                  ,yaxis_title="Water per volume liter",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Rain Analysis (Panching)

# In[27]:


Results_2015 = MonthlySumGroup(Panching_Blue,2018,'TOTAL RAINFALL (m3)')
Results_2016 = MonthlySumGroup(Panching_Blue,2019,'TOTAL RAINFALL (m3)')
Results_2017 = MonthlySumGroup(Panching_Blue,2020,'TOTAL RAINFALL (m3)')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Total Raining by month (Panching)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))


# # Water Evaporation Analysis (Panching):

# In[47]:


Results_2015 = MonthlySumGroup(Panching_Blue,2018,'TOTAL EVAPORATION (m3)')
Results_2016 = MonthlySumGroup(Panching_Blue,2019,'TOTAL EVAPORATION (m3)')
Results_2017 = MonthlySumGroup(Panching_Blue,2020,'TOTAL EVAPORATION (m3)')
fig = go.Figure()
fig.add_trace(go.Scatter(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Scatter(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Scatter(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Water Evaporation by month (Panching)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Temperature Analysis (Panching)

# In[29]:


results = Panching_Blue.groupby(['Month','TEMPERATURE']).sum()['TOTAL BWF'].to_frame().reset_index()
px.bar(results,x=results['Month'],y=results['TOTAL BWF'],color="TEMPERATURE",title=" Monthly sum of BLue Water Footprint in a specific temperature (Panching)")


# # Semambu Data Visualization:

# # Gray Water FootPrint (Semambu)

# In[30]:


Semambu_Gray['Year'] = pd.to_datetime(Semambu_Gray['DATE']).dt.year
Semambu_Gray['Month'] = pd.to_datetime(Semambu_Gray['DATE']).dt.month_name()


# In[45]:


Results_2018=MonthlySumGroup(Semambu_Gray,2018,'Total Grey')
Results_2019=MonthlySumGroup(Semambu_Gray,2019,'Total Grey')
Results_2020=MonthlySumGroup(Semambu_Gray,2020,'Total Grey')
Results_2017=MonthlySumGroup(Semambu_Gray,2017,'Total Grey')

fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2017'))
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Gray Water Footprint Monthly sum  per (m^3) of (Semambu)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Water Discharge analysis (Semambu)

# In[32]:


Results_2018=MonthlySumGroup(Semambu_Gray,2018,'WATER DISCHARGE')
Results_2019=MonthlySumGroup(Semambu_Gray,2019,'WATER DISCHARGE')
Results_2020=MonthlySumGroup(Semambu_Gray,2020,'WATER DISCHARGE')

fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Gray Water footprint Discharge Monthly sum of  per (m^3) of (Semambu_Gray)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Blue Water Analysis(Semambu)

# In[33]:


Semambu_Blue['Year'] = pd.to_datetime(Semambu_Blue['DATE']).dt.year
Semambu_Blue['Month'] = pd.to_datetime(Semambu_Blue['DATE']).dt.month_name()


# In[34]:


Results_2018=MonthlySumGroup(Semambu_Blue,2018,'TOTAL BWF')
Results_2019=MonthlySumGroup(Semambu_Blue,2019,'TOTAL BWF')
Results_2020=MonthlySumGroup(Semambu_Blue,2020,'TOTAL BWF')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Monthly Blue Water Foot Print sum per (m^3) of Semambu'
                  ,yaxis_title="Water per volume liter",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Rain Analysis (Semambu)

# In[35]:


Results_2015 = MonthlySumGroup(Semambu_Blue,2018,'TOTAL RAINFALL (m3)')
Results_2016 = MonthlySumGroup(Semambu_Blue,2019,'TOTAL RAINFALL (m3)')
Results_2017 = MonthlySumGroup(Semambu_Blue,2020,'TOTAL RAINFALL (m3)')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Total Raining by month (Semambu)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))


# # Water Evaporation Analysis (Semambu):

# In[36]:


Results_2015 = MonthlySumGroup(Semambu_Blue,2018,'TOTAL EVAPORATION (m3)')
Results_2016 = MonthlySumGroup(Semambu_Blue,2019,'TOTAL EVAPORATION (m3)')
Results_2017 = MonthlySumGroup(Semambu_Blue,2020,'TOTAL EVAPORATION (m3)')
fig = go.Figure()
fig.add_trace(go.Scatter(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Scatter(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Scatter(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Water Evaporation by month (Semambu)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Temperature Analysis (Semambu):
# 

# In[37]:


results = Semambu_Blue.groupby(['Month','TEMPERATURE']).sum()['TOTAL BWF'].to_frame().reset_index()
px.bar(results,x=results['Month'],y=results['TOTAL BWF'],color="TEMPERATURE",title=" Monthly sum of BLue Water Footprint in a specific temperature (Semambu)")


# # Sg Lembing Data Visualization:

# In[38]:


Sg_Lembing_Gray['Year'] = pd.to_datetime(Sg_Lembing_Gray['DATE']).dt.year
Sg_Lembing_Gray['Month'] = pd.to_datetime(Sg_Lembing_Gray['DATE']).dt.month_name()


# In[39]:


Results_2018=MonthlySumGroup(Sg_Lembing_Gray,2018,'Total Grey')
Results_2019=MonthlySumGroup(Sg_Lembing_Gray,2019,'Total Grey')
Results_2020=MonthlySumGroup(Sg_Lembing_Gray,2020,'Total Grey')
Results_2017=MonthlySumGroup(Sg_Lembing_Gray,2017,'Total Grey')

fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2017'))
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Gray Water Footprint Monthly sum  per (m^3) of (Sg Lembing)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Blue Water Analysis(Sg Lembing):

# In[40]:


Sg_Lembing_Blue['Year'] = pd.to_datetime(Sg_Lembing_Blue['DATE']).dt.year
Sg_Lembing_Blue['Month'] = pd.to_datetime(Sg_Lembing_Blue['DATE']).dt.month_name()


# In[41]:


Results_2018=MonthlySumGroup(Sg_Lembing_Blue,2018,'TOTAL BWF')
Results_2019=MonthlySumGroup(Sg_Lembing_Blue,2019,'TOTAL BWF')
Results_2020=MonthlySumGroup(Sg_Lembing_Blue,2020,'TOTAL BWF')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2018.index,y=Results_2018.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2019.index,y=Results_2019.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2020.index,y=Results_2020.values,name='2020'))
fig.update_layout(title='Monthly Blue Water Foot Print sum per (m^3) of Sg Lembing'
                  ,yaxis_title="Water per volume liter",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Rain Analysis (Sg Lembing):

# In[42]:


Results_2015 = MonthlySumGroup(Sg_Lembing_Blue,2018,'TOTAL RAINFALL (m3)')
Results_2016 = MonthlySumGroup(Sg_Lembing_Blue,2019,'TOTAL RAINFALL (m3)')
Results_2017 = MonthlySumGroup(Sg_Lembing_Blue,2020,'TOTAL RAINFALL (m3)')
fig = go.Figure()
fig.add_trace(go.Bar(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Bar(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Bar(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Total Raining by month (Sg Lembing)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))


# # Water Evaporation Analysis (Sg Lembing):

# In[43]:


Results_2015 = MonthlySumGroup(Sg_Lembing_Blue,2018,'TOTAL EVAPORATION (m3)')
Results_2016 = MonthlySumGroup(Sg_Lembing_Blue,2019,'TOTAL EVAPORATION (m3)')
Results_2017 = MonthlySumGroup(Sg_Lembing_Blue,2020,'TOTAL EVAPORATION (m3)')
fig = go.Figure()
fig.add_trace(go.Scatter(x=Results_2015.index,y=Results_2015.values,name='2018'))
fig.add_trace(go.Scatter(x=Results_2016.index,y=Results_2016.values,name='2019'))
fig.add_trace(go.Scatter(x=Results_2017.index,y=Results_2017.values,name='2020'))
fig.update_layout(title='Water Evaporation by month (Sg Lembing)'
                  ,yaxis_title="Water per (m^3)",
                 titlefont=dict(size =20, color='black', family='Arial, sans-serif'))
fig.show()


# # Temperature Analysis (Sg Lembing):

# In[44]:


results = Sg_Lembing_Blue.groupby(['Month','TEMPERATURE']).sum()['TOTAL BWF'].to_frame().reset_index()
px.bar(results,x=results['Month'],y=results['TOTAL BWF'],color="TEMPERATURE",title=" Monthly sum of BLue Water Footprint in a specific temperature (Sg Lembing)")


# In[ ]:




