from sklearn.preprocessing import StandardScaler
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go

def plot_transformer(df_tot, features,transformer=PCA(random_state=0, n_components=2),index='index',color='Covid',additional_text = '',pre_processer=StandardScaler()):
    pca_data = pd.DataFrame(transformer.fit_transform(pre_processer.fit_transform(df_tot[features])), columns=['0', '1'])
    pca_data[color] = df_tot[color]
    pca_data['ID'] = df_tot[index]
    if additional_text != '':
        pca_data[additional_text] = df_tot[additional_text]
    else:
        pca_data[additional_text] = ''
    fig = go.Figure()
    for i in df_tot[color].unique():
        pca_data_tmp = pca_data[pca_data[color] == i]
        sc = go.Scatter(x=pca_data_tmp['0'], y=pca_data_tmp['1'], mode='markers', marker=dict(size=3),
                        name=str(i), text= pca_data_tmp['ID'] + ' ' + pca_data_tmp[additional_text] , hovertemplate="<b>%{text}</b><br>")
        fig.add_trace(sc)
    fig.update_layout(
        showlegend=True,
        title='<b>2D Representation',
        #xaxis_title="First Component",
        #yaxis_title="Second Component",
        legend={'orientation': 'h'}
    )
    #fig.layout.xaxis.title = '1st PC'
    #fig.layout.yaxis.title = '2nd PC'
    fig.layout.height = 600
    #fig.layout.width = 800
    fig.layout.margin = dict(b=50, l=20, r=20, t=60)
    fig.layout.font = dict(size=26, color='black')
    fig.show()



def plot_spectra(df,features,color='Blue',title=''):
    fig = go.Figure()
    for i,row in df.iterrows():
        fig.add_scatter(x=features, y=row[features].values, mode='lines',name=row[0],line=dict(color=color))
    fig.update_layout(title=dict(text=title))
    fig.layout.xaxis.title = 'M/z'
    fig.layout.yaxis.title = 'Intensity'
    fig.layout.height = 550
    fig.layout.width = 1200
    fig.layout.margin = dict(b=50, l=20, r=20, t=50)
    fig.layout.font = dict(size=16, color='black')
    #fig.layout.xaxis.tickmode = 'linear'
    #fig.layout.xaxis.dtick = 20
    tick = [str(i) for i in features if (float(i) - int(float(i)) <= 0) and ((int(float(i))) % 5 == 0)]
    
    
    fig.update_layout(
    xaxis=dict(tickvals=tick, ticktext=tick),
)
    
    #fig.update_xaxes(
    #tickvals=vals,
    #)
    fig.show()
    
def plot_multiple_spectra(df,features,target,title=''):
    plot_spectra(df[df[target] == 0],features,color='Blue',title=f'{title} (Negatives)')
    plot_spectra(df[df[target] == 1],features,color='Red',title=f'{title} (Positives)')

    