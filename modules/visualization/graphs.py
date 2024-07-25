import pandas as pd
import matplotlib.pyplot as plt

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from plotly.subplots import make_subplots

def standard_visualisation(model_names, metrics, s):
    summary_stats = {'model': [], 'metric': [], 'mean': [], 'median': [], 'std': []}

    for i, df in enumerate(s):
        for metric in metrics:
            summary_stats['model'].append(model_names[i])
            summary_stats['metric'].append(metric)
            summary_stats['mean'].append(df.loc['mean', metric])
            summary_stats['median'].append(df.loc['median', metric])
            summary_stats['std'].append(df.loc['std', metric])

    summary_df = pd.DataFrame(summary_stats)

    # Plotting
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, len(metrics) * 5))

    for idx, metric in enumerate(metrics):
        metric_data = summary_df[summary_df['metric'] == metric]
        
        # Plotting mean with error bars representing standard deviation
        axes[idx].bar(metric_data['model'], metric_data['mean'], yerr=metric_data['std'], capsize=5, label='Mean')
        
        # Plotting median as a thick black line
        for i, model in enumerate(metric_data['model']):
            median_value = metric_data[metric_data['model'] == model]['median'].values[0]
            axes[idx].plot([i - 0.2, i + 0.2], [median_value, median_value], color='black', linewidth=4, label='Median' if i == 0 else "")

        axes[idx].set_title(f'{metric.upper()} Comparison')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#############################################################

def interactive_visualisation(model_names, metrics, s):

    summary_stats = {'model': [], 'metric': [], 'mean': [], 'median': [], 'std': []}

    for i, df in enumerate(s):
        for metric in metrics:
            summary_stats['model'].append(model_names[i])
            summary_stats['metric'].append(metric)
            summary_stats['mean'].append(df.loc['mean', metric])
            summary_stats['median'].append(df.loc['median', metric])
            summary_stats['std'].append(df.loc['std', metric])

    summary_df = pd.DataFrame(summary_stats)

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Model Metrics Comparison"),
        html.Div([
            html.Label('Select Metrics:'),
            dcc.Dropdown(
                id='metric-selector',
                options=[{'label': metric.upper(), 'value': metric} for metric in metrics],
                value=metrics,
                multi=True
            ),
            html.Label('Select Models:'),
            dcc.Dropdown(
                id='model-selector',
                options=[{'label': model, 'value': model} for model in model_names],
                value=model_names,
                multi=True
            )
        ], style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='metric-graph')
    ])

    @app.callback(
        Output('metric-graph', 'figure'),
        [Input('metric-selector', 'value'),
        Input('model-selector', 'value')]
    )
    def update_graph(selected_metrics, selected_models):
        fig = go.Figure()

        model_color_map = {}
        colors = ["red", "green", "blue", "black"]
        for i, mn in enumerate(model_names):
            model_color_map[mn] = colors[i]

        """    
        model_color_map = {
            'Arima': 'red',
            'Lag LLama': 'green',
            'Autoregressor': 'blue',
            "fine-tuned Llama": "black"
        }
        """

        for metric in selected_metrics:
            for model in selected_models:
                metric_data = summary_df[(summary_df['model'] == model) & (summary_df['metric'] == metric)]
                color = model_color_map[model]

                fig.add_trace(go.Bar(
                    x=[metric.upper()],
                    y=metric_data['mean'],
                    error_y=dict(type='data', array=metric_data['std'], visible=True),
                    name=f'{model} Mean',
                    hovertext=f'{model} - {metric.upper()}',
                    hoverinfo='text+y',
                    legendgroup=model,
                    marker_color=color,
                    offsetgroup=model
                ))

                # Adding median as a thick black line
                fig.add_trace(go.Scatter(
                    x=[metric.upper()],
                    y=metric_data['median'],
                    mode='markers',
                    marker=dict(color='black', size=10, symbol='line-ns-open'),
                    name=f'{model} Median',
                    hoverinfo='skip',
                    legendgroup=model,
                    showlegend=False,
                    offsetgroup=model
                ))

        # Updating layout
        fig.update_layout(
            title='Model Metrics Comparison',
            barmode='group',
            xaxis=dict(title='Metrics', categoryorder='array', categoryarray=[metric.upper() for metric in selected_metrics]),
            yaxis=dict(title='Metric Value'),
            legend=dict(orientation='h', y=-0.2),
            height=500,
            width=800
        )

        return fig

    app.run_server(debug=True)


##################################################################################

def prediction_visualisation(model_names, p, a):
    def visualize_predictions(p, a):
        """
        Visualizes the predictions and actual values of the time series.

        Parameters:
        p (list of pd.DataFrame): List containing the dataframes for the models.
        a (pd.DataFrame): DataFrame containing the actual values.

        Returns:
        fig: Plotly figure object.
        """
        model_mapping = {}
        for i, v in enumerate(model_names):
            model_mapping[v] = p[i]

        models = list(model_mapping.keys())
        folds = list(range(len(a)))

        fig = make_subplots(rows=1, cols=1)
        
        # Add traces for all models and folds, but make them invisible initially
        for model in models:
            for fold in folds:
                predicted_values = model_mapping[model].iloc[fold].values
                fig.add_trace(go.Scatter(x=list(range(1, len(predicted_values)+1)), 
                                        y=predicted_values, 
                                        mode='lines+markers', 
                                        name=f'{model} Fold {fold}',
                                        visible=False))
                
        for fold in folds:
            actual_values = a.iloc[fold].values
            fig.add_trace(go.Scatter(x=list(range(1, len(actual_values)+1)), 
                                    y=actual_values, 
                                    mode='lines+markers', 
                                    name=f'Actual Fold {fold}',
                                    visible=False))

        # Create buttons for folds
        fold_buttons = []
        for fold in folds:
            visibility = []
            for trace in fig.data:
                if f'Fold {fold}' in trace.name:
                    visibility.append(True)
                else:
                    visibility.append(False)
            button = dict(
                label=f'Fold {fold}',
                method='update',
                args=[{'visible': visibility},
                    {'title': f'Actual values and Model predictions for Fold {fold}'}]
            )
            fold_buttons.append(button)

        fig.update_layout(
            updatemenus=[
                dict(active=0,
                    buttons=fold_buttons,
                    direction='down',
                    showactive=True,
                    x=0.57,
                    xanchor="left",
                    y=1.17,
                    yanchor="top")
            ],
            title=f'Actual values and Model predictions for ',
            height=600,
            width=800,
            xaxis_title="Prediction Horizon",
            yaxis_title=f"target variable value"
        )
        
        # Initially show the first fold
        fig.update_traces(visible=True, selector=dict(name=f'Arima Fold 0'))
        fig.update_traces(visible=True, selector=dict(name=f'Lag llama Fold 0'))
        fig.update_traces(visible=True, selector=dict(name=f'Autoregressor Fold 0'))
        fig.update_traces(visible=True, selector=dict(name=f'Actual Fold 0'))

        return fig

    # Example usage
    fig = visualize_predictions(p, a)
    fig.show()