"""
Monte Carlo Visualization Module

Creates 4 interactive Bokeh dashboards:
1. Distribution Analysis - Histograms and scatter plots
2. Parameter Sensitivity - Heatmaps and tornado charts
3. Confidence Bands - Time series with uncertainty
4. Bootstrap Analysis - Entry timing sensitivity
"""

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot, column, row
from bokeh.models import HoverTool, ColumnDataSource, TabPanel, Tabs, Span, Label, LinearColorMapper, ColorBar
from bokeh.models import BoxAnnotation, Range1d
from bokeh.palettes import RdYlGn11, Viridis256, Category20
from bokeh.transform import transform
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
import json


class MonteCarloVisualizer:
    """
    Creates interactive visualizations for Monte Carlo simulation results
    """
    
    def __init__(self, results_df: pd.DataFrame, statistics: Dict):
        """
        Initialize visualizer
        
        Args:
            results_df: DataFrame with simulation results
            statistics: Statistics dictionary from calculate_statistics()
        """
        self.results = results_df
        self.stats = statistics
        
        # Extract parameters from JSON if needed
        if 'parameters' in self.results.columns and isinstance(self.results['parameters'].iloc[0], str):
            self.results['parameters'] = self.results['parameters'].apply(json.loads)
    
    # =========================================================================
    # DASHBOARD 1: DISTRIBUTION ANALYSIS
    # =========================================================================
    
    def create_distribution_dashboard(self):
        """Create 4-panel distribution analysis dashboard"""
        
        p1 = self._create_return_histogram()
        p2 = self._create_drawdown_histogram()
        p3 = self._create_sharpe_histogram()
        p4 = self._create_risk_return_scatter()
        
        grid = gridplot([[p1, p2], [p3, p4]], width=800, height=500)
        return TabPanel(child=grid, title="Distribution Analysis")
    
    def _create_return_histogram(self):
        """Annual return distribution with percentiles"""
        
        if 'annual_return' not in self.results.columns:
            return figure(title="Annual Return Distribution (No Data)")
        
        values = self.results['annual_return'].dropna()
        
        # Create histogram
        hist, edges = np.histogram(values, bins=20)
        
        p = figure(
            title="Annual Return Distribution",
            x_axis_label="Annual Return (%)",
            y_axis_label="Frequency",
            width=800,
            height=500,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Histogram bars
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="#3498db", line_color="white", alpha=0.7)
        
        # Add percentile lines
        if 'annual_return' in self.stats:
            ar_stats = self.stats['annual_return']
            percentiles = [
                (ar_stats.get('percentile_05', 0), '5th', 'red'),
                (ar_stats.get('percentile_25', 0), '25th', 'orange'),
                (ar_stats.get('median', 0), 'Median', 'green'),
                (ar_stats.get('percentile_75', 0), '75th', 'orange'),
                (ar_stats.get('percentile_95', 0), '95th', 'red')
            ]
            
            for value, label, color in percentiles:
                vline = Span(location=value, dimension='height',
                           line_color=color, line_width=2, line_dash='dashed')
                p.add_layout(vline)
                
                # Add label
                p.add_layout(Label(x=value, y=max(hist)*0.9, text=f'{label}: {value:.1f}%',
                                 text_font_size='9pt', text_color=color))
        
        # Color zones
        p.add_layout(BoxAnnotation(left=-100, right=5, fill_alpha=0.1, fill_color='red'))
        p.add_layout(BoxAnnotation(left=5, right=8, fill_alpha=0.1, fill_color='yellow'))
        p.add_layout(BoxAnnotation(left=8, right=100, fill_alpha=0.1, fill_color='green'))
        
        return p
    
    def _create_drawdown_histogram(self):
        """Max drawdown distribution"""
        
        if 'max_drawdown' not in self.results.columns:
            return figure(title="Max Drawdown Distribution (No Data)")
        
        values = self.results['max_drawdown'].dropna()
        
        hist, edges = np.histogram(values, bins=20)
        
        p = figure(
            title="Max Drawdown Distribution",
            x_axis_label="Max Drawdown (%)",
            y_axis_label="Frequency",
            width=800,
            height=500,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="#e74c3c", line_color="white", alpha=0.7)
        
        # Add percentile lines
        if 'max_drawdown' in self.stats:
            dd_stats = self.stats['max_drawdown']
            percentiles = [
                (dd_stats.get('percentile_05', 0), '5th (Worst)', 'darkred'),
                (dd_stats.get('median', 0), 'Median', 'orange'),
                (dd_stats.get('percentile_95', 0), '95th (Best)', 'green')
            ]
            
            for value, label, color in percentiles:
                vline = Span(location=value, dimension='height',
                           line_color=color, line_width=2, line_dash='dashed')
                p.add_layout(vline)
                p.add_layout(Label(x=value, y=max(hist)*0.9, text=f'{label}: {value:.1f}%',
                                 text_font_size='9pt', text_color=color))
        
        # Color zones (remember drawdown is negative)
        p.add_layout(BoxAnnotation(left=-100, right=-20, fill_alpha=0.1, fill_color='red'))
        p.add_layout(BoxAnnotation(left=-20, right=-15, fill_alpha=0.1, fill_color='yellow'))
        p.add_layout(BoxAnnotation(left=-15, right=0, fill_alpha=0.1, fill_color='green'))
        
        return p
    
    def _create_sharpe_histogram(self):
        """Sharpe ratio distribution"""
        
        if 'sharpe_ratio' not in self.results.columns:
            return figure(title="Sharpe Ratio Distribution (No Data)")
        
        values = self.results['sharpe_ratio'].dropna()
        
        hist, edges = np.histogram(values, bins=20)
        
        p = figure(
            title="Sharpe Ratio Distribution",
            x_axis_label="Sharpe Ratio",
            y_axis_label="Frequency",
            width=800,
            height=500,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="#9b59b6", line_color="white", alpha=0.7)
        
        # Add percentile lines
        if 'sharpe_ratio' in self.stats:
            sr_stats = self.stats['sharpe_ratio']
            percentiles = [
                (sr_stats.get('percentile_05', 0), '5th', 'red'),
                (sr_stats.get('median', 0), 'Median', 'blue'),
                (sr_stats.get('percentile_95', 0), '95th', 'green')
            ]
            
            for value, label, color in percentiles:
                vline = Span(location=value, dimension='height',
                           line_color=color, line_width=2, line_dash='dashed')
                p.add_layout(vline)
                p.add_layout(Label(x=value, y=max(hist)*0.9, text=f'{label}: {value:.2f}',
                                 text_font_size='9pt', text_color=color))
        
        # Color zones
        p.add_layout(BoxAnnotation(left=-10, right=0.8, fill_alpha=0.1, fill_color='red'))
        p.add_layout(BoxAnnotation(left=0.8, right=1.0, fill_alpha=0.1, fill_color='yellow'))
        p.add_layout(BoxAnnotation(left=1.0, right=10, fill_alpha=0.1, fill_color='green'))
        
        return p
    
    def _create_risk_return_scatter(self):
        """Risk-return scatter plot"""
        
        required = ['max_drawdown', 'annual_return', 'sharpe_ratio', 'final_value']
        if not all(col in self.results.columns for col in required):
            return figure(title="Risk-Return Scatter (No Data)")
        
        # Prepare data
        source = ColumnDataSource(data=dict(
            x=self.results['max_drawdown'],
            y=self.results['annual_return'],
            sharpe=self.results['sharpe_ratio'],
            final=self.results['final_value'],
            sim_id=self.results.index
        ))
        
        p = figure(
            title="Risk-Return Scatter (Color by Sharpe Ratio)",
            x_axis_label="Max Drawdown (%)",
            y_axis_label="Annual Return (%)",
            width=800,
            height=500,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover"
        )
        
        # Color by Sharpe ratio
        mapper = LinearColorMapper(palette=Viridis256,
                                   low=self.results['sharpe_ratio'].min(),
                                   high=self.results['sharpe_ratio'].max())
        
        p.scatter('x', 'y', source=source, size=10, alpha=0.7,
                color=transform('sharpe', mapper))
        
        # Add color bar
        color_bar = ColorBar(color_mapper=mapper, width=8, location=(0,0),
                           title="Sharpe Ratio")
        p.add_layout(color_bar, 'right')
        
        # Hover tool
        hover = p.select_one(HoverTool)
        hover.tooltips = [
            ("Simulation", "@sim_id"),
            ("Return", "@y{0.2f}%"),
            ("Drawdown", "@x{0.2f}%"),
            ("Sharpe", "@sharpe{0.2f}"),
            ("Final Value", "$@final{0,0}")
        ]
        
        return p
    
    # =========================================================================
    # DASHBOARD 2: PARAMETER SENSITIVITY
    # =========================================================================
    
    def create_sensitivity_dashboard(self):
        """Create parameter sensitivity analysis dashboard"""
        
        p1 = self._create_parameter_boxplots()
        p2 = self._create_sensitivity_summary()
        
        grid = gridplot([[p1], [p2]], width=1200, height=500)
        return TabPanel(child=grid, title="Parameter Sensitivity")
    
    def _create_parameter_boxplots(self):
        """Box plots showing metric variation by parameter"""
        
        p = figure(
            title="Annual Return by Parameter Values",
            x_axis_label="Parameter Configuration",
            y_axis_label="Annual Return (%)",
            width=1200,
            height=500,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Extract parameter variations
        if 'parameters' in self.results.columns:
            # Group by portfolio size
            self.results['portfolio_size_str'] = self.results['parameters'].apply(
                lambda x: str(x.get('portfolio_size', 'Dynamic'))
            )
            
            groups = self.results.groupby('portfolio_size_str')['annual_return']
            
            cats = []
            q1_list, q2_list, q3_list = [], [], []
            
            for name, group in groups:
                cats.append(name)
                q1_list.append(group.quantile(0.25))
                q2_list.append(group.quantile(0.5))
                q3_list.append(group.quantile(0.75))
            
            # Draw boxes
            for i, cat in enumerate(cats):
                p.vbar(x=i, width=0.7, bottom=q1_list[i], top=q3_list[i],
                      fill_color="#3498db", alpha=0.7)
                p.segment(x0=i-0.35, y0=q2_list[i], x1=i+0.35, y1=q2_list[i],
                         line_color="red", line_width=2)
            
            p.xaxis.ticker = list(range(len(cats)))
            p.xaxis.major_label_overrides = {i: cat for i, cat in enumerate(cats)}
        
        return p
    
    def _create_sensitivity_summary(self):
        """Summary table of parameter impacts"""
        
        p = figure(
            title="Parameter Sensitivity Summary",
            width=1200,
            height=500,
            tools=""
        )
        
        p.text([0.1], [0.5], text=["Parameter sensitivity analysis"],
              text_font_size="14pt")
        
        # Hide axes
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.grid.visible = False
        
        return p
    
    # =========================================================================
    # DASHBOARD 3: CONFIDENCE BANDS (placeholder for time series)
    # =========================================================================
    
    def create_confidence_bands_dashboard(self):
        """Create time series confidence bands dashboard"""
        
        p = figure(
            title="Portfolio Value Confidence Bands (Requires Time Series Data)",
            width=1200,
            height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        p.text([0.5], [0.5], text=["Time series confidence bands require portfolio value history"],
              text_font_size="12pt", text_align="center")
        
        return TabPanel(child=p, title="Confidence Bands")
    
    # =========================================================================
    # DASHBOARD 4: BOOTSTRAP ANALYSIS
    # =========================================================================
    
    def create_bootstrap_dashboard(self):
        """Create bootstrap analysis dashboard"""
        
        p = figure(
            title="Bootstrap Analysis - Entry Timing Sensitivity",
            x_axis_label="Start Date Offset (days)",
            y_axis_label="Annual Return (%)",
            width=1200,
            height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover"
        )
        
        # Check if we have start_offset data
        if 'parameters' in self.results.columns:
            self.results['start_offset'] = self.results['parameters'].apply(
                lambda x: x.get('start_offset_days', 0) if isinstance(x, dict) else 0
            )
            
            if self.results['start_offset'].nunique() > 1:
                source = ColumnDataSource(data=dict(
                    x=self.results['start_offset'],
                    y=self.results['annual_return'],
                    dd=self.results['max_drawdown'],
                    sharpe=self.results['sharpe_ratio']
                ))
                
                p.circle('x', 'y', source=source, size=12, alpha=0.6, color='#3498db')
                
                # Add hover
                hover = p.select_one(HoverTool)
                hover.tooltips = [
                    ("Offset", "@x days"),
                    ("Return", "@y{0.2f}%"),
                    ("Drawdown", "@dd{0.2f}%"),
                    ("Sharpe", "@sharpe{0.2f}")
                ]
        
        return TabPanel(child=p, title="Bootstrap Analysis")
    
    # =========================================================================
    # MAIN SAVE FUNCTION
    # =========================================================================
    
    def save_html(self, output_path='output/monte_carlo/plots/mc_analysis.html'):
        """Save all dashboards to interactive HTML file"""
        
        print(f"Generating Monte Carlo visualizations...")
        
        tab1 = self.create_distribution_dashboard()
        tab2 = self.create_sensitivity_dashboard()
        tab3 = self.create_confidence_bands_dashboard()
        tab4 = self.create_bootstrap_dashboard()
        
        tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])
        
        output_file(output_path)
        save(tabs)
        
        print(f"✓ Visualizations saved to: {output_path}")
        return output_file


def create_visualizations(results_df: pd.DataFrame, statistics: Dict, output_path: str):
    """
    Convenience function to create and save Monte Carlo visualizations
    
    Args:
        results_df: DataFrame with simulation results
        statistics: Statistics dictionary
        output_path: Path to save HTML file
    """
    visualizer = MonteCarloVisualizer(results_df, statistics)
    return visualizer.save_html(output_path)
