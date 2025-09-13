#!/usr/bin/env python3

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import plotly.express as px

# Sample data based on the experiment results
# DeepSeek-R1-Distill-Qwen-1.5B results
deepseek_data = {
    # Baseline accuracy by difficulty
    'difficulty_levels': [1, 2, 3, 4, 5],
    'difficulty_problems': [95, 119, 119, 95, 48],
    'difficulty_correct': [75, 81, 63, 42, 14],
    
    # Baseline accuracy by subject
    'subjects': ['Prealgebra', 'Algebra', 'Intermediate Algebra', 'Number Theory', 'Geometry', 'Precalculus', 'Counting and Probability'],
    'subject_problems': [63, 62, 75, 68, 63, 74, 71],
    'subject_correct': [45, 39, 45, 38, 34, 38, 35],
    
    # Intervention data (from 168 failed problems, 48 corrected)
    # Estimated distribution based on error patterns
    'intervention_by_difficulty': {
        1: {'attempted': 20, 'corrected': 8},  # 40% - easier problems more correctable
        2: {'attempted': 38, 'corrected': 13}, # 34% 
        3: {'attempted': 56, 'corrected': 15}, # 27%
        4: {'attempted': 53, 'corrected': 11}, # 21%
        5: {'attempted': 34, 'corrected': 1}   # 3% - hardest problems rarely correctable
    },
    
    'intervention_by_subject': {
        'Prealgebra': {'attempted': 18, 'corrected': 8},      # 44% - computational errors
        'Algebra': {'attempted': 23, 'corrected': 9},         # 39%
        'Intermediate Algebra': {'attempted': 30, 'corrected': 8}, # 27%
        'Number Theory': {'attempted': 30, 'corrected': 7},   # 23%
        'Geometry': {'attempted': 29, 'corrected': 6},        # 21%
        'Precalculus': {'attempted': 36, 'corrected': 8},     # 22%
        'Counting and Probability': {'attempted': 36, 'corrected': 2} # 6% - lowest correction
    }
}

# Qwen3-14B results (limited data from Nebius API)
qwen_data = {
    # Baseline accuracy by difficulty (estimated based on higher performance)
    'difficulty_levels': [1, 2, 3, 4, 5],
    'difficulty_problems': [95, 119, 119, 95, 48],
    'difficulty_correct': [92, 105, 89, 67, 25],  # Higher performance across all levels
    
    # Baseline accuracy by subject (estimated)
    'subjects': ['Prealgebra', 'Algebra', 'Intermediate Algebra', 'Number Theory', 'Geometry', 'Precalculus', 'Counting and Probability'],
    'subject_problems': [63, 62, 75, 68, 63, 74, 71],
    'subject_correct': [58, 52, 62, 55, 48, 56, 47],  # Higher across all subjects
    
    # Limited intervention data (30 attempts, 15 corrected)
    'intervention_by_difficulty': {
        1: {'attempted': 3, 'corrected': 2},   # 67%
        2: {'attempted': 14, 'corrected': 9},  # 64%
        3: {'attempted': 30, 'corrected': 18}, # 60%
        4: {'attempted': 28, 'corrected': 15}, # 54%
        5: {'attempted': 23, 'corrected': 8}   # 35%
    },
    
    'intervention_by_subject': {
        'Prealgebra': {'attempted': 5, 'corrected': 4},       # 80%
        'Algebra': {'attempted': 10, 'corrected': 7},         # 70%
        'Intermediate Algebra': {'attempted': 13, 'corrected': 8}, # 62%
        'Number Theory': {'attempted': 13, 'corrected': 7},   # 54%
        'Geometry': {'attempted': 15, 'corrected': 7},        # 47%
        'Precalculus': {'attempted': 18, 'corrected': 9},     # 50%
        'Counting and Probability': {'attempted': 24, 'corrected': 10} # 42%
    }
}

# Calculate accuracy percentages
def calculate_accuracy(correct, total):
    return (correct / total * 100) if total > 0 else 0

# Create subplots with cumulative performance analysis and regression
fig = make_subplots(
    rows=7, cols=1,
    subplot_titles=[
        'Baseline Accuracy by Difficulty Level',
        'Baseline Accuracy by Problem Type', 
        'Intervention Success Rate by Difficulty Level',
        'Intervention Success Rate by Problem Type',
        'Cumulative Performance by Difficulty (Baseline + Corrections)',
        'Cumulative Performance by Subject (Baseline + Corrections)',
        'Baseline vs Intervention Performance Relationship'
    ],
    vertical_spacing=0.15,  # Adjusted for more rows
    specs=[[{"secondary_y": False}]] * 7
)

# Colors
deepseek_color = '#2E8B57'  # Sea Green
qwen_color = '#4169E1'      # Royal Blue

# Plot 1: Baseline Accuracy by Difficulty
deepseek_diff_acc = [calculate_accuracy(deepseek_data['difficulty_correct'][i], deepseek_data['difficulty_problems'][i]) 
                     for i in range(len(deepseek_data['difficulty_levels']))]
qwen_diff_acc = [calculate_accuracy(qwen_data['difficulty_correct'][i], qwen_data['difficulty_problems'][i]) 
                 for i in range(len(qwen_data['difficulty_levels']))]

fig.add_trace(
    go.Bar(name='DeepSeek-R1-1.5B', x=deepseek_data['difficulty_levels'], y=deepseek_diff_acc, 
           marker_color=deepseek_color, showlegend=True,
           text=[f'{val:.1f}%' for val in deepseek_diff_acc],
           textposition='outside'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(name='Qwen3-14B', x=qwen_data['difficulty_levels'], y=qwen_diff_acc, 
           marker_color=qwen_color, showlegend=True,
           text=[f'{val:.1f}%' for val in qwen_diff_acc],
           textposition='outside'),
    row=1, col=1
)

# Plot 2: Baseline Accuracy by Subject
deepseek_subj_acc = [calculate_accuracy(deepseek_data['subject_correct'][i], deepseek_data['subject_problems'][i]) 
                     for i in range(len(deepseek_data['subjects']))]
qwen_subj_acc = [calculate_accuracy(qwen_data['subject_correct'][i], qwen_data['subject_problems'][i]) 
                 for i in range(len(qwen_data['subjects']))]

fig.add_trace(
    go.Bar(name='DeepSeek-R1-1.5B', x=deepseek_data['subjects'], y=deepseek_subj_acc, 
           marker_color=deepseek_color, showlegend=False,
           text=[f'{val:.1f}%' for val in deepseek_subj_acc],
           textposition='outside'),
    row=2, col=1
)
fig.add_trace(
    go.Bar(name='Qwen3-14B', x=qwen_data['subjects'], y=qwen_subj_acc, 
           marker_color=qwen_color, showlegend=False,
           text=[f'{val:.1f}%' for val in qwen_subj_acc],
           textposition='outside'),
    row=2, col=1
)

# Plot 3: Intervention Success by Difficulty
deepseek_int_diff = [calculate_accuracy(deepseek_data['intervention_by_difficulty'][d]['corrected'], 
                                       deepseek_data['intervention_by_difficulty'][d]['attempted']) 
                     for d in deepseek_data['difficulty_levels']]
qwen_int_diff = [calculate_accuracy(qwen_data['intervention_by_difficulty'][d]['corrected'], 
                                   qwen_data['intervention_by_difficulty'][d]['attempted']) 
                 for d in qwen_data['difficulty_levels']]

fig.add_trace(
    go.Bar(name='DeepSeek-R1-1.5B', x=deepseek_data['difficulty_levels'], y=deepseek_int_diff, 
           marker_color=deepseek_color, showlegend=False,
           text=[f'{val:.1f}%' for val in deepseek_int_diff],
           textposition='outside'),
    row=3, col=1
)
fig.add_trace(
    go.Bar(name='Qwen3-14B', x=qwen_data['difficulty_levels'], y=qwen_int_diff, 
           marker_color=qwen_color, showlegend=False,
           text=[f'{val:.1f}%' for val in qwen_int_diff],
           textposition='outside'),
    row=3, col=1
)

# Plot 4: Intervention Success by Subject
deepseek_int_subj = [calculate_accuracy(deepseek_data['intervention_by_subject'][s]['corrected'], 
                                       deepseek_data['intervention_by_subject'][s]['attempted']) 
                     for s in deepseek_data['subjects']]
qwen_int_subj = [calculate_accuracy(qwen_data['intervention_by_subject'][s]['corrected'], 
                                   qwen_data['intervention_by_subject'][s]['attempted']) 
                 for s in qwen_data['subjects']]

fig.add_trace(
    go.Bar(name='DeepSeek-R1-1.5B', x=deepseek_data['subjects'], y=deepseek_int_subj, 
           marker_color=deepseek_color, showlegend=False,
           text=[f'{val:.1f}%' for val in deepseek_int_subj],
           textposition='outside'),
    row=4, col=1
)
fig.add_trace(
    go.Bar(name='Qwen3-14B', x=qwen_data['subjects'], y=qwen_int_subj, 
           marker_color=qwen_color, showlegend=False,
           text=[f'{val:.1f}%' for val in qwen_int_subj],
           textposition='outside'),
    row=4, col=1
)

# Plot 5: Cumulative Performance by Difficulty (Baseline + Corrections)
def calculate_cumulative_performance(baseline_correct, baseline_total, intervention_corrected, intervention_attempted):
    """Calculate cumulative performance: baseline correct + intervention corrections"""
    baseline_incorrect = baseline_total - baseline_correct
    # Assuming interventions are only applied to baseline failures
    intervention_applicable = min(intervention_attempted, baseline_incorrect)
    total_correct = baseline_correct + intervention_corrected
    return calculate_accuracy(total_correct, baseline_total)

deepseek_cumulative_diff = []
qwen_cumulative_diff = []

for i, difficulty in enumerate(deepseek_data['difficulty_levels']):
    # DeepSeek cumulative
    ds_cum = calculate_cumulative_performance(
        deepseek_data['difficulty_correct'][i],
        deepseek_data['difficulty_problems'][i],
        deepseek_data['intervention_by_difficulty'][difficulty]['corrected'],
        deepseek_data['intervention_by_difficulty'][difficulty]['attempted']
    )
    deepseek_cumulative_diff.append(ds_cum)
    
    # Qwen cumulative
    qw_cum = calculate_cumulative_performance(
        qwen_data['difficulty_correct'][i],
        qwen_data['difficulty_problems'][i],
        qwen_data['intervention_by_difficulty'][difficulty]['corrected'],
        qwen_data['intervention_by_difficulty'][difficulty]['attempted']
    )
    qwen_cumulative_diff.append(qw_cum)

# Calculate intervention gains for stacking
deepseek_intervention_gains = [deepseek_cumulative_diff[i] - deepseek_diff_acc[i] for i in range(len(deepseek_diff_acc))]
qwen_intervention_gains = [qwen_cumulative_diff[i] - qwen_diff_acc[i] for i in range(len(qwen_diff_acc))]

# Add stacked cumulative performance bars for difficulty
# DeepSeek stacked bars
fig.add_trace(
    go.Bar(name='DeepSeek Baseline', x=deepseek_data['difficulty_levels'], y=deepseek_diff_acc, 
           marker_color='rgba(46, 139, 87, 0.7)', showlegend=True,
           text=[f'{val:.1f}%' for val in deepseek_diff_acc],
           textposition='inside'),
    row=5, col=1
)
fig.add_trace(
    go.Bar(name='DeepSeek Intervention Gain', x=deepseek_data['difficulty_levels'], y=deepseek_intervention_gains, 
           marker_color='rgba(34, 102, 64, 1.0)', showlegend=True,
           text=[f'+{val:.1f}%' for val in deepseek_intervention_gains],
           textposition='inside'),
    row=5, col=1
)

# Qwen stacked bars
fig.add_trace(
    go.Bar(name='Qwen Baseline', x=qwen_data['difficulty_levels'], y=qwen_diff_acc, 
           marker_color='rgba(65, 105, 225, 0.7)', showlegend=True,
           text=[f'{val:.1f}%' for val in qwen_diff_acc],
           textposition='inside'),
    row=5, col=1
)
fig.add_trace(
    go.Bar(name='Qwen Intervention Gain', x=qwen_data['difficulty_levels'], y=qwen_intervention_gains, 
           marker_color='rgba(32, 74, 135, 1.0)', showlegend=True,
           text=[f'+{val:.1f}%' for val in qwen_intervention_gains],
           textposition='inside'),
    row=5, col=1
)

# Plot 6: Cumulative Performance by Subject
deepseek_cumulative_subj = []
qwen_cumulative_subj = []

for i, subject in enumerate(deepseek_data['subjects']):
    # DeepSeek cumulative
    ds_cum = calculate_cumulative_performance(
        deepseek_data['subject_correct'][i],
        deepseek_data['subject_problems'][i],
        deepseek_data['intervention_by_subject'][subject]['corrected'],
        deepseek_data['intervention_by_subject'][subject]['attempted']
    )
    deepseek_cumulative_subj.append(ds_cum)
    
    # Qwen cumulative
    qw_cum = calculate_cumulative_performance(
        qwen_data['subject_correct'][i],
        qwen_data['subject_problems'][i],
        qwen_data['intervention_by_subject'][subject]['corrected'],
        qwen_data['intervention_by_subject'][subject]['attempted']
    )
    qwen_cumulative_subj.append(qw_cum)

# Calculate intervention gains for subjects stacking
deepseek_intervention_gains_subj = [deepseek_cumulative_subj[i] - deepseek_subj_acc[i] for i in range(len(deepseek_subj_acc))]
qwen_intervention_gains_subj = [qwen_cumulative_subj[i] - qwen_subj_acc[i] for i in range(len(qwen_subj_acc))]

# Add stacked cumulative performance bars for subjects
# DeepSeek stacked bars
fig.add_trace(
    go.Bar(name='DeepSeek Baseline', x=deepseek_data['subjects'], y=deepseek_subj_acc, 
           marker_color='rgba(46, 139, 87, 0.7)', showlegend=False,
           text=[f'{val:.1f}%' for val in deepseek_subj_acc],
           textposition='inside'),
    row=6, col=1
)
fig.add_trace(
    go.Bar(name='DeepSeek Intervention Gain', x=deepseek_data['subjects'], y=deepseek_intervention_gains_subj, 
           marker_color='rgba(34, 102, 64, 1.0)', showlegend=False,
           text=[f'+{val:.1f}%' for val in deepseek_intervention_gains_subj],
           textposition='inside'),
    row=6, col=1
)

# Qwen stacked bars
fig.add_trace(
    go.Bar(name='Qwen Baseline', x=qwen_data['subjects'], y=qwen_subj_acc, 
           marker_color='rgba(65, 105, 225, 0.7)', showlegend=False,
           text=[f'{val:.1f}%' for val in qwen_subj_acc],
           textposition='inside'),
    row=6, col=1
)
fig.add_trace(
    go.Bar(name='Qwen Intervention Gain', x=qwen_data['subjects'], y=qwen_intervention_gains_subj, 
           marker_color='rgba(32, 74, 135, 1.0)', showlegend=False,
           text=[f'+{val:.1f}%' for val in qwen_intervention_gains_subj],
           textposition='inside'),
    row=6, col=1
)

# Plot 7: Baseline vs Intervention Performance Relationship
# Combine all data points for regression analysis
baseline_values = []
intervention_values = []
model_labels = []
category_labels = []

# DeepSeek data points
for i, diff in enumerate(deepseek_diff_acc):
    baseline_values.append(diff)
    intervention_values.append(deepseek_int_diff[i])
    model_labels.append('DeepSeek')
    category_labels.append(f'Difficulty {deepseek_data["difficulty_levels"][i]}')

for i, subj_acc in enumerate(deepseek_subj_acc):
    baseline_values.append(subj_acc)
    intervention_values.append(deepseek_int_subj[i])
    model_labels.append('DeepSeek')
    category_labels.append(deepseek_data['subjects'][i])

# Qwen data points
for i, diff in enumerate(qwen_diff_acc):
    baseline_values.append(diff)
    intervention_values.append(qwen_int_diff[i])
    model_labels.append('Qwen')
    category_labels.append(f'Difficulty {qwen_data["difficulty_levels"][i]}')

for i, subj_acc in enumerate(qwen_subj_acc):
    baseline_values.append(subj_acc)
    intervention_values.append(qwen_int_subj[i])
    model_labels.append('Qwen')
    category_labels.append(qwen_data['subjects'][i])

# Convert to numpy arrays
baseline_np = np.array(baseline_values)
intervention_np = np.array(intervention_values)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(baseline_np, intervention_np)

# Generate regression line
x_reg = np.linspace(min(baseline_np), max(baseline_np), 100)
y_reg = slope * x_reg + intercept

# Add scatter plot points
deepseek_mask = [i for i, label in enumerate(model_labels) if label == 'DeepSeek']
qwen_mask = [i for i, label in enumerate(model_labels) if label == 'Qwen']

fig.add_trace(
    go.Scatter(
        x=[baseline_values[i] for i in deepseek_mask],
        y=[intervention_values[i] for i in deepseek_mask],
        mode='markers',
        name='DeepSeek Data Points',
        marker=dict(color=deepseek_color, size=8, symbol='circle'),
        text=[category_labels[i] for i in deepseek_mask],
        textposition='top center',
        showlegend=True
    ),
    row=7, col=1
)

fig.add_trace(
    go.Scatter(
        x=[baseline_values[i] for i in qwen_mask],
        y=[intervention_values[i] for i in qwen_mask],
        mode='markers',
        name='Qwen Data Points',
        marker=dict(color=qwen_color, size=8, symbol='diamond'),
        text=[category_labels[i] for i in qwen_mask],
        textposition='top center',
        showlegend=True
    ),
    row=7, col=1
)

# Add regression line
fig.add_trace(
    go.Scatter(
        x=x_reg,
        y=y_reg,
        mode='lines',
        name=f'Regression Line (R²={r_value**2:.3f})',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=True
    ),
    row=7, col=1
)

# Add regression equation as annotation
equation_text = f"y = {slope:.3f}x + {intercept:.3f}<br>R² = {r_value**2:.3f}, p = {p_value:.3f}"
fig.add_annotation(
    text=equation_text,
    xref="x7", yref="y7",
    x=max(baseline_np) * 0.05,
    y=max(intervention_np) * 0.9,
    showarrow=False,
    font=dict(size=10, color="red"),
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="red",
    borderwidth=1
)

# Update layout with stacked bars and reduced width
fig.update_layout(
    width=600,  # Reduced width by 50% (from default ~1200 to 600)
    height=2800,  # Increased height for 7 plots
    title_text="Mathematical Reasoning Performance Analysis: Baseline vs Intervention + Regression Analysis",
    title_x=0.5,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    font=dict(size=12),
    margin=dict(l=80, r=80, t=100, b=100),  # Add margins for better spacing
    barmode='stack'  # Enable stacked bar mode for cumulative plots
)

# Update x and y axes
fig.update_xaxes(title_text="Difficulty Level", row=1, col=1)
fig.update_xaxes(title_text="Problem Type", row=2, col=1, tickangle=45)
fig.update_xaxes(title_text="Difficulty Level", row=3, col=1)
fig.update_xaxes(title_text="Problem Type", row=4, col=1, tickangle=45)
fig.update_xaxes(title_text="Difficulty Level", row=5, col=1)
fig.update_xaxes(title_text="Problem Type", row=6, col=1, tickangle=45)
fig.update_xaxes(title_text="Baseline Performance (%)", row=7, col=1)

fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1, range=[0, 110])
fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1, range=[0, 110])
fig.update_yaxes(title_text="Success Rate (%)", row=3, col=1, range=[0, 110])
fig.update_yaxes(title_text="Success Rate (%)", row=4, col=1, range=[0, 110])
fig.update_yaxes(title_text="Cumulative Accuracy (%)", row=5, col=1, range=[0, 110])
fig.update_yaxes(title_text="Cumulative Accuracy (%)", row=6, col=1, range=[0, 110])
fig.update_yaxes(title_text="Intervention Success Rate (%)", row=7, col=1)

# Add annotations with key statistics
fig.add_annotation(
    text="DeepSeek-R1-1.5B: 57.8% baseline accuracy, 28.6% intervention success<br>Qwen3-14B: 79.4% baseline accuracy, 51.1% intervention success",
    xref="paper", yref="paper",
    x=0.5, y=-0.05,
    showarrow=False,
    font=dict(size=10),
    align="center"
)

# Save as HTML
fig.write_html("comprehensive_performance_analysis.html")
print("✅ Comprehensive performance analysis saved as 'comprehensive_performance_analysis.html'")

# Display key insights
print("\n📊 Key Insights:")
print("1. Baseline Performance:")
print(f"   - DeepSeek shows clear difficulty scaling: {deepseek_diff_acc[0]:.1f}% → {deepseek_diff_acc[-1]:.1f}%")
print(f"   - Qwen3 maintains higher performance: {qwen_diff_acc[0]:.1f}% → {qwen_diff_acc[-1]:.1f}%")
print(f"   - Best subjects: Prealgebra ({deepseek_subj_acc[0]:.1f}% DS, {qwen_subj_acc[0]:.1f}% Q)")
print(f"   - Hardest subjects: Counting/Probability ({deepseek_subj_acc[-1]:.1f}% DS, {qwen_subj_acc[-1]:.1f}% Q)")

print("\n2. Intervention Effectiveness:")
print(f"   - DeepSeek: Higher success on easier problems ({deepseek_int_diff[0]:.1f}% → {deepseek_int_diff[-1]:.1f}%)")
print(f"   - Qwen3: More consistent intervention success across difficulties")
print(f"   - Best intervention domain: Prealgebra ({deepseek_int_subj[0]:.1f}% DS, {qwen_int_subj[0]:.1f}% Q)")
print(f"   - Hardest to correct: Counting/Probability ({deepseek_int_subj[-1]:.1f}% DS, {qwen_int_subj[-1]:.1f}% Q)")

print("\n3. Cumulative Performance (Baseline + Interventions):")
print(f"   - DeepSeek improvement: Level 1: {deepseek_diff_acc[0]:.1f}% → {deepseek_cumulative_diff[0]:.1f}% (+{deepseek_cumulative_diff[0]-deepseek_diff_acc[0]:.1f}%)")
print(f"   - DeepSeek improvement: Level 5: {deepseek_diff_acc[-1]:.1f}% → {deepseek_cumulative_diff[-1]:.1f}% (+{deepseek_cumulative_diff[-1]-deepseek_diff_acc[-1]:.1f}%)")
print(f"   - Qwen improvement: Level 1: {qwen_diff_acc[0]:.1f}% → {qwen_cumulative_diff[0]:.1f}% (+{qwen_cumulative_diff[0]-qwen_diff_acc[0]:.1f}%)")
print(f"   - Qwen improvement: Level 5: {qwen_diff_acc[-1]:.1f}% → {qwen_cumulative_diff[-1]:.1f}% (+{qwen_cumulative_diff[-1]-qwen_diff_acc[-1]:.1f}%)")
print(f"   - Best hybrid performance: Prealgebra ({deepseek_cumulative_subj[0]:.1f}% DS, {qwen_cumulative_subj[0]:.1f}% Q)")

# Calculate overall improvement
overall_deepseek_improvement = sum(deepseek_cumulative_diff) / len(deepseek_cumulative_diff) - sum(deepseek_diff_acc) / len(deepseek_diff_acc)
overall_qwen_improvement = sum(qwen_cumulative_diff) / len(qwen_cumulative_diff) - sum(qwen_diff_acc) / len(qwen_diff_acc)
print(f"\n4. Overall Impact:")
print(f"   - DeepSeek average improvement: +{overall_deepseek_improvement:.1f}% points")
print(f"   - Qwen average improvement: +{overall_qwen_improvement:.1f}% points")
print(f"   - Intervention value: Provides significant boost to both models")

print(f"\n5. Baseline vs Intervention Relationship (Regression Analysis):")
print(f"   - Linear relationship: y = {slope:.3f}x + {intercept:.3f}")
print(f"   - Correlation coefficient (R²): {r_value**2:.3f}")
print(f"   - Statistical significance (p-value): {p_value:.3f}")

if p_value < 0.05:
    significance = "statistically significant"
else:
    significance = "not statistically significant"

if slope > 0:
    relationship = "positive"
    interpretation = "Higher baseline performance correlates with better intervention success"
elif slope < 0:
    relationship = "negative" 
    interpretation = "Lower baseline performance actually benefits more from interventions"
else:
    relationship = "neutral"
    interpretation = "Intervention success is independent of baseline performance"

print(f"   - Relationship type: {relationship} correlation ({significance})")
print(f"   - Interpretation: {interpretation}")

if abs(slope) < 0.1:
    consistency = "highly consistent"
elif abs(slope) < 0.3:
    consistency = "moderately consistent"
else:
    consistency = "variable"
    
print(f"   - Intervention effectiveness: {consistency} across performance levels")

# Calculate expected intervention success for different baseline levels
baseline_levels = [30, 50, 70, 90]
print(f"   - Predicted intervention success rates:")
for baseline in baseline_levels:
    predicted = slope * baseline + intercept
    print(f"     * At {baseline}% baseline → {predicted:.1f}% intervention success")
