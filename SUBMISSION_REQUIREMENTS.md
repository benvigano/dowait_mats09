# MATS9 Application Submission Requirements - Reference Guide

## Application Format Overview
- **Format**: Google Doc with public access ("anyone with link can view")
- **Structure**: Executive Summary (1-3 pages) + Detailed Analysis
- **Focus**: Mini-research project demonstrating exploration → understanding → distillation

## Executive Summary Requirements (CRITICAL - READ BY ALL REVIEWERS)
- **Length**: 1-3 pages maximum, ideally ~1 page
- **Word Count**: Maximum 600 words
- **Must Include**: Graphs and visual evidence
- **Format**: Bullet points work well
- **Purpose**: Convey key findings to someone with mech interp experience

### Executive Summary Structure
1. **Problem Statement**: What problem am I solving? Why is it interesting?
2. **High-level Takeaways**: Most interesting findings from the project
3. **Key Experiments**: One paragraph + graph per experiment
   - What the experiment was
   - What was found
   - Why it supports the takeaways

## Research Process Framework
### 1. Exploration Phase
- Build intuition and gain information
- Get hands dirty with data/prompts
- Maximize information gain per unit time
- Ask: "Have I learned anything in the last 30 minutes?"

### 2. Understanding Phase
- Test hypotheses with careful experiments
- Keep running doc of hypotheses
- Track type of claims being made:
  - Existence proofs (cherry-picking OK)
  - Method comparisons (need baselines)
- Avoid simple alternative explanations

### 3. Distillation Phase
- Clear, honest write-up
- Avoid cherry-picked qualitative examples
- Compare to baselines when applicable
- Structure around 1-2 key insights

## Evaluation Criteria (In Order of Importance)

### 1. Clarity (Top 20% Requirement)
- Clear claims with supporting evidence
- Sufficient detail to follow methodology
- Good structure, bullet points, graphs
- Define terms, label graphs clearly

### 2. Good Taste
- Interesting, tractable questions
- Compelling results that teach something new
- Originality is a big plus
- Alignment with Neel's research interests

### 3. Truth-seeking & Skepticism
- Question results, look for alternatives
- Sanity checks and self-awareness
- Negative/inconclusive results > poorly supported positive
- Acknowledge limitations and speculation

### 4. Technical Depth & Practicality
- Good handle on relevant tools
- Well-motivated design decisions
- Knowledge areas: mech interp papers, transformers, coding, linear algebra

### 5. Simplicity
- Try simple methods first
- Each complexity should have a reason
- Pragmatic and focused approach

### 6. Prioritization
- Go deep on 1-2 insights vs. superficial on many
- Avoid rabbit holes and random anomalies
- Balance depth vs. breadth appropriately

### 7. Productivity
- Fast feedback loops
- Efficient use of time
- Show thought process and decision-making

### 8. Enthusiasm & Curiosity
- Follow curiosity productively
- Make it fun to read

## Writing Guidelines

### Core Principles
- **Focus on Narrative**: Structure around key insights, not chronological experiments
- **Quality over Quantity**: One well-explained finding > ten superficial experiments
- **Show Your Work**: Explain why experiments were run, what hypotheses were tested
- **Zero Context Assumption**: Explain everything from ground up
- **Executive Summary is Critical**: Must stand alone and convey key points

### Red Flags to Avoid
- Relying only on cherry-picked qualitative examples
- Chronological experiment dump without narrative
- Overconfidence in shaky results
- Missing baselines when needed
- Unclear methodology or metrics

## Our Project Context

### Research Questions
1. **RQ1**: How does activation patching reveal which model components are most involved in mathematical reasoning processes?
2. **RQ2**: Can we identify specific layers/components that are causally important for correct mathematical reasoning?
3. **RQ3**: What does the causal tracing reveal about the model's internal computation during problem-solving?

### Our Approach
- **Model**: DeepSeek-R1-Distill-Qwen-1.5B (1.5B parameters)
- **Dataset**: MATH dataset problems with diverse difficulty levels
- **Method**: Activation patching (causal tracing) across different model components
- **Components Tested**: resid_pre, attn_out, mlp_out
- **Analysis**: Heatmaps showing logit difference recovery across layers and positions

### Key Experiments Completed
1. **Baseline Experiment**: Model performance on MATH problems (275/476 correct = 57.8%)
2. **Intervention Testing**: "Wait" phrase insertion at error points (48/168 corrections = 28.6%)
3. **Activation Patching**: Causal tracing across model components to identify important layers

### Potential Findings to Highlight
- Which layers are most causally important for mathematical reasoning
- How different components (attention vs. MLP) contribute to correct reasoning
- Patterns in where interventions are most effective
- Connection between intervention success and activation patching results

## Reference Papers to Consider Citing
1. **Inspiration Paper**: Previous MATS application on backtracking in CoT reasoning
2. **Recent Relevant Work**: https://arxiv.org/pdf/2506.08343
3. **Lanham Intervention Test**: As we're doing reverse Lanham insertion testing

## Technical Implementation Notes
- Used TransformerLens for activation patching
- Generated heatmaps with plotly
- Saved results as JSON and HTML files
- Implemented modular code structure across multiple Python files

## Success Metrics for Our Application
- **Clarity**: Can reviewer understand our causal tracing methodology and results?
- **Novelty**: Is activation patching on mathematical reasoning interesting/original?
- **Evidence**: Do our heatmaps and quantitative results support our claims?
- **Depth**: Did we go deep enough on the causal analysis vs. staying superficial?
- **Practical Value**: Does this contribute useful insights about model internals?

## Next Steps for Submission
1. Analyze activation patching results and identify key patterns
2. Create compelling visualizations of most important findings
3. Write executive summary focusing on 1-2 key insights
4. Structure full document around narrative of causal importance
5. Ensure all claims are well-supported by quantitative evidence
