import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class EnhancedMetricsAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize the analyzer for all metrics including enhanced ones"""
        self.df = pd.read_csv(csv_file_path)
        self.setup_plotting()

    def setup_plotting(self):
        """Setup plotting parameters"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)

    def generate_summary_report(self):
        """Generate comprehensive analysis of all metrics"""
        print("=" * 80)
        print("ENHANCED METRICS ANALYSIS")
        print("🎯 Four Core + Two Enhanced Metrics Analysis")
        print("=" * 80)

        # Basic statistics
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   Total evaluations: {len(self.df)}")

        # Success rate based on non-zero overall scores
        successful_rows = self.df[
            (self.df['gnn_overall_score'] > 0) |
            (self.df['mcts_overall_score'] > 0) |
            (self.df['rmodel_overall_score'] > 0)
            ]
        success_rate = len(successful_rows) / len(self.df) * 100
        print(f"   Success rate: {success_rate:.1f}% ({len(successful_rows)} successful)")

        # Core analyses
        self.analyze_all_metrics_performance()
        self.analyze_enhanced_metrics_impact()
        self.analyze_model_comparison()
        self.analyze_metric_correlations()
        self.analyze_domain_patterns()
        self.analyze_graph_performance_relationship()

    def analyze_all_metrics_performance(self):
        """Analyze performance across all six metrics"""
        print(f"\n🎯 ALL METRICS PERFORMANCE ANALYSIS")
        print("-" * 60)

        # All six metrics
        core_metrics = ['actionability', 'coherence', 'domain_specificity', 'technological_specificity']
        enhanced_metrics = ['understandability_enhanced', 'user_focus_enhanced']
        all_metrics = core_metrics + enhanced_metrics

        models = ['gnn', 'mcts', 'rmodel']

        # Create performance matrix
        performance_matrix = {}

        print(f"\n🔸 CORE FOUR METRICS:")
        for metric in core_metrics:
            performance_matrix[metric] = {}
            print(f"\n   📈 {metric.replace('_', ' ').title()}:")

            metric_scores = []
            for model in models:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]

                    if len(non_zero_values) > 0:
                        mean_score = non_zero_values.mean()
                        std_score = non_zero_values.std()
                        performance_matrix[metric][model] = mean_score
                        metric_scores.append((model.upper(), mean_score, std_score, len(non_zero_values)))
                    else:
                        performance_matrix[metric][model] = 0

            # Rank models for this metric
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            for i, (model, mean, std, count) in enumerate(metric_scores):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"     {rank} {model}: {mean:.3f} ± {std:.3f} ({count} samples)")

        print(f"\n🔸 ENHANCED METRICS:")
        for metric in enhanced_metrics:
            performance_matrix[metric] = {}
            print(f"\n   ✨ {metric.replace('_', ' ').title()}:")

            metric_scores = []
            for model in models:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]

                    if len(non_zero_values) > 0:
                        mean_score = non_zero_values.mean()
                        std_score = non_zero_values.std()
                        performance_matrix[metric][model] = mean_score
                        metric_scores.append((model.upper(), mean_score, std_score, len(non_zero_values)))
                    else:
                        performance_matrix[metric][model] = 0
                else:
                    print(f"     ⚠️  Column {col_name} not found in data")

            # Rank models for this metric
            if metric_scores:
                metric_scores.sort(key=lambda x: x[1], reverse=True)
                for i, (model, mean, std, count) in enumerate(metric_scores):
                    rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    print(f"     {rank} {model}: {mean:.3f} ± {std:.3f} ({count} samples)")
            else:
                print(f"     ❌ No data available for this metric")

        # Overall metric ranking (across all models)
        print(f"\n📊 OVERALL METRIC STRENGTH RANKING:")
        overall_metric_scores = {}
        for metric in all_metrics:
            if metric in performance_matrix:
                scores = [score for score in performance_matrix[metric].values() if score > 0]
                if scores:
                    overall_metric_scores[metric] = np.mean(scores)

        ranked_metrics = sorted(overall_metric_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (metric, avg_score) in enumerate(ranked_metrics):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i + 1}️⃣"
            metric_type = "Enhanced" if "enhanced" in metric else "Core"
            print(f"     {rank} {metric.replace('_', ' ').title()} ({metric_type}): {avg_score:.3f}")

    def analyze_enhanced_metrics_impact(self):
        """Analyze the impact and insights from enhanced metrics"""
        print(f"\n✨ ENHANCED METRICS IMPACT ANALYSIS")
        print("-" * 60)

        models = ['gnn', 'mcts', 'rmodel']

        # Check data availability first
        enhanced_columns = []
        for model in models:
            for metric in ['understandability_enhanced', 'user_focus_enhanced']:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    enhanced_columns.append(col_name)

        if not enhanced_columns:
            print("   ❌ Enhanced metrics data not available in the dataset")
            print("   💡 Make sure your experiment script includes the enhanced metrics")
            return

        print(f"   ✅ Found {len(enhanced_columns)} enhanced metric columns")

        # 1. Understandability Analysis
        print(f"\n🔸 Understandability Analysis:")
        understandability_scores = []
        for model in models:
            col_name = f"{model}_understandability_enhanced"
            if col_name in self.df.columns:
                values = self.df[col_name].dropna()
                non_zero_values = values[values > 0]
                if len(non_zero_values) > 0:
                    avg_score = non_zero_values.mean()
                    understandability_scores.append((model.upper(), avg_score, len(non_zero_values)))

        if understandability_scores:
            understandability_scores.sort(key=lambda x: x[1], reverse=True)
            for i, (model, score, count) in enumerate(understandability_scores):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"     {rank} {model}: {score:.3f} ({count} samples)")

                # Interpretation
                if score > 7.0:
                    interpretation = "Excellent - Very easy to understand"
                elif score > 5.0:
                    interpretation = "Good - Moderately easy to understand"
                elif score > 3.0:
                    interpretation = "Fair - Some clarity issues"
                else:
                    interpretation = "Poor - Difficult to understand"
                print(f"         → {interpretation}")

        # 2. User Focus Analysis
        print(f"\n🔸 User Focus (Prompt Alignment) Analysis:")
        user_focus_scores = []
        for model in models:
            col_name = f"{model}_user_focus_enhanced"
            if col_name in self.df.columns:
                values = self.df[col_name].dropna()
                non_zero_values = values[values > 0]
                if len(non_zero_values) > 0:
                    avg_score = non_zero_values.mean()
                    user_focus_scores.append((model.upper(), avg_score, len(non_zero_values)))

        if user_focus_scores:
            user_focus_scores.sort(key=lambda x: x[1], reverse=True)
            for i, (model, score, count) in enumerate(user_focus_scores):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"     {rank} {model}: {score:.3f} ({count} samples)")

                # Interpretation
                if score > 7.0:
                    interpretation = "Excellent - Very well aligned with user intent"
                elif score > 5.0:
                    interpretation = "Good - Moderately aligned with user intent"
                elif score > 3.0:
                    interpretation = "Fair - Some alignment issues"
                else:
                    interpretation = "Poor - Poorly aligned with user intent"
                print(f"         → {interpretation}")

        # 3. Enhanced vs Core Metrics Correlation
        print(f"\n🔸 Enhanced vs Core Metrics Relationships:")
        self.analyze_enhanced_core_correlations()

    def analyze_enhanced_core_correlations(self):
        """Analyze correlations between enhanced and core metrics"""
        core_metrics = ['actionability', 'coherence', 'domain_specificity', 'technological_specificity']
        enhanced_metrics = ['understandability_enhanced', 'user_focus_enhanced']
        models = ['gnn', 'mcts', 'rmodel']

        significant_correlations = []

        for model in models:
            for core_metric in core_metrics:
                for enhanced_metric in enhanced_metrics:
                    core_col = f"{model}_{core_metric}"
                    enhanced_col = f"{model}_{enhanced_metric}"

                    if core_col in self.df.columns and enhanced_col in self.df.columns:
                        # Get clean data
                        clean_data = self.df[[core_col, enhanced_col]].dropna()
                        clean_data = clean_data[(clean_data[core_col] > 0) & (clean_data[enhanced_col] > 0)]

                        if len(clean_data) > 10:
                            try:
                                corr, p_val = stats.pearsonr(clean_data[core_col], clean_data[enhanced_col])

                                if abs(corr) > 0.3 and p_val < 0.05:
                                    significant_correlations.append((
                                        model.upper(), core_metric, enhanced_metric, corr, p_val, len(clean_data)
                                    ))
                            except:
                                continue

        if significant_correlations:
            # Sort by correlation strength
            significant_correlations.sort(key=lambda x: abs(x[3]), reverse=True)

            for model, core_metric, enhanced_metric, corr, p_val, n in significant_correlations[:10]:
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "weak"
                print(
                    f"     {model} {core_metric} ↔ {enhanced_metric}: {strength} {direction} (r={corr:.3f}, p={p_val:.3f}, n={n})")
        else:
            print("     No significant correlations found between enhanced and core metrics")

    def analyze_model_comparison(self):
        """Enhanced model comparison including all metrics"""
        print(f"\n🤖 COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
        print("-" * 60)

        models = ['gnn', 'mcts', 'rmodel']

        # Overall scores
        print(f"\n🏆 Overall Performance (Core Four Metrics):")
        model_performances = []

        for model in models:
            col_name = f"{model}_overall_score"
            if col_name in self.df.columns:
                values = self.df[col_name].dropna()
                non_zero_values = values[values > 0]

                if len(non_zero_values) > 0:
                    mean_score = non_zero_values.mean()
                    std_score = non_zero_values.std()
                    max_score = non_zero_values.max()
                    model_performances.append((model.upper(), mean_score, std_score, max_score, len(non_zero_values)))

        # Rank models
        model_performances.sort(key=lambda x: x[1], reverse=True)
        for i, (model, mean, std, max_val, count) in enumerate(model_performances):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"     {rank} {model}: {mean:.3f} ± {std:.3f} (max: {max_val:.3f}, n={count})")

        # Enhanced Metrics Summary
        print(f"\n✨ Enhanced Metrics Summary:")
        for metric in ['understandability_enhanced', 'user_focus_enhanced']:
            print(f"\n   {metric.replace('_', ' ').title()}:")
            metric_performances = []

            for model in models:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]

                    if len(non_zero_values) > 0:
                        mean_score = non_zero_values.mean()
                        std_score = non_zero_values.std()
                        metric_performances.append((model.upper(), mean_score, std_score, len(non_zero_values)))

            if metric_performances:
                metric_performances.sort(key=lambda x: x[1], reverse=True)
                for i, (model, mean, std, count) in enumerate(metric_performances):
                    rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    print(f"     {rank} {model}: {mean:.3f} ± {std:.3f} ({count} samples)")

        # Best model wins distribution
        if 'best_performing_model' in self.df.columns:
            best_models = self.df['best_performing_model'].value_counts()
            print(f"\n🎯 Model Win Distribution (Core Metrics):")
            for model, wins in best_models.items():
                percentage = (wins / len(self.df)) * 100
                print(f"     {model}: {wins} wins ({percentage:.1f}%)")

        # Performance gap analysis
        if 'performance_gap' in self.df.columns:
            gaps = self.df['performance_gap'].dropna()
            non_zero_gaps = gaps[gaps > 0]
            if len(non_zero_gaps) > 0:
                print(f"\n📊 Performance Gap Analysis:")
                print(f"     Average gap: {non_zero_gaps.mean():.3f}")
                print(f"     Median gap: {non_zero_gaps.median():.3f}")
                print(f"     Max gap: {non_zero_gaps.max():.3f}")

                # Categorize competition level
                if non_zero_gaps.mean() < 0.1:
                    competition = "Very close competition"
                elif non_zero_gaps.mean() < 0.2:
                    competition = "Close competition"
                elif non_zero_gaps.mean() < 0.3:
                    competition = "Moderate differences"
                else:
                    competition = "Clear performance differences"
                print(f"     Competition level: {competition}")

    def analyze_metric_correlations(self):
        """Analyze correlations between all metrics"""
        print(f"\n📈 COMPREHENSIVE METRIC CORRELATION ANALYSIS")
        print("-" * 60)

        all_metrics = ['actionability', 'coherence', 'domain_specificity', 'technological_specificity',
                       'understandability_enhanced', 'user_focus_enhanced']

        # Analyze correlations within each model
        for model in ['gnn', 'mcts', 'rmodel']:
            print(f"\n🔸 {model.upper()} Model - All Metric Correlations:")

            model_columns = []
            for metric in all_metrics:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    model_columns.append(col_name)

            if len(model_columns) >= 2:
                # Filter to rows with non-zero values
                model_data = self.df[model_columns].dropna()
                model_data = model_data[(model_data > 0).all(axis=1)]

                if len(model_data) > 10:
                    correlations = []

                    # Check all metric pairs
                    for i, col1 in enumerate(model_columns):
                        for j, col2 in enumerate(model_columns):
                            if i < j:  # Avoid duplicates
                                try:
                                    corr, p_val = stats.pearsonr(model_data[col1], model_data[col2])

                                    if abs(corr) > 0.3 and p_val < 0.05:
                                        metric1 = col1.replace(f"{model}_", "")
                                        metric2 = col2.replace(f"{model}_", "")
                                        correlations.append((metric1, metric2, corr, p_val))
                                except:
                                    continue

                    if correlations:
                        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                        for metric1, metric2, corr, p_val in correlations[:8]:  # Top 8 correlations
                            direction = "positive" if corr > 0 else "negative"
                            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "weak"

                            # Mark enhanced metrics
                            m1_type = " (Enhanced)" if "enhanced" in metric1 else ""
                            m2_type = " (Enhanced)" if "enhanced" in metric2 else ""

                            print(
                                f"       {metric1.replace('_', ' ').title()}{m1_type} ↔ {metric2.replace('_', ' ').title()}{m2_type}")
                            print(f"         → {strength} {direction} (r={corr:.3f}, p={p_val:.3f})")
                    else:
                        print(f"       No significant correlations found")
                else:
                    print(f"       Insufficient data ({len(model_data)} samples)")
            else:
                print(f"       Insufficient metrics available ({len(model_columns)} columns)")

    def analyze_domain_patterns(self):
        """Analyze performance patterns by domain including enhanced metrics"""
        print(f"\n🎯 DOMAIN-SPECIFIC PERFORMANCE PATTERNS (All Metrics)")
        print("-" * 60)

        if 'dominant_domain' not in self.df.columns:
            print("   Domain information not available")
            return

        domains = self.df['dominant_domain'].value_counts()
        print(f"\n📊 Domain Distribution:")
        for domain, count in domains.items():
            percentage = (count / len(self.df)) * 100
            print(f"     {domain}: {count} ({percentage:.1f}%)")

        # Performance by domain for all metrics
        print(f"\n🏆 Performance by Domain (All Metrics):")
        all_metrics = ['actionability', 'coherence', 'domain_specificity', 'technological_specificity',
                       'understandability_enhanced', 'user_focus_enhanced']

        for domain in domains.index[:5]:  # Top 5 domains
            domain_data = self.df[self.df['dominant_domain'] == domain]

            if len(domain_data) > 3:  # Need sufficient samples
                print(f"\n   🏷️  {domain.title()} Domain:")

                for metric in all_metrics:
                    # Aggregate across all models for this metric
                    all_values = []
                    for model in ['gnn', 'mcts', 'rmodel']:
                        col_name = f"{model}_{metric}"
                        if col_name in domain_data.columns:
                            values = domain_data[col_name].dropna()
                            non_zero_values = values[values > 0]
                            all_values.extend(non_zero_values.tolist())

                    if all_values:
                        avg_score = np.mean(all_values)
                        metric_type = " (Enhanced)" if "enhanced" in metric else ""
                        print(f"     {metric.replace('_', ' ').title()}{metric_type}: {avg_score:.3f}")

    def analyze_graph_performance_relationship(self):
        """Analyze relationship between graph structure and performance"""
        print(f"\n📊 GRAPH-PERFORMANCE RELATIONSHIP ANALYSIS")
        print("-" * 60)

        # Graph efficiency analysis
        if 'gnn_graph_performance_ratio' in self.df.columns and 'mcts_graph_performance_ratio' in self.df.columns:
            print(f"\n🔸 Graph Efficiency (Lower ratio = better performance per node):")

            for model in ['gnn', 'mcts']:
                col_name = f"{model}_graph_performance_ratio"
                ratios = self.df[col_name].dropna()
                # Filter extreme values
                ratios = ratios[(ratios > 0) & (ratios < 100)]

                if len(ratios) > 0:
                    print(f"     {model.upper()}: avg={ratios.mean():.2f}, median={ratios.median():.2f}")

        # Graph size vs performance correlation (including enhanced metrics)
        correlations_to_check = [
            ('gnn_node_count', 'gnn_overall_score'),
            ('mcts_node_count', 'mcts_overall_score'),
            ('gnn_graph_density', 'gnn_overall_score'),
            ('mcts_graph_density', 'mcts_overall_score')
        ]

        # Add enhanced metrics correlations
        for model in ['gnn', 'mcts']:
            for metric in ['understandability_enhanced', 'user_focus_enhanced']:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    correlations_to_check.append((f"{model}_node_count", col_name))
                    correlations_to_check.append((f"{model}_graph_density", col_name))

        print(f"\n🔸 Graph Structure vs Performance Correlations:")
        significant_correlations = []

        for var1, var2 in correlations_to_check:
            if var1 in self.df.columns and var2 in self.df.columns:
                # Get clean data
                clean_data = self.df[[var1, var2]].dropna()
                clean_data = clean_data[(clean_data[var1] > 0) & (clean_data[var2] > 0)]

                if len(clean_data) > 10:
                    try:
                        corr, p_val = stats.pearsonr(clean_data[var1], clean_data[var2])

                        if abs(corr) > 0.2 and p_val < 0.1:  # Slightly lower threshold
                            significant_correlations.append((var1, var2, corr, p_val, len(clean_data)))
                    except:
                        continue

        if significant_correlations:
            significant_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            for var1, var2, corr, p_val, n in significant_correlations:
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"

                # Mark enhanced metrics
                var2_type = " (Enhanced)" if "enhanced" in var2 else ""

                print(
                    f"     {var1} ↔ {var2.replace('_', ' ')}{var2_type}: {strength} {direction} (r={corr:.3f}, p={p_val:.3f}, n={n})")
        else:
            print(f"     No significant correlations found")

    def create_enhanced_visualizations(self, save_plots=True):
        """Create visualizations including enhanced metrics"""
        print(f"\n📊 GENERATING ENHANCED VISUALIZATIONS...")

        fig_count = 1

        # 1. Six Metrics Radar Chart Comparison
        all_metrics = ['actionability', 'coherence', 'domain_specificity', 'technological_specificity',
                       'understandability_enhanced', 'user_focus_enhanced']
        models = ['gnn', 'mcts', 'rmodel']

        # Collect data for radar chart
        radar_data = {}
        for model in models:
            radar_data[model] = []
            for metric in all_metrics:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]
                    avg_score = non_zero_values.mean() if len(non_zero_values) > 0 else 0
                    # Normalize enhanced metrics to 0-1 scale (they're on 0-10 scale)
                    if "enhanced" in metric:
                        avg_score = avg_score / 10.0
                    radar_data[model].append(avg_score)
                else:
                    radar_data[model].append(0)

        if any(sum(scores) > 0 for scores in radar_data.values()):
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

            colors = ['blue', 'green', 'red']
            for i, (model, scores) in enumerate(radar_data.items()):
                if sum(scores) > 0:
                    scores += scores[:1]  # Complete the circle
                    ax.plot(angles, scores, 'o-', linewidth=2, label=model.upper(), color=colors[i])
                    ax.fill(angles, scores, alpha=0.25, color=colors[i])

            # Enhanced labels
            labels = []
            for metric in all_metrics:
                label = metric.replace('_', '\n').title()
                if "enhanced" in metric:
                    label += "\n(Enhanced)"
                labels.append(label)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_title('Six Metrics Comparison by Model\n(Core + Enhanced)', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)

            if save_plots:
                plt.savefig('./evals_results/six_metrics_radar_chart.png', dpi=300, bbox_inches='tight')
            plt.show()
            fig_count += 1

        # 2. Enhanced Metrics Comparison
        enhanced_data = []
        for model in models:
            for metric in ['understandability_enhanced', 'user_focus_enhanced']:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]
                    for val in non_zero_values:
                        enhanced_data.append({
                            'Model': model.upper(),
                            'Metric': metric.replace('_', ' ').title(),
                            'Score': val
                        })

        if enhanced_data:
            plt.figure(fig_count, figsize=(12, 6))
            fig_count += 1

            enhanced_df = pd.DataFrame(enhanced_data)
            sns.boxplot(data=enhanced_df, x='Metric', y='Score', hue='Model')
            plt.title('Enhanced Metrics Performance Comparison')
            plt.ylabel('Score (0-10)')
            plt.xticks(rotation=45)
            plt.legend(title='Model')
            plt.grid(True, alpha=0.3)

            if save_plots:
                plt.savefig('./evals_results/enhanced_metrics_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 3. All Metrics Heatmap
        heatmap_data = []
        for model in models:
            model_row = []
            for metric in all_metrics:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]
                    avg_score = non_zero_values.mean() if len(non_zero_values) > 0 else 0
                    # Normalize enhanced metrics to 0-1 scale
                    if "enhanced" in metric:
                        avg_score = avg_score / 10.0
                    model_row.append(avg_score)
                else:
                    model_row.append(0)
            heatmap_data.append(model_row)

        if any(sum(row) > 0 for row in heatmap_data):
            plt.figure(fig_count, figsize=(14, 8))
            fig_count += 1

            # Enhanced labels for heatmap
            metric_labels = []
            for metric in all_metrics:
                label = metric.replace('_', ' ').title()
                if "enhanced" in metric:
                    label += " (Enhanced)"
                metric_labels.append(label)

            heatmap_df = pd.DataFrame(heatmap_data,
                                      index=[m.upper() for m in models],
                                      columns=metric_labels)

            sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', fmt='.3f',
                        cbar_kws={'label': 'Average Score (Normalized)'})
            plt.title('All Metrics Performance Heatmap by Model\n(Core + Enhanced Metrics)')
            plt.ylabel('Model')
            plt.xlabel('Metric')
            plt.xticks(rotation=45, ha='right')

            if save_plots:
                plt.savefig('./evals_results/all_metrics_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 4. Enhanced vs Core Metrics Correlation Plot
        if len(all_metrics) >= 4:
            plt.figure(fig_count, figsize=(12, 8))
            fig_count += 1

            # Create correlation matrix for all metrics
            correlation_data = {}
            for metric in all_metrics:
                all_values = []
                for model in models:
                    col_name = f"{model}_{metric}"
                    if col_name in self.df.columns:
                        values = self.df[col_name].dropna()
                        non_zero_values = values[values > 0]
                        # Normalize enhanced metrics
                        if "enhanced" in metric:
                            non_zero_values = non_zero_values / 10.0
                        all_values.extend(non_zero_values.tolist())

                if all_values:
                    correlation_data[metric.replace('_', ' ').title()] = all_values

            if len(correlation_data) >= 2:
                # Ensure all lists have the same length
                min_length = min(len(values) for values in correlation_data.values())
                if min_length > 10:
                    correlation_data = {k: v[:min_length] for k, v in correlation_data.items()}

                    corr_df = pd.DataFrame(correlation_data)
                    correlation_matrix = corr_df.corr()

                    # Create mask for enhanced metrics
                    mask = np.zeros_like(correlation_matrix, dtype=bool)
                    enhanced_indices = [i for i, col in enumerate(correlation_matrix.columns)
                                        if "Enhanced" in col or "enhanced" in col.lower()]

                    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                                fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
                    plt.title('All Metrics Correlation Matrix\n(Combined across all models)')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)

                    if save_plots:
                        plt.savefig('./evals_results/all_metrics_correlation_matrix.png', dpi=300, bbox_inches='tight')
                    plt.show()

        print(f"   Generated {fig_count - 1} visualization(s)")
        if save_plots:
            print(f"   Plots saved to ./evals_results/")

    def export_enhanced_summary_statistics(self, filename="enhanced_metrics_summary.csv"):
        """Export comprehensive summary statistics including enhanced metrics"""
        summary_stats = {}

        # All metrics for each model
        all_metrics = ['actionability', 'coherence', 'domain_specificity', 'technological_specificity',
                       'understandability_enhanced', 'user_focus_enhanced']

        key_columns = []
        for model in ['gnn', 'mcts', 'rmodel']:
            for metric in all_metrics:
                key_columns.append(f"{model}_{metric}")
            key_columns.append(f"{model}_overall_score")

        for col in key_columns:
            if col in self.df.columns:
                values = self.df[col].dropna()
                non_zero_values = values[values > 0]

                if len(non_zero_values) > 0:
                    summary_stats[col] = {
                        'count': len(non_zero_values),
                        'mean': non_zero_values.mean(),
                        'std': non_zero_values.std(),
                        'min': non_zero_values.min(),
                        'q25': non_zero_values.quantile(0.25),
                        'median': non_zero_values.median(),
                        'q75': non_zero_values.quantile(0.75),
                        'max': non_zero_values.max(),
                        'metric_type': 'Enhanced' if 'enhanced' in col else 'Core'
                    }

        if summary_stats:
            summary_df = pd.DataFrame(summary_stats).T
            summary_df.to_csv(f"./evals_results/{filename}")
            print(f"   Enhanced summary statistics exported to ./evals_results/{filename}")

    def generate_enhanced_actionable_insights(self):
        """Generate actionable insights including enhanced metrics"""
        print(f"\n💡 ENHANCED ACTIONABLE INSIGHTS")
        print("=" * 60)

        all_metrics = ['actionability', 'coherence', 'domain_specificity', 'technological_specificity',
                       'understandability_enhanced', 'user_focus_enhanced']
        models = ['gnn', 'mcts', 'rmodel']

        # Find best and worst performing areas across all metrics
        performance_matrix = {}
        for model in models:
            for metric in all_metrics:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]
                    if len(non_zero_values) > 0:
                        avg_score = non_zero_values.mean()
                        # Normalize enhanced metrics for fair comparison
                        if "enhanced" in metric:
                            avg_score = avg_score / 10.0
                        performance_matrix[f"{model}_{metric}"] = avg_score

        if performance_matrix:
            # Find strengths and weaknesses
            sorted_performance = sorted(performance_matrix.items(), key=lambda x: x[1], reverse=True)

            print(f"\n🚀 TOP STRENGTHS (All Metrics):")
            for i, (metric_model, score) in enumerate(sorted_performance[:8]):
                model, metric = metric_model.split('_', 1)
                metric_type = "(Enhanced)" if "enhanced" in metric else "(Core)"
                display_score = score * 10 if "enhanced" in metric else score
                print(
                    f"   {i + 1}. {model.upper()} {metric.replace('_', ' ').title()} {metric_type}: {display_score:.3f}")

            print(f"\n⚠️  AREAS FOR IMPROVEMENT (All Metrics):")
            for i, (metric_model, score) in enumerate(sorted_performance[-8:]):
                model, metric = metric_model.split('_', 1)
                metric_type = "(Enhanced)" if "enhanced" in metric else "(Core)"
                display_score = score * 10 if "enhanced" in metric else score
                print(
                    f"   {i + 1}. {model.upper()} {metric.replace('_', ' ').title()} {metric_type}: {display_score:.3f}")

        # Enhanced insights
        print(f"\n✨ ENHANCED METRICS INSIGHTS:")

        # Understandability insights
        print(f"\n🔸 Understandability Insights:")
        understandability_scores = []
        for model in models:
            col_name = f"{model}_understandability_enhanced"
            if col_name in self.df.columns:
                values = self.df[col_name].dropna()
                non_zero_values = values[values > 0]
                if len(non_zero_values) > 0:
                    avg_score = non_zero_values.mean()
                    understandability_scores.append((model, avg_score))

        if understandability_scores:
            understandability_scores.sort(key=lambda x: x[1], reverse=True)
            best_understand = understandability_scores[0]
            worst_understand = understandability_scores[-1]

            print(f"   🏆 Best: {best_understand[0].upper()} ({best_understand[1]:.2f}/10)")
            print(f"   🔧 Needs work: {worst_understand[0].upper()} ({worst_understand[1]:.2f}/10)")

            if best_understand[1] < 6.0:
                print(f"   💡 Overall recommendation: All models need improvement in instruction clarity")
            elif worst_understand[1] < 4.0:
                print(f"   💡 Priority: Focus on improving {worst_understand[0].upper()} model's instruction structure")

        # User Focus insights
        print(f"\n🔸 User Focus (Prompt Alignment) Insights:")
        user_focus_scores = []
        for model in models:
            col_name = f"{model}_user_focus_enhanced"
            if col_name in self.df.columns:
                values = self.df[col_name].dropna()
                non_zero_values = values[values > 0]
                if len(non_zero_values) > 0:
                    avg_score = non_zero_values.mean()
                    user_focus_scores.append((model, avg_score))

        if user_focus_scores:
            user_focus_scores.sort(key=lambda x: x[1], reverse=True)
            best_focus = user_focus_scores[0]
            worst_focus = user_focus_scores[-1]

            print(f"   🏆 Best: {best_focus[0].upper()} ({best_focus[1]:.2f}/10)")
            print(f"   🔧 Needs work: {worst_focus[0].upper()} ({worst_focus[1]:.2f}/10)")

            if best_focus[1] < 6.0:
                print(f"   💡 Overall recommendation: All models struggle with prompt alignment")
            elif worst_focus[1] < 4.0:
                print(f"   💡 Priority: {worst_focus[0].upper()} model needs better user intent preservation")

        # Model-specific comprehensive recommendations
        print(f"\n📋 COMPREHENSIVE MODEL RECOMMENDATIONS:")

        for model in models:
            model_scores = {}
            for metric in all_metrics:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]
                    if len(non_zero_values) > 0:
                        avg_score = non_zero_values.mean()
                        # Normalize for comparison
                        if "enhanced" in metric:
                            avg_score = avg_score / 10.0
                        model_scores[metric] = avg_score

            if model_scores:
                # Find best and worst metrics for this model
                best_metric = max(model_scores, key=model_scores.get)
                worst_metric = min(model_scores, key=model_scores.get)

                best_score = model_scores[best_metric]
                worst_score = model_scores[worst_metric]

                # Display scores in original scale
                if "enhanced" in best_metric:
                    best_display = best_score * 10
                    best_scale = "/10"
                else:
                    best_display = best_score
                    best_scale = ""

                if "enhanced" in worst_metric:
                    worst_display = worst_score * 10
                    worst_scale = "/10"
                else:
                    worst_display = worst_score
                    worst_scale = ""

                print(f"\n   🤖 {model.upper()} Model:")
                print(f"     ✅ Strength: {best_metric.replace('_', ' ').title()} ({best_display:.3f}{best_scale})")
                print(f"     🔧 Focus area: {worst_metric.replace('_', ' ').title()} ({worst_display:.3f}{worst_scale})")

                # Specific recommendations based on worst metric
                recommendations = {
                    'actionability': "Make instructions more specific and implementable with concrete steps",
                    'coherence': "Improve step-by-step sequencing and logical flow between instructions",
                    'domain_specificity': "Include more domain-specific terminology and context",
                    'technological_specificity': "Add specific tools, APIs, and technical implementation details",
                    'understandability_enhanced': "Simplify language, improve structure, and reduce cognitive load",
                    'user_focus_enhanced': "Better preserve user intent and align with original prompt requirements"
                }

                recommendation = recommendations.get(worst_metric, "Focus on improving this metric")
                print(f"     💡 Recommendation: {recommendation}")

        # Final strategic recommendations
        print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")

        # Check if enhanced metrics reveal different insights than core metrics
        core_avg = {}
        enhanced_avg = {}

        for model in models:
            core_scores = []
            enhanced_scores = []

            for metric in ['actionability', 'coherence', 'domain_specificity', 'technological_specificity']:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]
                    if len(non_zero_values) > 0:
                        core_scores.append(non_zero_values.mean())

            for metric in ['understandability_enhanced', 'user_focus_enhanced']:
                col_name = f"{model}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]
                    if len(non_zero_values) > 0:
                        enhanced_scores.append(non_zero_values.mean() / 10.0)  # Normalize

            if core_scores:
                core_avg[model] = np.mean(core_scores)
            if enhanced_scores:
                enhanced_avg[model] = np.mean(enhanced_scores)

        # Compare core vs enhanced performance
        if core_avg and enhanced_avg:
            print(f"\n   📊 Core vs Enhanced Metrics Gap Analysis:")
            for model in models:
                if model in core_avg and model in enhanced_avg:
                    gap = core_avg[model] - enhanced_avg[model]
                    if abs(gap) > 0.1:
                        if gap > 0:
                            print(f"     {model.upper()}: Core metrics stronger than enhanced ({gap:.3f} difference)")
                            print(f"       → Focus on improving understandability and user alignment")
                        else:
                            print(
                                f"     {model.upper()}: Enhanced metrics stronger than core ({abs(gap):.3f} difference)")
                            print(f"       → Leverage clarity strengths to improve technical implementation")
                    else:
                        print(f"     {model.upper()}: Balanced performance across core and enhanced metrics")


# Usage example with enhanced functionality
if __name__ == "__main__":
    try:
        # Try to find the results file
        possible_files = [
            "./evals_results/final_focused_experiment_results.csv",
            "./evals_results/verified_graph_experiment_results.csv",
            "./evals_results/graph_focused_experiment_results.csv"
        ]

        csv_file = None
        for file_path in possible_files:
            try:
                analyzer = EnhancedMetricsAnalyzer(file_path)
                csv_file = file_path
                break
            except FileNotFoundError:
                continue

        if csv_file is None:
            print("❌ No results file found. Please run the experiment script first.")
            print(f"Looking for: {possible_files}")
        else:
            print(f"✅ Analyzing Enhanced Metrics from: {csv_file}")

            # Generate comprehensive analysis
            analyzer.generate_summary_report()

            # Create enhanced visualizations
            analyzer.create_enhanced_visualizations(save_plots=True)

            # Export enhanced summary statistics
            analyzer.export_enhanced_summary_statistics()

            # Generate enhanced actionable insights
            analyzer.generate_enhanced_actionable_insights()

            print(f"\n🎉 Enhanced Metrics analysis complete!")
            print(f"📊 Analyzed Metrics:")
            print(f"   Core Metrics:")
            print(f"     • Actionability: How implementable are the instructions?")
            print(f"     • Coherence: Logical sequencing and flow")
            print(f"     • Domain Specificity: Relevance to domain context")
            print(f"     • Technological Specificity: Technical implementation detail")
            print(f"   Enhanced Metrics:")
            print(f"     • Understandability: Structural clarity and cognitive load")
            print(f"     • User Focus: Alignment with original user prompt")
            print(f"📁 Check ./evals_results/ for visualizations and detailed statistics")

    except Exception as e:
        print(f"❌ Error during enhanced analysis: {e}")
        import traceback

        traceback.print_exc()