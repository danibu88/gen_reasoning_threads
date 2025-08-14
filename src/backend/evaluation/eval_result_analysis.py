import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class GraphFocusedResultsAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize the analyzer with the graph-focused results CSV"""
        self.df = pd.read_csv(csv_file_path)
        self.setup_plotting()

    def setup_plotting(self):
        """Setup plotting parameters"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)

    def generate_summary_report(self):
        """Generate a graph-focused summary report"""
        print("=" * 80)
        print("GRAPH-FOCUSED EVALUATION RESULTS ANALYSIS")
        print("=" * 80)

        # Basic statistics
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   Total evaluations: {len(self.df)}")
        print(f"   Unique domains: {self.df['domain'].nunique()}")

        # Calculate success rate based on graph data
        successful_graphs = self.df[(self.df['gnn_node_count'] > 0) | (self.df['mcts_node_count'] > 0)]
        success_rate = len(successful_graphs) / len(self.df) * 100
        print(f"   Graph generation success rate: {success_rate:.1f}%")

        # Core analyses
        self.analyze_graph_structures()
        self.analyze_model_performance()
        self.analyze_graph_comparisons()
        self.analyze_subgraph_selections()
        self.analyze_correlations()

    def analyze_graph_structures(self):
        """Analyze graph structure metrics for GNN and MCTS"""
        print(f"\n📊 GRAPH STRUCTURE ANALYSIS")
        print("-" * 50)

        graph_types = ['gnn', 'mcts']

        # Core structure metrics to analyze
        core_metrics = [
            'node_count', 'edge_count', 'average_degree', 'graph_density',
            'clustering_coefficient', 'connected_components', 'largest_component_ratio'
        ]

        for graph_type in graph_types:
            print(f"\n🔸 {graph_type.upper()} Graph Statistics:")

            for metric in core_metrics:
                col_name = f"{graph_type}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    if len(values) > 0 and values.sum() > 0:  # Only show non-zero metrics
                        non_zero_values = values[values > 0]
                        if len(non_zero_values) > 0:
                            print(f"     {metric.replace('_', ' ').title()}:")
                            print(f"       Mean: {non_zero_values.mean():.3f}, Std: {non_zero_values.std():.3f}")
                            print(f"       Range: {non_zero_values.min():.3f} - {non_zero_values.max():.3f}")
                            print(f"       Non-zero samples: {len(non_zero_values)}/{len(values)}")

        # Graph size comparison
        print(f"\n🔸 Graph Size Comparison:")
        gnn_sizes = self.df['gnn_node_count'].dropna()
        mcts_sizes = self.df['mcts_node_count'].dropna()

        if len(gnn_sizes) > 0 and len(mcts_sizes) > 0:
            gnn_non_zero = gnn_sizes[gnn_sizes > 0]
            mcts_non_zero = mcts_sizes[mcts_sizes > 0]

            if len(gnn_non_zero) > 0:
                print(f"     GNN graphs: avg={gnn_non_zero.mean():.1f} nodes, max={gnn_non_zero.max()}")
            if len(mcts_non_zero) > 0:
                print(f"     MCTS graphs: avg={mcts_non_zero.mean():.1f} nodes, max={mcts_non_zero.max()}")

            # Size ratio analysis
            if 'graph_size_ratio' in self.df.columns:
                size_ratios = self.df['graph_size_ratio'].dropna()
                if len(size_ratios) > 0:
                    print(f"     Size ratio (GNN/MCTS): avg={size_ratios.mean():.2f}")

    def analyze_model_performance(self):
        """Analyze model performance comparison"""
        print(f"\n🤖 MODEL PERFORMANCE ANALYSIS")
        print("-" * 50)

        models = ['gnn', 'mcts', 'rmodel']

        # Overall effectiveness comparison
        print(f"\n🏆 Overall Effectiveness Scores:")
        performance_data = {}

        for model in models:
            col_name = f"{model}_overall_effectiveness"
            if col_name in self.df.columns:
                values = self.df[col_name].dropna()
                non_zero_values = values[values > 0]
                if len(non_zero_values) > 0:
                    performance_data[model] = {
                        'mean': non_zero_values.mean(),
                        'std': non_zero_values.std(),
                        'count': len(non_zero_values),
                        'max': non_zero_values.max()
                    }

        # Rank models by performance
        if performance_data:
            ranked_models = sorted(performance_data.items(), key=lambda x: x[1]['mean'], reverse=True)

            for i, (model, stats) in enumerate(ranked_models):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(
                    f"     {rank} {model.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} samples)")

        # Best performing model distribution
        if 'best_performing_model' in self.df.columns:
            best_models = self.df['best_performing_model'].value_counts()
            print(f"\n🎯 Best Model Distribution:")
            for model, count in best_models.items():
                percentage = (count / len(self.df)) * 100
                print(f"     {model}: {count} wins ({percentage:.1f}%)")

        # Performance gap analysis
        if 'performance_gap' in self.df.columns:
            gaps = self.df['performance_gap'].dropna()
            non_zero_gaps = gaps[gaps > 0]
            if len(non_zero_gaps) > 0:
                print(f"\n📊 Performance Gap Analysis:")
                print(f"     Average gap: {non_zero_gaps.mean():.4f}")
                print(f"     Max gap: {non_zero_gaps.max():.4f}")

    def analyze_graph_comparisons(self):
        """Analyze graph-to-graph comparisons"""
        print(f"\n📐 GRAPH COMPARISON ANALYSIS")
        print("-" * 50)

        comparison_metrics = [
            'graph_size_ratio', 'graph_density_ratio', 'graph_connectivity_ratio'
        ]

        for metric in comparison_metrics:
            if metric in self.df.columns:
                values = self.df[metric].dropna()
                # Filter out extreme values (likely division by small numbers)
                filtered_values = values[(values > 0.1) & (values < 10)]

                if len(filtered_values) > 0:
                    print(f"\n🔸 {metric.replace('_', ' ').title()}:")
                    print(f"     Mean: {filtered_values.mean():.3f}")
                    print(f"     Median: {filtered_values.median():.3f}")
                    print(f"     Range: {filtered_values.min():.3f} - {filtered_values.max():.3f}")

                    # Interpretation
                    if 'size' in metric:
                        if filtered_values.mean() > 1.5:
                            print(f"     → GNN graphs are typically larger than MCTS graphs")
                        elif filtered_values.mean() < 0.67:
                            print(f"     → MCTS graphs are typically larger than GNN graphs")
                        else:
                            print(f"     → Similar graph sizes between GNN and MCTS")

    def analyze_subgraph_selections(self):
        """Analyze which subgraphs were selected for each model"""
        print(f"\n🎯 SUBGRAPH SELECTION ANALYSIS")
        print("-" * 50)

        # GNN subgraph sources
        if 'gnn_subgraph_source' in self.df.columns:
            gnn_sources = self.df['gnn_subgraph_source'].value_counts()
            print(f"\n🔸 GNN Subgraph Sources:")
            for source, count in gnn_sources.items():
                percentage = (count / len(self.df)) * 100
                print(f"     {source}: {count} ({percentage:.1f}%)")

        # MCTS subgraph sources
        if 'mcts_subgraph_source' in self.df.columns:
            mcts_sources = self.df['mcts_subgraph_source'].value_counts()
            print(f"\n🔸 MCTS Subgraph Sources:")
            for source, count in mcts_sources.items():
                percentage = (count / len(self.df)) * 100
                print(f"     {source}: {count} ({percentage:.1f}%)")

        # Available subgraphs analysis
        if 'available_subgraphs' in self.df.columns:
            print(f"\n🔸 Available Subgraph Patterns:")
            # Count different patterns of available subgraphs
            patterns = self.df['available_subgraphs'].value_counts().head(5)
            for pattern, count in patterns.items():
                print(f"     {count} cases: {pattern}")

    def analyze_correlations(self):
        """Analyze correlations between graph metrics and model performance"""
        print(f"\n📈 CORRELATION ANALYSIS")
        print("-" * 50)

        # Key correlations to examine
        correlations_to_check = [
            # Graph size vs performance
            ('gnn_node_count', 'gnn_overall_effectiveness'),
            ('mcts_node_count', 'mcts_overall_effectiveness'),

            # Graph structure vs performance
            ('gnn_clustering_coefficient', 'gnn_overall_effectiveness'),
            ('mcts_clustering_coefficient', 'mcts_overall_effectiveness'),
            ('gnn_graph_density', 'gnn_overall_effectiveness'),
            ('mcts_graph_density', 'mcts_overall_effectiveness'),

            # Graph complexity vs prompt complexity
            ('prompt_complexity_score', 'gnn_node_count'),
            ('prompt_complexity_score', 'mcts_node_count'),

            # Cross-model comparisons
            ('gnn_node_count', 'mcts_node_count'),
            ('gnn_overall_effectiveness', 'mcts_overall_effectiveness'),
        ]

        significant_correlations = []

        for var1, var2 in correlations_to_check:
            if var1 in self.df.columns and var2 in self.df.columns:
                # Get non-zero, non-null values
                df_clean = self.df[[var1, var2]].dropna()
                df_clean = df_clean[(df_clean[var1] > 0) & (df_clean[var2] > 0)]

                if len(df_clean) > 10:  # Need sufficient data points
                    corr, p_value = stats.pearsonr(df_clean[var1], df_clean[var2])

                    if abs(corr) > 0.3 and p_value < 0.05:  # Significant correlation
                        significant_correlations.append((var1, var2, corr, p_value, len(df_clean)))

        if significant_correlations:
            print(f"\n🔸 Significant Correlations Found:")
            for var1, var2, corr, p_val, n_samples in sorted(significant_correlations,
                                                             key=lambda x: abs(x[2]), reverse=True):
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "weak"
                print(f"     {var1} ↔ {var2}")
                print(f"       {strength} {direction}: r={corr:.3f}, p={p_val:.3f} (n={n_samples})")
        else:
            print(f"     No significant correlations found (threshold: |r| > 0.3, p < 0.05)")

    def create_visualizations(self, save_plots=True):
        """Create graph-focused visualizations"""
        print(f"\n📊 GENERATING GRAPH-FOCUSED VISUALIZATIONS...")

        fig_count = 1

        # 1. Model Performance Comparison
        models = ['gnn', 'mcts', 'rmodel']
        effectiveness_data = []

        for model in models:
            col_name = f"{model}_overall_effectiveness"
            if col_name in self.df.columns:
                values = self.df[col_name].dropna()
                non_zero_values = values[values > 0]
                for val in non_zero_values:
                    effectiveness_data.append({'Model': model.upper(), 'Effectiveness': val})

        if effectiveness_data:
            plt.figure(fig_count, figsize=(10, 6))
            fig_count += 1

            perf_df = pd.DataFrame(effectiveness_data)
            sns.boxplot(data=perf_df, x='Model', y='Effectiveness')
            plt.title('Model Performance Comparison - Overall Effectiveness')
            plt.ylabel('Effectiveness Score')
            plt.grid(True, alpha=0.3)

            if save_plots:
                plt.savefig('./evals_results/model_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 2. Graph Structure Comparison
        structure_data = []
        for graph_type in ['gnn', 'mcts']:
            for metric in ['node_count', 'edge_count', 'clustering_coefficient']:
                col_name = f"{graph_type}_{metric}"
                if col_name in self.df.columns:
                    values = self.df[col_name].dropna()
                    non_zero_values = values[values > 0]
                    for val in non_zero_values:
                        structure_data.append({
                            'Graph Type': graph_type.upper(),
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': val
                        })

        if structure_data:
            structure_df = pd.DataFrame(structure_data)

            metrics = structure_df['Metric'].unique()
            n_metrics = len(metrics)

            if n_metrics > 0:
                fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(6 * min(n_metrics, 3), 5))
                if n_metrics == 1:
                    axes = [axes]
                elif n_metrics == 2:
                    axes = axes

                for i, metric in enumerate(metrics[:3]):  # Limit to 3 subplots
                    metric_data = structure_df[structure_df['Metric'] == metric]
                    if not metric_data.empty and i < len(axes):
                        sns.boxplot(data=metric_data, x='Graph Type', y='Value', ax=axes[i])
                        axes[i].set_title(f'{metric} by Graph Type')
                        axes[i].grid(True, alpha=0.3)

                plt.tight_layout()
                if save_plots:
                    plt.savefig('./evals_results/graph_structure_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()

        # 3. Graph Size vs Performance Scatter Plot
        gnn_data = self.df[['gnn_node_count', 'gnn_overall_effectiveness']].dropna()
        mcts_data = self.df[['mcts_node_count', 'mcts_overall_effectiveness']].dropna()

        # Filter to non-zero values
        gnn_data = gnn_data[(gnn_data['gnn_node_count'] > 0) & (gnn_data['gnn_overall_effectiveness'] > 0)]
        mcts_data = mcts_data[(mcts_data['mcts_node_count'] > 0) & (mcts_data['mcts_overall_effectiveness'] > 0)]

        if len(gnn_data) > 0 or len(mcts_data) > 0:
            plt.figure(fig_count, figsize=(12, 5))
            fig_count += 1

            if len(gnn_data) > 0 and len(mcts_data) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
                ax2 = None

            if len(gnn_data) > 0:
                ax1.scatter(gnn_data['gnn_node_count'], gnn_data['gnn_overall_effectiveness'],
                            alpha=0.6, color='blue', label='GNN')
                ax1.set_xlabel('GNN Node Count')
                ax1.set_ylabel('GNN Effectiveness')
                ax1.set_title('GNN: Graph Size vs Performance')
                ax1.grid(True, alpha=0.3)

                # Add trend line
                z = np.polyfit(gnn_data['gnn_node_count'], gnn_data['gnn_overall_effectiveness'], 1)
                p = np.poly1d(z)
                ax1.plot(gnn_data['gnn_node_count'], p(gnn_data['gnn_node_count']), "r--", alpha=0.8)

            if len(mcts_data) > 0 and ax2 is not None:
                ax2.scatter(mcts_data['mcts_node_count'], mcts_data['mcts_overall_effectiveness'],
                            alpha=0.6, color='green', label='MCTS')
                ax2.set_xlabel('MCTS Node Count')
                ax2.set_ylabel('MCTS Effectiveness')
                ax2.set_title('MCTS: Graph Size vs Performance')
                ax2.grid(True, alpha=0.3)

                # Add trend line
                z = np.polyfit(mcts_data['mcts_node_count'], mcts_data['mcts_overall_effectiveness'], 1)
                p = np.poly1d(z)
                ax2.plot(mcts_data['mcts_node_count'], p(mcts_data['mcts_node_count']), "r--", alpha=0.8)

            plt.tight_layout()
            if save_plots:
                plt.savefig('./evals_results/graph_size_vs_performance.png', dpi=300, bbox_inches='tight')
            plt.show()

        print(f"   Generated {fig_count - 1} visualization(s)")
        if save_plots:
            print(f"   Plots saved to ./evals_results/")

    def export_summary_statistics(self, filename="graph_focused_summary_statistics.csv"):
        """Export graph-focused summary statistics"""
        summary_stats = {}

        # Focus on graph and performance metrics
        key_metrics = [
            'gnn_node_count', 'gnn_edge_count', 'gnn_clustering_coefficient', 'gnn_graph_density',
            'mcts_node_count', 'mcts_edge_count', 'mcts_clustering_coefficient', 'mcts_graph_density',
            'gnn_overall_effectiveness', 'mcts_overall_effectiveness', 'rmodel_overall_effectiveness',
            'graph_size_ratio', 'graph_density_ratio', 'performance_gap'
        ]

        for col in key_metrics:
            if col in self.df.columns:
                values = self.df[col].dropna()
                non_zero_values = values[values > 0] if 'ratio' not in col else values

                if len(non_zero_values) > 0:
                    summary_stats[col] = {
                        'count': len(non_zero_values),
                        'mean': non_zero_values.mean(),
                        'std': non_zero_values.std(),
                        'min': non_zero_values.min(),
                        'q25': non_zero_values.quantile(0.25),
                        'median': non_zero_values.median(),
                        'q75': non_zero_values.quantile(0.75),
                        'max': non_zero_values.max()
                    }

        # Convert to DataFrame and save
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats).T
            summary_df.to_csv(f"./evals_results/{filename}")
            print(f"   Summary statistics exported to ./evals_results/{filename}")


# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    try:
        # Try the new verified filename first, then fall back to old names
        possible_files = [
            "./evals_results/verified_graph_experiment_results.csv",
            "./evals_results/graph_focused_experiment_results.csv",
            "./evals_results/comprehensive_experiment_results.csv"
        ]

        csv_file = None
        for file_path in possible_files:
            try:
                analyzer = GraphFocusedResultsAnalyzer(file_path)
                csv_file = file_path
                break
            except FileNotFoundError:
                continue

        if csv_file is None:
            print("❌ No results file found. Please run the experiment script first.")
            print(f"Looking for: {possible_files}")
        else:
            print(f"✅ Analyzing results from: {csv_file}")

            # Generate comprehensive analysis
            analyzer.generate_summary_report()

            # Create visualizations
            analyzer.create_visualizations(save_plots=True)

            # Export summary statistics
            analyzer.export_summary_statistics()

            print(f"\n🎉 Graph-focused analysis complete! Check ./evals_results/ for outputs.")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        print("Please check that the results file exists and has the expected format.")